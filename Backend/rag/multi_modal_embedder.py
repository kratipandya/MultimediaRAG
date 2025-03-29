#pylint: disable=all
from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import os
import cv2
import faiss
import torch
import librosa
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModel,
    CLIPProcessor,
    CLIPModel,
    WhisperProcessor,
    WhisperForConditionalGeneration
)
from pytubefix import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

@dataclass
class TextMetadata:
    id: str
    title: str = ""
    source: str = ""
    text: str = ""
    score: float = 0.0


@dataclass
class ImageMetadata:
    id: str
    path: str
    caption: str = ""
    source: str = ""
    score: float = 0.0


@dataclass
class VideoMetadata:
    video_id: str
    title: str = ""
    description: str = ""
    author: str = ""
    duration_seconds: int = 0
    transcript: str = ""
    has_transcript: bool = False
    error: str = ""
    score: float = 0.0


@dataclass
class AudioMetadata:
    id: str
    path: str
    duration_seconds: float = 0
    sample_rate: int = 0
    transcript: str = ""
    source: str = ""
    error: str = ""
    score: float = 0.0


@dataclass
class MultimodalEmbedder:
    # Model names with sensible defaults
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    whisper_model_name: str = "openai/whisper-tiny.en"
    index_directory: str = "faiss"

    # Fields initialized in __post_init__
    text_dim:  int = field(init=False)
    image_dim: int = field(init=False)
    video_dim: int = field(init=False)
    audio_dim: int = field(init=False)

    text_index:  Any = field(init=False, default=None)
    image_index: Any = field(init=False, default=None)
    video_index: Any = field(init=False, default=None)
    audio_index: Any = field(init=False, default=None)

    text_tokenizer: Any = field(init=False, default=None)
    text_model: Any = field(init=False, default=None)
    clip_processor: Any = field(init=False, default=None)
    clip_model: Any = field(init=False, default=None)
    whisper_processor: Any = field(init=False, default=None)
    whisper_model: Any = field(init=False, default=None)

    device: Any = field(init=False, default=None)

    text_metadata: List[TextMetadata] = field(default_factory=list)
    image_metadata: List[ImageMetadata] = field(default_factory=list)
    video_metadata: List[VideoMetadata] = field(default_factory=list)
    audio_metadata: List[AudioMetadata] = field(default_factory=list)

    text_to_image_projection: Any = field(init=False, default=None)
    image_to_text_projection: Any = field(init=False, default=None)

    def __post_init__(self):
        """Initialize the embedder with models and vectors"""
        # Initialize dimensions
        self.text_dim  = 384  # Dimension of text embeddings
        self.audio_dim = 384  # Same as text embeddings for transcriptions
        self.image_dim = 512  # Dimension of CLIP visual embeddings
        self.video_dim = 512  # Same as image dimension

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize models
        logger.info("Loading text embedding model...")
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        self.text_model = AutoModel.from_pretrained(self.text_model_name)
        self.text_model.to(self.device)

        logger.info("Loading image/video embedding model...")
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
        self.clip_model.to(self.device)

        logger.info("Loading audio transcription model...")
        self.whisper_processor = WhisperProcessor.from_pretrained(self.whisper_model_name)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(self.whisper_model_name)
        self.whisper_model.to(self.device)

        # Add projection layers to map between embedding spaces
        self.text_to_image_projection = torch.nn.Linear(self.text_dim, self.image_dim)
        self.text_to_image_projection.to(self.device)

        self.image_to_text_projection = torch.nn.Linear(self.image_dim, self.text_dim)
        self.image_to_text_projection.to(self.device)

        # Create FAISS indices
        self.setup_indices()

    def setup_indices(self):
        """Create separate FAISS indices for each modality"""
        # Using IndexFlatIP for cosine similarity (inner product on normalized vectors)
        self.text_index  = faiss.IndexFlatIP(self.text_dim)
        self.image_index = faiss.IndexFlatIP(self.image_dim)
        self.video_index = faiss.IndexFlatIP(self.video_dim)
        self.audio_index = faiss.IndexFlatIP(self.audio_dim)

    def search(self, query: Dict[str, Any], k: int = 5) -> dict:
        """
        Search indexed content based on multiple query types (text, image, audio).
        Each query type searches across all modalities, and the best results are returned.

        Args:
            query: Dictionary containing query information:
                - text: Text query string (optional)
                - image: Path to image file for image query (optional)
                - audio: Path to audio file for audio query (optional)
            k: Number of results to return per modality

        Returns:
            dict: Dict with results organized by modality (text, image, video, audio).
        """
        # Generate embeddings for each query type if available
        embeddings = {}
        # Initialize results
        results = {"audio_transcript": None}

        # Process text query
        if query_text := query.get("text"):
            logger.info(f"Processing text query: {query_text[:50]}...")
            embeddings["text"] = self.get_text_embedding(query_text)

            # Also get text-to-image projection for image search
            embeddings["text_to_image"] = self.get_text_to_image_embedding(embeddings["text"])

        # Process image query
        if image_path := query.get("image"):
            logger.info(f"Processing image query: {image_path}")
            try:
                embeddings["image"] = self.get_image_embedding(image_path)
                embeddings["image_to_text"] = self.get_image_to_text_embedding(image_path)
            except Exception as e:
                logger.error(f"Error processing image query: {e}")

        # Process audio query
        if audio_path := query.get("audio"):
            logger.info(f"Processing audio query: {audio_path}")
            try:
                embedding, audio_metadata = self.get_audio_embedding(audio_path)
                embeddings["audio"] = embedding
                results["audio_transcript"] = audio_metadata.transcript
                logger.info(f"Audio transcript: {audio_metadata.transcript[:50]}...")
            except Exception as e:
                logger.error(f"Error processing audio query: {e}")

        # Initialize best results for each modality
        best_results = {
            "text": [],
            "image": [],
            "video": [],
            "audio": []
        }

        # Search each modality with all available embeddings
        # and keep track of the best results

        # 1. Search text index with all compatible embeddings
        if self.text_index.ntotal > 0:
            for embed_type, embed in embeddings.items():
                if embed_type in ["text", "image_to_text", "audio"]:
                    # These embedding types are compatible with text search
                    search_results = self.search_modality(embed, self.text_index, self.text_metadata, k)
                    best_results["text"] = self._merge_best_results(best_results["text"], search_results, k)

        # 2. Search image index with all compatible embeddings
        if self.image_index.ntotal > 0:
            for embed_type, embed in embeddings.items():
                if embed_type in ["image", "text_to_image"]:
                    # These embedding types are compatible with image search
                    search_results = self.search_modality(embed, self.image_index, self.image_metadata, k)
                    best_results["image"] = self._merge_best_results(best_results["image"], search_results, k)

        # 3. Search video index with all compatible embeddings
        if self.video_index.ntotal > 0:
            for embed_type, embed in embeddings.items():
                if embed_type in ["image", "text_to_image"]:
                    # These embedding types are compatible with video search
                    search_results = self.search_modality(embed, self.video_index, self.video_metadata, k)
                    best_results["video"] = self._merge_best_results(best_results["video"], search_results, k)

        # 4. Search audio index with all compatible embeddings
        if self.audio_index.ntotal > 0:
            for embed_type, embed in embeddings.items():
                if embed_type in ["audio", "text", "image_to_text"]:
                    # These embedding types are compatible with audio search
                    search_results = self.search_modality(embed, self.audio_index, self.audio_metadata, k)
                    best_results["audio"] = self._merge_best_results(best_results["audio"], search_results, k)

        # Update results with best matches
        results.update(best_results)

        # Log result counts
        for key, value in results.items():
            if isinstance(value, list):
                logger.info(f"Found {len(value)} results for {key}")

        return results

    def _merge_best_results(self, existing_results, new_results, k):
        """
        Merge two lists of results, keeping only the k items with the highest scores.

        Args:
            existing_results: Current list of results
            new_results: New results to merge
            k: Maximum number of results to keep

        Returns:
            Merged list with the top k results
        """
        # Create a dictionary to track best scores for each unique item
        merged = {}

        # Add existing results to the dictionary
        for item in existing_results:
            item_id = item.get("id", "") or item.get("video_id", "")
            merged[item_id] = item

        # Add or update with new results if they have better scores
        for item in new_results:
            item_id = item.get("id", "") or item.get("video_id", "")
            if item_id not in merged or item.get("score", 0) > merged[item_id].get("score", 0):
                merged[item_id] = item

        # Sort by score (highest first) and take top k
        sorted_results = sorted(merged.values(), key=lambda x: x.get("score", 0), reverse=True)
        return sorted_results[:k]

    def _perform_search(self, text_embedding=None, image_embedding=None,
                        audio_embedding=None, image_for_text_embedding=None, k=5):
        """
        Helper method to perform search across all modalities based on available embeddings.
        Centralizes the search logic for better consistency.
        """
        results = {}

        # Search text index if we have any compatible embedding
        if any(emb is not None for emb in [text_embedding, audio_embedding, image_for_text_embedding]):
            # Determine which embedding to use (prioritize in this order)
            if text_embedding is not None:
                search_embedding = text_embedding
            elif image_for_text_embedding is not None:
                search_embedding = image_for_text_embedding
            else:
                search_embedding = audio_embedding

            results["text"] = self.search_modality(search_embedding, self.text_index, self.text_metadata, k)

        # Search image index
        if text_embedding is not None or image_embedding is not None:
            # For text-to-image search, convert text embedding to image space
            if text_embedding is not None and image_embedding is None:
                search_embedding = self.get_text_to_image_embedding(text_embedding)
            else:
                search_embedding = image_embedding

            results["image"] = self.search_modality(search_embedding, self.image_index, self.image_metadata, k)

        # Search video index - similar approach to images
        if text_embedding is not None or image_embedding is not None:
            # For text-to-video search, same as text-to-image
            if text_embedding is not None and image_embedding is None:
                search_embedding = self.get_text_to_image_embedding(text_embedding)
            else:
                search_embedding = image_embedding

            results["video"] = self.search_modality(search_embedding, self.video_index, self.video_metadata, k)

        # Search audio index
        # Always use audio embedding directly when available (no projection needed)
        if audio_embedding is not None:
            results["audio"] = self.search_modality(audio_embedding, self.audio_index, self.audio_metadata, k)
        # Fall back to text-based search if no audio embedding
        elif text_embedding is not None:
            results["audio"] = self.search_modality(text_embedding, self.audio_index, self.audio_metadata, k)
        # Use image-to-text if that's all we have
        elif image_for_text_embedding is not None:
            results["audio"] = self.search_modality(image_for_text_embedding, self.audio_index, self.audio_metadata, k)

        return results

    def search_modality(self, query_embedding: np.ndarray, modality_index, modality_metadata, k: int = 5):
        """
        Generic search method for any modality.

        Args:
            query_embedding: The query embedding vector
            modality_index: FAISS index for the modality
            modality_metadata: List of metadata for the modality
            k: Number of results to return

        Returns:
            List of results with scores
        """
        if modality_index.ntotal == 0:
            # No items in this index
            return []

        scores, indices = modality_index.search(query_embedding, min(k, modality_index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(modality_metadata):
                result_dict = vars(modality_metadata[idx])
                result_dict["score"] = float(score)
                results.append(result_dict)

        return results

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Generate embeddings for text using the text embedding model"""
        inputs = self.text_tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.text_model(**inputs)

        # Use mean pooling to get a single vector per text
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def get_text_to_image_embedding(self, text_input: Union[str, np.ndarray]) -> np.ndarray:
        """
        Generate embeddings that align with the image embedding space.
        This can accept either a text string or an existing text embedding.
        """
        if isinstance(text_input, str):
            # Process raw text through CLIP
            inputs = self.clip_processor(text=text_input, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                embeddings = text_features.cpu().numpy()
        else:
            # Apply projection to existing text embedding
            tensor_embedding = torch.tensor(text_input).to(self.device)
            with torch.no_grad():
                projected = self.text_to_image_projection(tensor_embedding)
                embeddings = projected.cpu().numpy()

        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def get_image_to_text_embedding(self, image_path: str) -> np.ndarray:
        """
        Generate image embeddings that align well with the text embedding space.
        This is specifically designed for image-to-text search.

        Args:
            image_path: Path to the image file

        Returns:
            Embedding vector aligned with text space
        """
        try:
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # Get image features from CLIP
                image_features = self.clip_model.get_image_features(**inputs)

                # Apply projection to align with text embedding space
                projected_features = self.image_to_text_projection(image_features)
                embeddings = projected_features.cpu().numpy()

            # Normalize for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings

        except Exception as e:
            logger.error(f"Error processing image for text search: {e}")
            return np.zeros((1, self.text_dim))

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """Generate embeddings for an image using CLIP"""
        try:
            # Load and process image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)

            # Get image embeddings and normalize
            embeddings = outputs.cpu().numpy()
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return np.zeros((1, self.image_dim))

    def get_transcript_from_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file to text using Whisper model.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        try:
            audio_array, sample_rate = librosa.load(audio_path, sr=16000)
            inputs = self.whisper_processor(
                audio_array,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.whisper_model.generate(**inputs, max_length=448)
                transcript = self.whisper_processor.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )[0]

            return transcript
        except Exception as e:
            logger.error(f"Error transcribing audio {audio_path}: {e}")
            return ""

    def get_youtube_video_embedding(self, video_id: str, num_frames: int = 5) -> Tuple[np.ndarray, VideoMetadata]:
        """ Generate embeddings for a YouTube video by:
            1. Downloading the video
            2. Extracting frames at regular intervals
            3. Getting CLIP embeddings for each frame
            4. Averaging them into a single embedding
            5. Getting transcript if available
        """
        # Create a temporary directory for the video
        video_folderpath = os.path.join("embedding_data", "video")
        os.makedirs(video_folderpath, exist_ok=True)
        # Initialize YouTube object regardless of whether we need to download
        video_path = os.path.join(video_folderpath, f"{video_id}.mp4")
        video_name = f"https://www.youtube.com/watch?v={video_id}"
        # print(f"{video_name = }")
        yt = YouTube(str(video_name))

        # Store metadata
        metadata = VideoMetadata(
            video_id=video_id,
            title=yt.title,
            description=yt.description,
            author=yt.author,
            duration_seconds=yt.length
        )

        # Download the video if not already downloaded
        if not os.path.exists(video_path):
            logger.info(f"Downloading youtube video {video_id}")
            stream = yt.streams.filter(progressive=True, file_extension='.mp4').first()
            stream.download(output_path=video_folderpath, filename=f"{video_id}.mp4")

        # Get transcript if available
        transcript_text = ""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([item['text'] for item in transcript])
            metadata.transcript = transcript_text
            metadata.has_transcript = True
        except Exception as e:
            logger.warning(f"No transcript available for {video_id}: {e}")

        # Process the video to extract frames
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Extract frames at regular intervals
        frame_embeddings = []
        interval = total_frames // (num_frames + 1)

        for i in range(1, num_frames + 1):
            frame_position = i * interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            ret, frame = cap.read()

            if ret:
                # Convert frame to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Get CLIP embedding for the frame
                inputs = self.clip_processor(images=frame, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.clip_model.get_image_features(**inputs)

                frame_embedding = outputs.cpu().numpy()
                frame_embeddings.append(frame_embedding)

        cap.release()

        # Process embeddings
        if frame_embeddings:
            # Average the frame embeddings
            visual_embedding = np.mean(np.vstack(frame_embeddings), axis=0, keepdims=True)
            # Normalize the embedding
            final_embedding = visual_embedding / np.linalg.norm(visual_embedding)
            return final_embedding, metadata

        # Fallback if no frames were processed
        return np.zeros((1, self.video_dim)), metadata

    def get_audio_embedding(self, audio_path: str) -> Tuple[np.ndarray, AudioMetadata]:
        """
        Generate embeddings for an audio file by:
        1. Loading the audio file
        2. Transcribing it using Whisper
        3. Getting text embeddings for the transcription

        Args:
            audio_path: Path to the audio file

        Returns:
            Tuple of audio embedding and metadata
        """
        try:
            # Create metadata object
            metadata = AudioMetadata(
                id=os.path.basename(audio_path),
                path=audio_path
            )

            # Load audio using librosa
            audio_array, sample_rate = librosa.load(audio_path, sr=16000)
            metadata.sample_rate = sample_rate
            metadata.duration_seconds = len(audio_array) / sample_rate

            # Get transcript
            transcript = self.get_transcript_from_audio(audio_path)
            metadata.transcript = transcript

            # Get text embedding of the transcription
            text_embedding = self.get_text_embedding(transcript)

            return text_embedding, metadata

        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {e}")
            error_metadata = AudioMetadata(
                id=os.path.basename(audio_path),
                path=audio_path,
                error=str(e)
            )
            return np.zeros((1, self.audio_dim)), error_metadata

    def index_documents(self, documents: List[Dict[str, str]]):
        """Index text documents with metadata"""
        for doc in tqdm(documents, desc="Indexing documents"):
            embedding = self.get_text_embedding(doc["text"])
            self.text_index.add(embedding)

            # Create metadata and add truncated text
            text_content = doc.get("text", "")
            truncated_text = text_content[:300] + "..." if len(text_content) > 300 else text_content

            metadata = TextMetadata(
                id=doc.get("id", str(len(self.text_metadata))),
                title=doc.get("title", ""),
                source=doc.get("source", ""),
                text=truncated_text
            )
            self.text_metadata.append(metadata)

    def index_images(self, image_paths: List[Dict[str, str]]):
        """Index images with metadata"""
        for img in tqdm(image_paths, desc="Indexing images"):
            embedding = self.get_image_embedding(img["path"])
            self.image_index.add(embedding)

            metadata = ImageMetadata(
                id=img.get("id", str(len(self.image_metadata))),
                path=img["path"],
                caption=img.get("caption", ""),
                source=img.get("source", "")
            )
            self.image_metadata.append(metadata)

    def index_audio_files(self, audio_paths: List[str]):
        """Index audio files with metadata"""
        for audio_path in tqdm(audio_paths, desc="Indexing audio files"):
            embedding, metadata = self.get_audio_embedding(audio_path)
            self.audio_index.add(embedding)
            self.audio_metadata.append(metadata)

    def index_youtube_videos(self, video_ids: List[str]):
        """Index YouTube videos with metadata"""
        for video_id in tqdm(video_ids, desc="Indexing videos"):
            embedding, metadata = self.get_youtube_video_embedding(video_id)
            self.video_index.add(embedding)
            self.video_metadata.append(metadata)

    # ========= Index Saving and Loading Functions =========

    def save_indices(self, directory: Optional[str] = None):
        """
        Save all FAISS indices and metadata to disk.

        Args:
            directory: Directory to save indices and metadata.
                      If None, uses the default directory specified in __init__.
        """
        directory = directory or self.index_directory
        logger.info(f"Saving all indices to {directory}")
        os.makedirs(directory, exist_ok=True)

        # Save FAISS indices
        faiss.write_index(self.text_index, os.path.join(directory, "text_index.faiss"))
        faiss.write_index(self.image_index, os.path.join(directory, "image_index.faiss"))
        faiss.write_index(self.audio_index, os.path.join(directory, "audio_index.faiss"))
        faiss.write_index(self.video_index, os.path.join(directory, "video_index.faiss"))

        # Convert dataclasses to dicts for serialization
        text_metadata_dicts = [vars(item) for item in self.text_metadata]
        image_metadata_dicts = [vars(item) for item in self.image_metadata]
        audio_metadata_dicts = [vars(item) for item in self.audio_metadata]
        video_metadata_dicts = [vars(item) for item in self.video_metadata]

        # Save metadata
        pd.DataFrame(text_metadata_dicts).to_json(os.path.join(directory, "text_metadata.json"), orient="records")
        pd.DataFrame(image_metadata_dicts).to_json(os.path.join(directory, "image_metadata.json"), orient="records")
        pd.DataFrame(audio_metadata_dicts).to_json(os.path.join(directory, "audio_metadata.json"), orient="records")
        pd.DataFrame(video_metadata_dicts).to_json(os.path.join(directory, "video_metadata.json"), orient="records")

    def save_indices_images(self):
        """Save only image indices and metadata"""
        directory = self.index_directory
        logger.info(f"Saving image indices to {directory}")
        os.makedirs(directory, exist_ok=True)

        faiss.write_index(self.image_index, os.path.join(directory, "image_index.faiss"))
        image_metadata_dicts = [vars(item) for item in self.image_metadata]
        pd.DataFrame(image_metadata_dicts).to_json(os.path.join(directory, "image_metadata.json"), orient="records")

    def save_indices_audio(self):
        """Save only audio indices and metadata"""
        directory = self.index_directory
        logger.info(f"Saving audio indices to {directory}")
        os.makedirs(directory, exist_ok=True)

        faiss.write_index(self.audio_index, os.path.join(directory, "audio_index.faiss"))
        audio_metadata_dicts = [vars(item) for item in self.audio_metadata]
        pd.DataFrame(audio_metadata_dicts).to_json(os.path.join(directory, "audio_metadata.json"), orient="records")

    def save_indices_video(self):
        """Save only video indices and metadata"""
        directory = self.index_directory
        logger.info(f"Saving video indices to {directory}")
        os.makedirs(directory, exist_ok=True)

        faiss.write_index(self.video_index, os.path.join(directory, "video_index.faiss"))
        video_metadata_dicts = [vars(item) for item in self.video_metadata]
        pd.DataFrame(video_metadata_dicts).to_json(os.path.join(directory, "video_metadata.json"), orient="records")

    def save_indices_text(self):
        """Save only text indices and metadata"""
        directory = self.index_directory
        logger.info(f"Saving text indices to {directory}")
        os.makedirs(directory, exist_ok=True)

        faiss.write_index(self.text_index, os.path.join(directory, "text_index.faiss"))
        text_metadata_dicts = [vars(item) for item in self.text_metadata]
        pd.DataFrame(text_metadata_dicts).to_json(os.path.join(directory, "text_metadata.json"), orient="records")

    def load_indices(self, directory: Optional[str] = None):
        """
        Load FAISS indices and metadata from disk.

        Args:
            directory: Directory to load indices from.
                      If None, uses the default directory specified in __init__.
        """
        directory = directory or self.index_directory
        dir_path = os.path.dirname(os.path.realpath(__file__))
        full_dir_path = os.path.join(dir_path, directory)

        logger.info(f"Loading indices from {full_dir_path}")

        try:
            # Load FAISS indices
            self.text_index = faiss.read_index(os.path.join(full_dir_path, "text_index.faiss"))
            self.image_index = faiss.read_index(os.path.join(full_dir_path, "image_index.faiss"))
            self.video_index = faiss.read_index(os.path.join(full_dir_path, "video_index.faiss"))
            self.audio_index = faiss.read_index(os.path.join(full_dir_path, "audio_index.faiss"))

            # Load metadata and convert to dataclasses
            text_data = pd.read_json(os.path.join(full_dir_path, "text_metadata.json"), orient="records").to_dict("records")
            self.text_metadata = [TextMetadata(**item) for item in text_data]

            image_data = pd.read_json(os.path.join(full_dir_path, "image_metadata.json"), orient="records").to_dict("records")
            self.image_metadata = [ImageMetadata(**item) for item in image_data]

            video_data = pd.read_json(os.path.join(full_dir_path, "video_metadata.json"), orient="records").to_dict("records")
            self.video_metadata = [VideoMetadata(**item) for item in video_data]

            audio_data = pd.read_json(os.path.join(full_dir_path, "audio_metadata.json"), orient="records").to_dict("records")
            self.audio_metadata = [AudioMetadata(**item) for item in audio_data]

            logger.info(f"Successfully loaded indices: text({self.text_index.ntotal}), image({self.image_index.ntotal}), "
                      f"video({self.video_index.ntotal}), audio({self.audio_index.ntotal})")
        except Exception as e:
            logger.error(f"Error loading indices: {e}")
            # Initialize empty indices as fallback
            self.setup_indices()
