""" Build Multi-Modal Embeddings for faiss"""
# pylint: disable=all
import os
import glob
import re
from multi_modal_embedder import MultimodalEmbedder


def embed_videos():
    filepath = glob.glob("embedding_data/**/*.mp4")
    video_filenames = [os.path.basename(x) for x in filepath]
    print(f"Found {len(video_filenames) = } videos...")
    pass

    embedder = MultimodalEmbedder()

    video_ids = [x.split(".mp4")[0] for x in video_filenames]
    embedder.index_youtube_videos(video_ids)
    embedder.save_indices_video()

def embed_audios():
    filepath = glob.glob("embedding_data/audio/*")
    print("Processing audio files..")
    print(*filepath, sep="\n")

    embedder = MultimodalEmbedder()
    embedder.index_audio_files(filepath)
    embedder.save_indices_audio()

def embed_images():
    filepaths = glob.glob("embedding_data/image/*")
    print("Processing image files..")
    print(*filepaths, sep="\n")

    metadata_images = [{
            "id": os.path.basename(filepath),
            "path": filepath,
            "caption": os.path.basename(filepath),
            "source" : "Unknown",
        } for filepath in filepaths]

    embedder = MultimodalEmbedder()
    embedder.index_images(metadata_images)
    embedder.save_indices_images()

if __name__=="__main__":
    embed_audios()
    embed_videos()
    embed_images()
    # embed_text() # Generated using notebook
