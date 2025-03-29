#pylint: disable=all
import json
import logging
import glob
import traceback
import torch
from typing import Dict, Optional
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .multi_modal_embedder import MultimodalEmbedder
from .prompt import get_prompt

# Set up logging
logger = logging.getLogger(__name__)

class ArXivRAGSystem:
    """
    An enhanced RAG (Retrieval-Augmented Generation) system for textual context
    in ArXiv research paper and multi-media scientific content with
    improved prompt engineering and error handling.
    """

    def __init__(self, _config: Dict[str, str]=None):
        """
        Initialize the RAG system with the given configuration.

        Args:
            config: A dictionary containing configuration parameters:
                - faiss_index_path: Path to the FAISS index file
                - mapping_path: Path to the document mapping file
                - projection_path: Path to the projection model file
                - image_folder: Path to the folder containing images
                - use_mm_embedder: Whether to use the multimodal embedder (default: False)
        """
        #self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Initialize components
        # if self.config.get("use_mm_embedder", False):
        #     logger.info("Using Multimodal Embedder")
        #     self.mm_embedder = self._initialize_mm_embedder()
        #     self.retriever = None
        # else:
        #     logger.info("Using ArXiv Retriever")
        #     self.mm_embedder = None
        #     self.retriever = self._initialize_retriever()

        self.llm = self._initialize_llm()
        # self.qa_chain = self._create_qa_chain()

    def _initialize_mm_embedder(self) -> MultimodalEmbedder:
        """Initialize the multimodal embedder."""
        try:
            embedder = MultimodalEmbedder()
            embedder.load_indices()
            logger.info("Initialized Multimodal Embedder")
            return embedder
        except Exception as e:
            logger.error(f"Failed to initialize multimodal embedder: {e}")
            raise

    def _initialize_llm(self):
        """Initialize the language model with optimized settings."""
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # Optimized generation parameters for DeepSeek models
            text_generation_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,  # Increased for more detailed answers
                temperature=0.5,     # Reduced for more focused responses
                top_p=0.95,          # Slightly increased for more creativity while staying factual
                repetition_penalty=1.15,  # Slightly increased to avoid repetition
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

            llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
            logger.info(f"Initialized language model: {self.model_name}")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize language model: {e}")
            raise


    def find_document_path(self, paper_id: str) -> Optional[str]:
        """
        Find the document path for a given paper ID.

        Args:
            paper_id: The ID of the paper

        Returns:
            The path to the PDF file or None if not found
        """
        try:
            pdf_files = glob.glob(f"data/pdfs/{paper_id}.pdf")
            if not pdf_files:
                return None
            return pdf_files[0]
        except Exception as e:
            logger.error(f"Error finding document path: {e}")
            return None

    def clean_answer(self, text: str) -> str:
        """
        Clean the generated answer by removing duplicate content and fixing formatting.
        """
        # Split into sentences
        sentences = text.split('. ')
        seen = set()
        unique = []

        for sent in sentences:
            # Use first 50 chars as a key to detect duplicates
            key = sent[:50].lower() if len(sent) >= 50 else sent.lower()
            if key not in seen:
                seen.add(key)
                unique.append(sent)

        # Join sentences back together
        cleaned_text = '. '.join(unique)
        # Fix any potential extra spaces or artifacts
        cleaned_text = cleaned_text.replace('  ', ' ').strip()

        return cleaned_text

    def query(self, question:str, context:list, k:int = 5, score_threshold:float = 0.6) -> Dict:
        """
        Query the RAG system with a question and optional context.
        Incorporates a thinking step to improve answer quality.

        Args:
            question: The question to answer
            context: Optional pre-defined context to use instead of or in addition to retrieved content
            k: Number of documents to retrieve
            score_threshold: Threshold for filtering documents by score

        Returns:
            A dictionary containing the full, unmodified response from the model
        """
        logger.info(f"Processing query: {question}")
        logger.debug(f"Context provided: {len(context)} documents")

        try:
            # Generate prompt with thinking step
            prompt = get_prompt(question, context)

            # Configure thinking parameters
            thinking_params = {
                "max_new_tokens": 2048,     # More tokens for in-depth thinking
                "temperature": 0.35,        # Lower temperature for more focused reasoning
                "repetition_penalty": 1.2,  # Slightly higher to prevent circular reasoning
                "top_p": 0.85               # More focused sampling for logical coherence
            }

            # Generate response with thinking parameters
            logger.info("Generating response with thinking step...")
            generated_response = self.llm.invoke(
                prompt,
                max_new_tokens=thinking_params["max_new_tokens"],
                temperature=thinking_params["temperature"],
                repetition_penalty=thinking_params["repetition_penalty"],
                top_p=thinking_params["top_p"]
            )
            logger.info("Generation completed")

            return generated_response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.error(traceback.format_exc())
            return f"An error occurred while processing your query: {str(e)}"