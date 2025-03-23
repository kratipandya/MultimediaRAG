#pylint: disable=all
from typing import Optional
import os
import json
import pickle
import hashlib
import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from robyn import Request
import logging
import traceback

from RAGembedder.arxiv_rag_system import ArXivRAGSystem
from RAGembedder.multi_modal_embedder import MultimodalEmbedder

from api_dirty import EnumEncoder, init_logger, handle_formdata_save
logger = logging.getLogger(__name__)
init_logger(logger)


type Hash = str;

class QueueStatus(Enum):
    NOT_REGISTERED = 0
    REGISTERED = 1
    FAISS = 2
    RAG = 3
    FINISHED = 4

@dataclass
class QueryHandler:
    """ Handles queries made to backend """

    # Queue -list that rag_thread handles
    faiss_queue: set[int] = field(default_factory=set)
    rag_queue: set[int] = field(default_factory=set)
    # Define where to save everything
    query_path:  os.PathLike = field(default="query")
    upload_path: os.PathLike = field(default="uploads")
    result_path: os.PathLike = field(default="faiss_results")

    embedder:MultimodalEmbedder = field(default=None)
    embedder_thread:threading.Thread = field(default=None)

    rag:ArXivRAGSystem = field(default=None)
    rag_thread:threading.Thread = field(default=None)


    def __post_init__(self):
        os.makedirs(self.query_path, exist_ok=True)
        os.makedirs(self.upload_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)

        self.faiss_queue = set([])
        self.rag_queue = set([])

        self.embedder = MultimodalEmbedder()
        self.embedder.load_indices()

        self.rag = ArXivRAGSystem()

        def embedder_operator(embedder: MultimodalEmbedder):
            """FAISS thread-operator"""
            while True:
                try:
                    # Checking if not empty and ...
                    if self.faiss_queue and (hash := self.faiss_queue.pop()):
                        query = self.get_query_json(hash)
                        result = embedder.search(query)
                        self.register_faiss_result(hash, result)
                        self.rag_queue.add(hash)
                    else:
                        time.sleep(1)
                except Exception as e:
                    logger.error(f"FAISS THREAD ERROR: {e}")
                    logger.error(traceback.format_exc())

        self.embedder_thread = threading.Thread(target=embedder_operator, args=[self.embedder])
        self.embedder_thread.start()

        def rag_operator(rag: ArXivRAGSystem):
            """RAG thread-operator"""
            while True:
                try:
                    if self.rag_queue and (hash := self.rag_queue.pop()):
                        result = self.get_result_json(hash)
                        texts = result["results"].get("text", None)
                        if texts:
                            context = [(
                                        f"Article {i+1}:"
                                        + x.get("title")
                                        + "\n"
                                        + x.get("text", "")
                                    ) for i, x in enumerate(texts[:3])]
                        else:
                            context = [""]

                        answer = rag.query(result.get("query_text"), context=context)
                        answer = answer if answer is not None else ""
                        self.register_rag_answer(hash, answer)
                    else:
                        time.sleep(1)
                except Exception as e:
                    logger.error(f"FAISS THREAD ERROR: {e}")
                    logger.error(traceback.format_exc())

        self.rag_thread = threading.Thread(target=rag_operator, args=[self.rag])
        self.rag_thread.start()

    def register_query(self, request: Request) -> Optional[Hash]:
        try:
            os.makedirs(self.query_path, exist_ok=True)

            # Get query information
            filenames  = [os.path.abspath(os.path.join("uploads", x)) for x in request.files.keys()]
            image_path = next(filter(lambda file: file.endswith(".png"), filenames), None)
            audio_path = next(filter(lambda file: file.endswith(".wav"), filenames), None)

            query_information = self._generate_query_json(
                status=QueueStatus.REGISTERED,
                text=request.form_data["query"] if len(request.form_data["query"]) > 0 else None,
                image=image_path,
                audio=audio_path,
            )

            # Get hash based on query_information
            hash = hashlib.sha256(pickle.dumps(query_information)).hexdigest()
            query_information['id'] = str(hash)

            if self.is_registered(hash):
                logger.info(f"Query with hash {hash} already registered.")
                return hash

            # Save query
            filepath = os.path.join(self.query_path, hash + ".json")
            with open(filepath, "w", encoding="utf-8") as file:
                json.dump(query_information, file, ensure_ascii=False, indent=2, cls=EnumEncoder)

            handle_formdata_save(request.files, self.upload_path)
            self.faiss_queue_push(hash)

        except Exception as e:
            logger.error(f"Register query log error {e}")
            logger.error(traceback.format_exc())
            return None

        logger.info(f"Queue: {list(self.faiss_queue)}")
        return hash

    def faiss_queue_push(self, hash: Hash):
        """Push query and make faiss indices based search"""
        logger.info(f"Queue pushed hash {str(hash)}")
        self.faiss_queue.add(hash)

    def register_faiss_result(self, hash: Hash, result, answer=None):
        self._change_status(hash, QueueStatus.FAISS)
        self.save_result_json(hash, result, answer)
        # self.faiss_queue.remove(hash)

    def register_rag_answer(self, hash, answer):
        self._change_status(hash, QueueStatus.FINISHED)
        self.save_rag_answer(hash, answer)

    def get_faiss_results(self, hash: Hash) -> dict:
        return self.get_result_json(hash) or {"hash": hash, "result": []}

    def is_registered(self, hash: Hash) -> bool:
        return self.get_status(hash) != QueueStatus.NOT_REGISTERED

    def get_status(self, hash: Hash) -> QueueStatus:
        # May crash intentionally
        hash_json = self.get_query_json(hash)
        if hash_json:
            return hash_json.get("status", QueueStatus.NOT_REGISTERED)
        else:
            return QueueStatus.NOT_REGISTERED

    def save_result_json(self, hash, results, answer=None):
        results_json = {
            "hash": str(hash),
            "answer": answer,
            "query_text": self.get_query_json(hash).get("text", None),
            "results": results,
        }
        self._json_dump(self.path_to_results_json(hash), results_json)

    def save_rag_answer(self, hash, answer):
        result = self.get_result_json(hash)
        result["answer"] = answer
        self._json_dump(self.path_to_results_json(hash), result)

    def get_query_json(self, hash) -> Optional[dict]:
        """Get sended query json"""
        return self._json_load(self.path_to_query_json(hash=hash))

    def get_result_json(self, hash) -> Optional[dict]:
        """Get faiss results json"""
        return self._json_load(self.path_to_results_json(hash=hash))


    def path_to_results_json(self, hash) -> os.PathLike:
        return os.path.join(self.result_path, str(hash) + ".json")

    def path_to_query_json(self, hash) -> os.PathLike:
        return os.path.join(self.query_path, str(hash) + ".json")

    ################## INTERNAL ##########################

    def _change_status(self, hash: Hash, status: QueueStatus):
        query = self.get_query_json(hash=hash)
        query["status"] = status
        self._json_dump(self.path_to_query_json(hash), query)


    def _generate_query_json(self, status, **kwargs):
        return {
            "text": kwargs.get("text", None),
            "image": kwargs.get("image", None),
            "audio": kwargs.get("audio", None),
            "status" : status,
        }

    def _json_dump(self, filepath, content):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as file:
                json.dump(content, file, ensure_ascii=False, indent=2, cls=EnumEncoder)
            return True
        except Exception as e:
            logger.error(f"_json_dump {e}")
            return False

    def _json_load(self, filepath) -> Optional[any]:
        json_file = None
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                json_file = json.load(file)
        return json_file




if __name__=="__main__":
    query_handler = QueryHandler()
