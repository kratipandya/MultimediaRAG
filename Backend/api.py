# pylint: disable=all
import os
import logging
# from RAG import ArXivRAGSystem
####################### ENDPOINT HANDLING ####################
from robyn import Robyn, ALLOW_CORS
from robyn.robyn import Request, QueryParams
app = Robyn(__file__)
####################### API OF APIS ####################
from api_queue import QueryHandler
from api_dirty import init_logger, process_images, process_audio
####################### CORS ####################
ALLOW_CORS(app, origins = ["http://localhost:3000"])
####################### LOGGING ####################
logger = logging.getLogger(__name__)
init_logger(logger)
####################### CONFIG AND RAG ####################
# Handles the queue of queries
# config = load_configuration(logger)
# rag_system = ArXivRAGSystem(config=config)
####################### BACKEND API CALL INTERFACE ####################

QUERY_HANDLER = QueryHandler()


@app.get("/")
async def home(request):
    """
    Query support for text, .wav, and .png search
    """
    return "Hello, world!"



@app.post("/query")
async def query(request: Request):
    query_hash = QUERY_HANDLER.register_query(request=request)
    return {
        "status": "success",
        "query_id": query_hash,
    }


@app.get("/result")
async def result_status(_request: Request, query_params: QueryParams):
    logger.info(query_params)

    if (hash := query_params.get("q")) is not None:
        result = QUERY_HANDLER.get_result_json(hash) or {}

        # Process images
        if 'results' in result and 'image' in result['results']:
            image_data = result['results']['image']
            if image_data:
                result['results']['image'] = process_images(image_data, logger)

                nimages = len(result['results']['image'])
                # logger.info(f"Processed {nimages} images for response")

        # Process audio
        if 'results' in result and 'audio' in result['results']:
            audio_data = result['results']['audio']
            if audio_data:
                result['results']['audio'] = process_audio(audio_data, logger)

                naudio = len(result['results']['audio'])
                # logger.info(f"Processed {naudio} audio files for response")

        return result
    else:
        return { "status": "waiting" }



app.start(port=8080)
