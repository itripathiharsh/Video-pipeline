import threading
import logging
from fastapi import FastAPI
from config import Config
from src.storage.s3_storage import S3Storage
from src.main_cloud import process_single_video

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Initialization ---
cfg = Config()
s3 = S3Storage(bucket_name=cfg.S3_BUCKET)

# 🔒 Global lock to prevent parallel overload on the GPU/CPU
is_processing = False

@app.post("/process")
def process_video(data: dict):
    global is_processing

    logger.info(f"Incoming request: {data}")

    s3_key = data.get("s3_key")

    # 1. Validation
    if not s3_key:
        logger.warning("Request failed: Missing s3_key")
        return {"status": "error", "message": "Missing s3_key"}

    # 2. Check if the system is already occupied
    if is_processing:
        logger.info(f"System busy. Rejecting request for: {s3_key}")
        return {"status": "busy"}

    # 3. Background Task Definition
    def run():
        global is_processing
        try:
            is_processing = True
            logger.info(f"Starting background processing for: {s3_key}")
            process_single_video(s3_key, cfg, s3)
            logger.info(f"Successfully finished processing: {s3_key}")
        except Exception as e:
            logger.error(f"Error during processing {s3_key}: {str(e)}")
        finally:
            is_processing = False

    # 4. Start Thread
    # daemon=True ensures the thread exits if the main program stops
    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    return {
        "status": "started", 
        "s3_key": s3_key
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
