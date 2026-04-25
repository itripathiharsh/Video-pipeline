from fastapi import FastAPI
from config import Config
from src.storage.s3_storage import S3Storage
from src.main_cloud import process_single_video

import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

cfg = Config()
s3 = S3Storage(bucket_name=cfg.S3_BUCKET)

# 🔒 Prevent parallel overload
is_processing = False

@app.post("/process")
def process_video(data: dict):
    global is_processing

    s3_key = data.get("s3_key")

    if not s3_key:
        return {"status": "error", "message": "Missing s3_key"}

    if is_processing:
        return {"status": "busy"}

    def run():
        global is_processing
        try:
            is_processing = True
            process_single_video(s3_key, cfg, s3)
        finally:
            is_processing = False

    # run in background thread (non-blocking)
    threading.Thread(target=run).start()

    return {"status": "started"}
