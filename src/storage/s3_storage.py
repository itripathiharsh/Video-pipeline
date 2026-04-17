import boto3
import os
import logging
from typing import List, Optional, Tuple

# ----------------------------
# LOGGER SETUP
# ----------------------------
logger = logging.getLogger(__name__)


class S3Storage:
    """
    Handles all S3 interactions for the CCTV pipeline.

    Responsibilities:
    - List schools
    - Fetch latest video per school
    - Download video to EC2 (/tmp)
    - Upload processed clips back to S3
    """

    def __init__(self, bucket_name: str):
        self.bucket = bucket_name
        self.s3 = boto3.client("s3")

    # ----------------------------
    # LIST SCHOOLS (ROBUST FIX)
    # ----------------------------

    def list_schools(self, prefix: str = "School CCTV Dataset/") -> List[str]:
        """
        Robustly list school folders from S3. 


        Works even when CommonPrefixes is missing.
        """

        try:
        print(">>> DEBUG: list_schools called")  
        
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )

            files = response.get("Contents", [])

            schools = set()

            for f in files:
                key = f["Key"]

                parts = key.split("/")

                # Ensure structure: School CCTV Dataset/<School>/file.mp4
                if len(parts) >= 3:
                    school_prefix = "/".join(parts[:2]) + "/"
                    schools.add(school_prefix)

            schools = list(schools)

            logger.info(f"Found {len(schools)} schools: {schools}")
            return schools

        except Exception as e:
            logger.error(f"Error listing schools: {e}")
            return []

    # ----------------------------
    # GET LATEST VIDEO
    # ----------------------------

    def get_latest_video(self, school_prefix: str) -> Optional[str]:
        """
        Get latest .mp4 file from a school folder.
        """
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=school_prefix
            )

            files = response.get("Contents", [])

            videos = [f for f in files if f["Key"].endswith(".mp4")]

            if not videos:
                logger.warning(f"No videos found in {school_prefix}")
                return None

            videos.sort(key=lambda x: x["LastModified"], reverse=True)

            latest_key = videos[0]["Key"]

            logger.info(f"Latest video for {school_prefix}: {latest_key}")
            return latest_key

        except Exception as e:
            logger.error(f"Error fetching latest video: {e}")
            return None

    # ----------------------------
    # DOWNLOAD VIDEO
    # ----------------------------

    def download_video(self, s3_key: str, local_path: str) -> str:
        """
        Download video from S3 to EC2 local storage (/tmp).
        """
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            logger.info(f"Downloading {s3_key} → {local_path}")

            self.s3.download_file(self.bucket, s3_key, local_path)

            return local_path

        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    # ----------------------------
    # UPLOAD CLIPS
    # ----------------------------

    def upload_clips(
        self,
        local_clip_paths: List[str],
        school_name: str,
        date_folder: str
    ):
        """
        Upload processed clips to S3.
        """
        for clip_path in local_clip_paths:
            try:
                file_name = os.path.basename(clip_path)

                s3_key = f"input_video/{school_name}/{date_folder}/{file_name}"

                logger.info(f"Uploading {clip_path} → {s3_key}")

                self.s3.upload_file(clip_path, self.bucket, s3_key)

            except Exception as e:
                logger.error(f"Failed to upload {clip_path}: {e}")

    # ----------------------------
    # PARSE S3 KEY
    # ----------------------------

    def parse_s3_key(self, key: str) -> Tuple[str, str]:
        """
        Extract school name and date from S3 key.

        Example:
        School CCTV Dataset/Lucknow Public School/17-04-26.mp4
        → ("Lucknow Public School", "17-04-26")
        """
        try:
            parts = key.split("/")

            school_name = parts[1]
            file_name = parts[-1]
            date = file_name.replace(".mp4", "")

            return school_name, date

        except Exception as e:
            logger.error(f"Failed to parse S3 key: {key} | Error: {e}")
            raise
