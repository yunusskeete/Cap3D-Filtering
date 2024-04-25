"""
Module for downloading and updating caption files from a specified URL.

This module contains functionality for downloading caption files from a given URL
and checks whether the existing file is up to date using a checksum validation.
If the file is outdated, it is renamed with a timestamp, and the new version is
downloaded. It uses logging for status updates and error handling, and supports 
streamed downloads with customizable chunk sizes.

Functions:
- None (the code runs as a script)

Constants:
- CAPTIONS_DOWNLOAD_URL: The URL from which to download the captions.
- path_to_captions: The local path where the caption file will be stored.
- FILE_PREFIX: The prefix for accessing additional files online.
- REQUESTS_TIMEOUT: The timeout duration for HTTP requests.
- CHUNK_SIZE: The chunk size for streamed downloads.

Classes:
- None

Dependencies:
- requests: Used for HTTP requests to download files.
- datetime: For managing and formatting dates/times.
- os: For file system operations like renaming and path handling.
- sys: For manipulating the Python path.
- pathlib: Provides a cleaner way to work with file paths.

Example Usage:
    You can run this script to download the latest caption file and rename existing
    files with a timestamp if they are outdated:
    ```python
    python download_captions.py
    ```
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Union

import requests

# Ensure the parent directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

from utils.checksum import perform_checksum

CAPTIONS_DOWNLOAD_URL = "https://huggingface.co/datasets/tiange/Cap3D/resolve/main/Cap3D_automated_Objaverse_full.csv?download=true"
path_to_captions: Union[str, Path] = "Cap3D_automated_Objaverse_full.csv"
FILE_PREFIX = "https://huggingface.co/datasets/tiange/Cap3D/raw/main/"

REQUESTS_TIMEOUT = 60
CHUNK_SIZE = 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger()  # Get the root logger

PERFORM_DOWNLOAD: bool = True

if os.path.exists(path_to_captions):
    print(f"Captions exists at '{path_to_captions}'")

    # Validate file contents to verify if file has been updated
    pointer_file_url: str = FILE_PREFIX + path_to_captions
    file_up_to_date: bool = perform_checksum(
        file_path=path_to_captions, pointer_file_url=pointer_file_url, logger=logger
    )

    if file_up_to_date:
        PERFORM_DOWNLOAD = False
        logger.info("File up to date, no download will be attempted.")

    else:
        # Save the current file as a historic version with timestamp
        rename_path = path_to_captions.replace(
            ".csv", f"-{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.csv"
        )
        os.rename(path_to_captions, rename_path)
        assert not os.path.exists(
            path_to_captions
        ), f"File renaming failed, aborting file download to avoid overwrite. (File exists at '{path_to_captions}')"

if PERFORM_DOWNLOAD:
    # Download latest file
    try:
        with requests.get(
            CAPTIONS_DOWNLOAD_URL, stream=True, timeout=REQUESTS_TIMEOUT
        ) as response:
            # Raise exception if status code is anything other than 200
            response.raise_for_status()

            file_size: int = int(response.headers.get("content-length", 0))
            print(f"File size: {file_size}")

            # Write to file in "append binary" mode ("ab")
            with open(path_to_captions, "ab") as file:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        file.write(chunk)
                    else:
                        logger.warning("Empty chunk")

        logger.info(
            "Download file process successfully completed (url: '%s', destination: '%s')",
            CAPTIONS_DOWNLOAD_URL,
            os.path.abspath(path_to_captions),
        )

    except requests.exceptions.RequestException as e:
        logger.error(
            "An error occurred during the download file process: %s (url: '%s', destination: '%s')",
            e,
            CAPTIONS_DOWNLOAD_URL,
            path_to_captions,
        )
