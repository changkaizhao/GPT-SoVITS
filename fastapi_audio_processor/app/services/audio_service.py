from fastapi import UploadFile
import os
from typing import Any
from app.utils.file_utils import save_file

async def process_audio_file(id: str, audio_file: UploadFile) -> str:
    directory_path = os.path.join("temp", id)
    os.makedirs(directory_path, exist_ok=True)
    
    file_path = await save_file(audio_file, directory_path)
    return file_path