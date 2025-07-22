import os
from fastapi import UploadFile

def create_directory(id: str, base_path: str = "temp") -> str:
    directory_path = os.path.join(base_path, id)
    os.makedirs(directory_path, exist_ok=True)
    return directory_path

def save_audio_file(file: UploadFile, directory: str) -> str:
    file_path = os.path.join(directory, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    return file_path