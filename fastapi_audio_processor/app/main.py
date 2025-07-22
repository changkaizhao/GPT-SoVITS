from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_audio_processor.app.routers import tts


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tts.router)


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Audio Processor!"}
