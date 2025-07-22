# FastAPI Audio Processor

This project is a FastAPI application that processes audio files. It accepts JSON data containing an "id" field and an audio file, creates a directory named after the "id" in a temporary folder, and saves the audio file into that directory.

## Project Structure

```
fastapi-audio-processor
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── models
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── routers
│   │   ├── __init__.py
│   │   └── audio.py
│   ├── services
│   │   ├── __init__.py
│   │   └── audio_service.py
│   └── utils
│       ├── __init__.py
│       └── file_utils.py
├── temp
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd fastapi-audio-processor
   ```

2. Create a virtual environment and activate it:
   ```
   conda create -n fastapi-audio-processor python=3.9
   conda activate fastapi-audio-processor
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   uvicorn app.main:app --reload
   ```

## Usage

To use the API, send a POST request to the `/audio` endpoint with the following JSON body and an audio file:

### Request Example

```
POST /audio
Content-Type: multipart/form-data

{
  "id": "unique_id"
}
```

### Response

The server will create a directory named `unique_id` in the `temp` folder and save the audio file in that directory.

### Steps

>1. "/home/roby/miniconda3/envs/GPTSoVits/bin/python" -s tools/slice_audio.py "/home/roby/proj/video_translater/temp/iVZok8e2S3g/Israel launches strikes against Iran ｜ Air India’s deadly plane crash [iVZok8e2S3g]/vocals.mp3" "output/slicer_opt" -34 5000 500 10 5000 0.9 0.25 0 1

>2. "/home/roby/miniconda3/envs/GPTSoVits/bin/python" -s tools/asr/fasterwhisper_asr.py -i "output/slicer_opt" -o "output/asr_opt" -s large-v3 -l en -p float16

>3. "/home/roby/miniconda3/envs/GPTSoVits/bin/python" -s GPT_SoVITS/prepare_datasets/1-get-text.py

>4. "/home/roby/miniconda3/envs/GPTSoVits/bin/python" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py

>5. "/home/roby/miniconda3/envs/GPTSoVits/bin/python" -s GPT_SoVITS/prepare_datasets/2-get-sv.py

>6. "/home/roby/miniconda3/envs/GPTSoVits/bin/python" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py


>7. "/home/roby/miniconda3/envs/GPTSoVits/bin/python" -s GPT_SoVITS/s2_train.py --config "/home/roby/proj/GPT-SoVITS/TEMP/tmp_s2.json"

>8. "/home/roby/miniconda3/envs/GPTSoVits/bin/python" -s GPT_SoVITS/s1_train.py --config_file "/home/roby/proj/GPT-SoVITS/TEMP/tmp_s1.yaml"




## License

This project is licensed under the MIT License.