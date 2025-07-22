import os
import soundfile as sf

from GPT_SoVITS.inference_webui import (
    change_gpt_weights,
    change_sovits_weights,
    get_tts_wav,
)
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()


def inference(
    GPT_model_path,
    SoVITS_model_path,
    ref_audio_path,
    ref_text,
    ref_language,
    target_text,
    target_language,
    output_path,
    file_name="output.wav",
):
    # Change model weights
    change_gpt_weights(gpt_path=GPT_model_path)
    change_sovits_weights(
        sovits_path=SoVITS_model_path,
        prompt_language=ref_language,
        text_language=target_language,
    )

    # Synthesize audio
    synthesis_result = get_tts_wav(
        ref_wav_path=ref_audio_path,
        prompt_text=ref_text,
        prompt_language=ref_language,
        text=target_text,
        text_language=target_language,
        top_p=1,
        temperature=1,
    )

    result_list = list(synthesis_result)

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        output_wav_path = os.path.join(output_path, file_name)
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_wav_path}")


def main():
    inference(
        GPT_model_path="path/to/gpt/model",
        SoVITS_model_path="path/to/sovits/model",
        ref_audio_path="path/to/reference/audio.wav",
        ref_text="Reference text for the audio.",
        ref_language="English",
        target_text="Target text to synthesize.",
        target_language="Chinese",
        output_path="path/to/output/directory",
        file_name="output.wav",
    )
