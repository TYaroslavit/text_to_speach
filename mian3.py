import nltk
import os
import numpy as np
import scipy
from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE
os.environ["SUNO_USE_SMALL_MODELS"] = True

def text_to_audio(voice_preset='v2/it_speaker_3'):

    text = open("testo.txt", "r", encoding="utf8")
    lines = text.readlines()
    count = 0
    for l in lines:
        newText = l.replace("\s", "").strip()
        if (newText.__len__() > 0):
            print(newText)
            sentences = nltk.sent_tokenize(newText)
            silence = np.zeros(int(0.25 * SAMPLE_RATE))
            pieces = []
            for sentence in sentences:
                semantic_tokens = generate_text_semantic(
                    sentence,
                    history_prompt=voice_preset,
                    # temp=GEN_TEMP,
                    min_eos_p=0.05,
                )
                audio_array = semantic_to_waveform(semantic_tokens, history_prompt=voice_preset)
                pieces += [audio_array, silence.copy()]
            scipy.io.wavfile.write(f'{voice_preset.split("/")[1]}{count}_long.wav', rate=SAMPLE_RATE,
                                   data=np.concatenate(pieces))


    # for sentence in sentences:
    #     audio_array = generate_audio(sentence, history_prompt=voice_preset)
    #     pieces += [audio_array, silence.copy()]






def main():
    nltk.download('punkt')
    preload_models()
    text_to_audio()


if __name__ == '__main__':
    main()
