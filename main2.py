from transformers import BarkModel, AutoProcessor
import torch
import scipy
torch.cuda.empty_cache()

def text_to_audio(bark_model='suno/bark',voice_preset='v2/it_speaker_3'):
    model = BarkModel.from_pretrained(bark_model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(bark_model)


    text = open("testo.txt", "r")
    lines = text.readlines()
    count = 0
    for l in lines:
        newText = l.replace("\s", "").strip()
        if (newText.__len__() > 0):
            print(l.strip())
            print("sono alla riga ",count)
            inputs = processor(l, voice_preset=voice_preset).to(device)
            audio_array = model.generate(**inputs)
            audio_array = audio_array.cpu().numpy().squeeze()
            sample_rate = model.generation_config.sample_rate
            scipy.io.wavfile.write("/text_to_speach/voices/"+f'{count}.wav', rate=sample_rate, data=audio_array)
            count = count +1
        





def main():
    text_to_audio()


if __name__ == '__main__':
    main()
