import sys
import wave
import os
import audioop
import requests
import json

from timeit import default_timer as timer
import numpy as np
from gingerit.gingerit import GingerIt
from playsound import playsound

from deepspeech import Model

spell_check = GingerIt()
directory_in_str = "test_audio/"
directory = os.fsencode(directory_in_str)
sender = "client"


def change_samplerate(audio_in, inrate):
    # s_read = wave.open(audio_path,'r')
    n_frames = audio_in.getnframes()
    channels = audio_in.getnchannels()
    data = audio_in.readframes(n_frames)
    converted = audioop.ratecv(data, 2, channels, inrate, 16000, None)
    converted = audioop.tomono(converted[0], 2, 1, 0)
    op = np.frombuffer(converted, np.int16)
    return 16000, op


BEAM_WIDTH = 500
LM_ALPHA = 0.75
LM_BETA = 1.85

speech_model_path = 'DeepSpeech/Models/output_graph.pb'
alphabet = 'DeepSpeech/Models/alphabet.txt'
lm = 'DeepSpeech/Models/lm.binary'
trie = 'DeepSpeech/Models/trie'
N_FEATURES = 261
N_CONTEXT = 9

model_load_start = timer()
ds = Model(speech_model_path, N_FEATURES, N_CONTEXT, alphabet, BEAM_WIDTH)
model_load_end = timer() - model_load_start
print('Loaded S2T model in {:.3}s.'.format(model_load_end))


def test_files():
    count = 0
    time_for_all_files = 0
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"):
            start_time = timer()
            fn2 = directory_in_str + filename
            #playsound(fn2)
            fin = wave.open(fn2, 'rb')
            fs = fin.getframerate()
            if fs != 16000:
                print('Resampling from ({}) to 16kHz.'.format(fs), file=sys.stderr)
                fs, audio = change_samplerate(fin, fs)
                audio_length = fin.getnframes() * (1 / 16000)
                fin.close()
            else:
                audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
                audio_length = fin.getnframes() * (1 / 16000)
                fin.close()

            print('Running inference.', file=sys.stderr)
            inference_start = timer()
            qasked = ds.stt(audio, fs)
            inference_end = timer() - inference_start
            print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)
            print("Inferred:", qasked)
            qasked = spell_check.parse(qasked)['result']
            print("Question:", qasked)
            gen_start = timer()

            # SEND HERE
            r = requests.post('http://localhost:5002/webhooks/rest/webhook',
                              json={"sender": sender, "message": qasked})
            print("Bot says, ")
            print(r)
            #Change
            
            if len(r.json()) == 0:
                bot_message = "Sorry I couldn't answer that"
            else:
                bot_message = r.json()[0]['text']

            print(bot_message)
            print("Answer generated in {:.3}s.".format(timer() - gen_start))
            print("Sending to T2S server")
            # t2s_time = timer()
            #Changed port for T2S from 1003 to 5004
            s2t_req = requests.post('http://localhost:5004/t2s', data=json.dumps({'ans_output': str(bot_message)}))
            sample_time = timer() - start_time
            print("Time for sample: {:.3}s.\n".format(sample_time))
            time_for_all_files += sample_time
            count += 1
            print("******")
    print("Time for all samples :", time_for_all_files, "s")
    print("Average time: {:.3}s".format(time_for_all_files / count))

test_files()