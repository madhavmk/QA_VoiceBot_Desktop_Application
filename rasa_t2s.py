from flask import Flask, request
import requests
from Tacotron_TTS.synthesizer import Synthesizer
from timeit import default_timer as timer

model_load_start = timer()
synthesizer = Synthesizer()
synthesizer.load('Tacotron_TTS/tacotron_model_data/model.ckpt')
model_load_end = timer() - model_load_start
print('Loaded T2S model in {:.3}s.'.format(model_load_end))

app = Flask(__name__)


@app.route("/t2s", methods = ['GET', 'POST'])
def receive_process_send():

    ### receive TEXT

    ans_output = request.get_json(force=True)['ans_output']
    print(ans_output)

    ### Convert text to Speech file

    aud_timer = timer()
    aud_out = synthesizer.synthesize(ans_output)
    print('Took {:.3}s for audio synthesis.'.format(timer() - aud_timer))
    fname = f'Audio_Outputs/__input_{ans_output[:10]}.wav'
    with open(fname, mode='wb') as f:
        f.write(aud_out)

    ### Sending answer back to main server

    #main_url = 'http://localhost:1000/receive_t2s'
    #wav_file = {'file': open('t2s_output.wav', 'rb')}
    #requests.post(main_url, files=wav_file)

    return ''


@app.route("/")
def index():
    return "Text to speech server running !!"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5004)
