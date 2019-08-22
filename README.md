# Voicebot

This is an end-to-end voicebot that aims to answer open domain questions, and is intended to be used as a benchmarking tool

## Design
**![](https://lh5.googleusercontent.com/2oFMv1ybATD_cmMO0CwzB-RAk6Nz-VG1wwDioIGWahLR4bVG51TIHbhHIUGTSpaLcVQS41QZIPOfX00VbZGCPa5O98st_VRsNlJnC3qEehpnEJrYLyLUOdCy-wiD34IC26wCac4KnxY)**


## Requirements and Setup

 - python 3.6
 - pytorch (1.1.0)
 - tensorflow (1.12)
 - wikipedia (1.14)
 - deepspeech (0.5.0)
 - spacy (2.1.5)
 - gingerit (0.8.0)
 - pytorch-pretrained-bert (0.6.2)
 - playsound (1.2.2)
 - sounddevice (0.3.13)
 - soundfile (0.10.2)
 - inflect (2.1)
 - librosa (0.7.0)
 - matplotlib (3.1.1)
 - unidecode (1.1.1)
 - numpy (1.17.0)
 
We recommend using a virtual environment to run this to prevent any conflicts with things like numpy.

You can install any of the Spacy NER models you prefer (We have used 'en_core_web_md') by: (We have to run it with Administaion permissions)
 - python -m spacy download en_core_web_md

You will also require the following models 

 - [BERT model fine tuned for SQuAD](https://drive.google.com/file/d/1hktnjAJOdOwPxTK3R-KST9-kUQFYPusM/view?usp=sharing)
 - [Deepspeech 0.5.0 model](https://github.com/mozilla/DeepSpeech/releases/download/v0.5.0/deepspeech-0.5.0-models.tar.gz)
 - [Tacotron model](http://data.keithito.com/data/speech/tacotron-20180906.tar.gz)
 - [WaveRNN model](https://drive.google.com/open?id=1mv0-1uTZpAGrH9GIjvFgjw-YeYg7mjiN)

 An info.txt file is located in every directory where a specific model is required.
 Extract the contents of the models and place them in their respective folders in the project. (BERT, DeepSpeech/Models and Tacotron_TTS/tacotron-models-data folders respectively. WaveRNN should be extracted under the Vocoder_WaveRNN folder)

Open domain QA will also require an internet connection, to get information from Wikipedia. 
 
 
## Running the program
Run the Voicebot file to start the application. You will be prompted to select the TTS system of your choice after the other models have loaded.
 > The WaveRNN + Tacotron is very resource heavy and produces poor results when run on systems with 8GB of RAM. The speech produced is a lot more natural sounding but often have garbage audio produced towards the end. The standalone tacotron is much lighter, and will not have as poor results on systems with lower resources

Once the TTS has been loaded you will be prompted to select the running mode. This will let you choose between a microphone for input audio, or allow you to use a folder of audio files to test. 
To add your own audio to the testing set, simply place the wav file in the test-audio folder. For best results, use an American male voice, with a normal or slow speed setting from a site like [this](http://www.fromtexttospeech.com/). 

## Running on Windows 10
Run VoiceBot-windows.py
Initially designed for the Windows platform. As such, all features should work perfectly.
Outputs can be accessed from '/Vocoder_WaveRNN/WaveRNN_outputs' OR '/Tacotron_TTS/Tacotron_outputs' subfolders

## Running on Ubuntu
Rin the VoiceBot-linux.py file
playsound library and sounddevice library is not compatible.
So, audio cannot be recorded or played on or from the console.
VoiceBot can work only from questions pre-recorded in 'test_audio' folder.
Outputs can be accessed from '/Vocoder_WaveRNN/WaveRNN_outputs' OR '/Tacotron_TTS/Tacotron_outputs' subfolders

## References

 - [Mozilla Deepspeech](https://github.com/mozilla/DeepSpeech)
 - [Keith Ito's Tacotron implementation](https://github.com/keithito/tacotron)
 - [Hugging Face's BERT for QA implementation](https://github.com/huggingface/pytorch-transformers)
 - [Fatchord's WaveRNN](https://github.com/fatchord/WaveRNN)
 - [BERT model trained by Surbhi Bhardwaj](https://github.com/surbhardwaj/BERT-QnA-Squad_2.0_Finetuned_Model)
