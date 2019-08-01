# Voicebot

This is an end-to-end voicebot that aims to answer open domain questions, and is intended to be used as a benchmarking tool

## Design
**![](https://lh5.googleusercontent.com/2oFMv1ybATD_cmMO0CwzB-RAk6Nz-VG1wwDioIGWahLR4bVG51TIHbhHIUGTSpaLcVQS41QZIPOfX00VbZGCPa5O98st_VRsNlJnC3qEehpnEJrYLyLUOdCy-wiD34IC26wCac4KnxY)**


## Requirements and Setup

 - Pytorch
 - Tensorflow (1.12 but 1.14 is also supported)
 - Wikipedia-API
 - Deepspeech
 - Spacy
 - GingerIt
 - pytorch-pretrained-bert
 - synthesizer
 - playsound
 - sounddevice
 - soundfile
 - pygame

You will also require the following models 

 - [BERT model fine tuned for SQuAD](https://drive.google.com/file/d/1hktnjAJOdOwPxTK3R-KST9-kUQFYPusM/view?usp=sharing)
 - [Deepspeech 0.5.0 model](https://github.com/mozilla/DeepSpeech/releases/download/v0.5.0/deepspeech-0.5.0-models.tar.gz)
 - [Tacotron model](http://data.keithito.com/data/speech/tacotron-20180906.tar.gz)
 - [WaveRNN model](https://drive.google.com/open?id=1mv0-1uTZpAGrH9GIjvFgjw-YeYg7mjiN)

 Extract the contents of the models and place them in their respective folders in the project. (BERT, DeepSpeech/Models and tacotron-models folders respectively. WaveRNN should be extracted under Vocoder_WaveRNN in a separate WaveRNN_weights folder)

Open domain QA will also require an internet connection, to get information from Wikipedia. 
 
 
## Running the program
Run the Voicebot file to start the application. You will be prompted to select the TTS system of your choice after the other models have loaded.
 > The WaveRNN + Tacotron is very resource heavy and produces poor results when run on systems with 8GB of RAM. The speech produced is a lot more natural sounding but often have garbage audio produced towards the end. The standalone tacotron is much lighter, and will not have as poor results on systems with lower resources

Once the TTS has been loaded you will be prompted to select the running mode. This will let you choose between a microphone for input audio, or allow you to use a folder of audio files to test. 
To add your own audio to the testing set, simply place the wav file in the test-audio folder. For best results, use an American male voice, with a normal or slow speed setting from a site like [this](http://www.fromtexttospeech.com/). 

## References

 - [Mozilla Deepspeech](https://github.com/mozilla/DeepSpeech)
 - [Keith Ito's Tacotron implementation](https://github.com/keithito/tacotron)
 - [Hugging Face's BERT for QA implementation](https://github.com/huggingface/pytorch-transformers)
 - [Fatchord's WaveRNN](https://github.com/fatchord/WaveRNN)
 - [BERT model trained by Surbhi Bhardwaj](https://github.com/surbhardwaj/BERT-QnA-Squad_2.0_Finetuned_Model)
