import torch
from vocoder_models.fatchord_version import WaveRNN
import vocoder_hparams as hp
from vocoder_utils.text.symbols import symbols
from vocoder_models.tacotron import Tacotron
import argparse
from vocoder_utils.text import text_to_sequence
from vocoder_utils.display import save_attention, simple_table
import zipfile, os
import sounddevice as sd

if __name__ == "__main__" :

    print('\nInitialising WaveRNN Model...\n')

    # Instantiate WaveRNN Model
    voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                        fc_dims=hp.voc_fc_dims,
                        bits=hp.bits,
                        pad=hp.voc_pad,
                        upsample_factors=hp.voc_upsample_factors,
                        feat_dims=hp.num_mels,
                        compute_dims=hp.voc_compute_dims,
                        res_out_dims=hp.voc_res_out_dims,
                        res_blocks=hp.voc_res_blocks,
                        hop_length=hp.hop_length,
                        sample_rate=hp.sample_rate,
                        mode='MOL')

    voc_model.restore('quick_start/voc_weights/latest_weights.pyt')

    print('\nInitialising Tacotron Model...\n')

    # Instantiate Tacotron Model
    tts_model = Tacotron(embed_dims=hp.tts_embed_dims,
                         num_chars=len(symbols),
                         encoder_dims=hp.tts_encoder_dims,
                         decoder_dims=hp.tts_decoder_dims,
                         n_mels=hp.num_mels,
                         fft_bins=hp.num_mels,
                         postnet_dims=hp.tts_postnet_dims,
                         encoder_K=hp.tts_encoder_K,
                         lstm_dims=hp.tts_lstm_dims,
                         postnet_K=hp.tts_postnet_K,
                         num_highways=hp.tts_num_highways,
                         dropout=hp.tts_dropout)


    tts_model.restore('quick_start/tts_weights/latest_weights.pyt')

    input_text = input("Ya fam type yo shizz:")

    if input_text :
        inputs = [text_to_sequence(input_text.strip(), hp.tts_cleaner_names)]
    else :
        with open('sentences.txt') as f :
            inputs = [text_to_sequence(l.strip(), hp.tts_cleaner_names) for l in f]

    voc_k = voc_model.get_step() // 1000
    tts_k = tts_model.get_step() // 1000

    r = tts_model.get_r()

    for i, x in enumerate(inputs, 1) :

        print(f'\n| Generating {i}/{len(inputs)}')
        _, m, attention = tts_model.generate(x)

        if input_text :
            save_path = f'quick_start/__input_{input_text[:10]}_{tts_k}k.wav'
        else :
            save_path = f'quick_start/{i}_batched{str(batched)}_{tts_k}k.wav'

        save_attention(attention, save_path)

        m = torch.tensor(m).unsqueeze(0)
        m = (m + 4) / 8

        batched = 1

        op = voc_model.generate(m, save_path, batched, hp.voc_target, hp.voc_overlap, hp.mu_law)
        print(type(op))
        sd.play(op,22050)
    print('\n\nDone.\n')
