"""
This code was developed by Anuj Tambwekar, Madhav Kashyap and Rohit Menon of PES University
Refer to the README for the sources of the Deepspeech, Tacotron and WaveRNN implementations/folders
The answer extraction code references the Hugging face run_squad.py example, modified for our use
"""

import sys
import wave
import os
import audioop
import collections

from timeit import default_timer as timer
import numpy as np
import torch
import wikipedia
import spacy
import sounddevice as sd
import soundfile as sf
from gingerit.gingerit import GingerIt
from playsound import playsound

from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertConfig

from deepspeech import Model

from Tacotron_TTS.synthesizer import Synthesizer

from Vocoder_WaveRNN.vocoder_models.fatchord_version import WaveRNN
from Vocoder_WaveRNN import vocoder_hparams as hp
from Vocoder_WaveRNN.vocoder_utils.text import symbols
from Vocoder_WaveRNN.vocoder_models.tacotron import Tacotron
from Vocoder_WaveRNN.vocoder_utils.text import text_to_sequence

spell_check = GingerIt()


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

model_load_start = timer()
model_path = 'BERT/bert_model.bin'
config_file = 'BERT/bert_config.json'
max_answer_length = 30
max_query_length = 64
doc_stride = 128
max_seq_length = 384

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = BertConfig(config_file)
model = BertForQuestionAnswering(config)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model_load_end = timer() - model_load_start
print('Loaded BERT model in {:.3}s.'.format(model_load_end))
print()
tts_choice = int(input("Input 0 for tacotron and 1 for WaveRNN >>> "))

if tts_choice != 1:
    tts_choice = 0

if tts_choice == 0:
    print("Loading regular tacotron....")
    model_load_start = timer()
    synthesizer = Synthesizer()
    synthesizer.load('Tacotron_TTS\\tacotron_model_data\\model.ckpt')
    model_load_end = timer() - model_load_start
    print('Loaded T2S model in {:.3}s.'.format(model_load_end))
else:
    print("Loading fatchord wavernn implementation...")
    model_load_start = timer()
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

    voc_model.restore('Vocoder_WaveRNN//WaveRNN_weights//voc_weights//latest_weights.pyt')

    print('\nInitialising Tacotron_TTS Model...\n')

    # Instantiate Tacotron_TTS Model
    tts_model = Tacotron(embed_dims=hp.tts_embed_dims,
                         num_chars=len(symbols.symbols),
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

    tts_model.restore('Vocoder_WaveRNN//WaveRNN_weights//tts_weights//latest_weights.pyt')
    model_load_end = timer() - model_load_start
    print('Loaded T2S model in {:.3}s.'.format(model_load_end))


def is_whitespace(char):
    if char == " " or char == "\t" or char == "\r" or char == "\n" or ord(char) == 0x202F:
        return True
    return False


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])

    return best_indexes


def check_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


class InputFeatures(object):

    def __init__(self, doc_span_index, tokens, token_is_max_context, token_to_orig_map,
                 input_ids, input_mask, segment_ids, doc_tokens):
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_is_max_context = token_is_max_context
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.doc_tokens = doc_tokens


def input_to_features(question, context):
    """Loads a data file into a list of `InputBatch`s."""
    inputbatch = []
    query_tokens = tokenizer.tokenize(question)
    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]  # reduce question tokens to max input size

    doc_tokens = []
    prev_is_whitespace = True
    for c in context:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []

    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = check_max_context(doc_spans, doc_span_index,
                                               split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        inputbatch.append(InputFeatures(doc_span_index=doc_span_index,
                                        tokens=tokens,
                                        token_is_max_context=token_is_max_context,
                                        token_to_orig_map=token_to_orig_map,
                                        input_ids=input_ids,
                                        input_mask=input_mask,
                                        segment_ids=segment_ids,
                                        doc_tokens=doc_tokens))
    return inputbatch


def bert_predict(context, question):
    input_features = input_to_features(question, context)
    print("Number of batches:", len(input_features))
    predicts = []
    for f in input_features:
        all_input_ids = torch.tensor([f.input_ids], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids], dtype=torch.long)
        input_ids = all_input_ids.to(device)
        input_mask = all_input_mask.to(device)
        segment_ids = all_segment_ids.to(device)

        with torch.no_grad():
            start_logits, end_logits = model(input_ids, segment_ids, input_mask)
        start_logits = start_logits[0].detach().cpu().tolist()
        end_logits = end_logits[0].detach().cpu().tolist()

        output = predict(f, start_logits, end_logits)
        predicts.append(output)

    predicts = sorted(
        predicts,
        key=lambda x: x[1],
        reverse=True)

    return predicts[0][0]


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (index, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = index
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return ns_text, ns_to_s_map

    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def predict(features, start_logit, end_logit):
    n_best_size = 10
    _PrelimPrediction = collections.namedtuple("PrelimPrediction",
                                               ["start_index", "end_index", "start_logit",
                                                "end_logit"])

    _NbestPrediction = collections.namedtuple("NbestPrediction", ["text", "start_logit", "end_logit"])

    prelim_predictions = []
    start_indexes = _get_best_indexes(start_logit, n_best_size)
    end_indexes = _get_best_indexes(end_logit, n_best_size)
    # print(start_indexes)
    # print(end_indexes)

    for start_index in start_indexes:
        for end_index in end_indexes:
            # we remove the indexes which are invalid
            if start_index >= len(features.tokens):
                continue
            if end_index >= len(features.tokens):
                continue
            if start_index not in features.token_to_orig_map:
                continue
            if end_index not in features.token_to_orig_map:
                continue
            if not features.token_is_max_context.get(start_index, False):
                continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue

            prelim_predictions.append(
                _PrelimPrediction(
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=start_logit[start_index],
                    end_logit=end_logit[end_index]))

    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    final_text = "Sorry, I wasn't able to find an answer :("
    score = 0
    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= 1:  # n best size before
            break

        feature = features
        if pred.start_index > 0:  # this is a non-null prediction
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = feature.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, True)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
        else:
            final_text = ""
            seen_predictions[final_text] = True

        score = pred.start_logit + pred.end_logit

        nbest.append(
            _NbestPrediction(
                text=final_text,
                start_logit=pred.start_logit,
                end_logit=pred.end_logit))

    return final_text, score


priorities = {"PERSON": 1, "EVENT": 2, "ORG": 3, "PRODUCT": 4, "LOC": 5, "GPE": 6, "NORP": 7, "LANGUAGE": 8,
              "DATE": 9, "OTHER": 10}

nlp = spacy.load("en_core_web_md")  # Much worse but faster NER with "en_core_web_sm"

LOCALINFO = {"you": 'Data/About_Self',
             "yourself": 'Data/About_Self',
             "You": 'Data/About_Self',
             "Yourself": 'Data/About_Self',
             "PESU": 'Data/About_PESU',
             "PES University": 'Data/About_PESU'}

DATAKEYS = LOCALINFO.keys()


def spacy_ner(text):
    doc = nlp(text)
    tagged_text = []
    for token in doc:
        tagged_text.append((token.text, token.tag_))
    prev = ""
    ents_label_list = []
    for X in doc.ents:
        if X.label_ not in priorities.keys():
            ents_label_list.append((X.text, "OTHER"))
        else:
            if prev == "DATE" and X.label_ == "EVENT":
                old_ent = ents_label_list.pop()
                new_ent = (old_ent[0] + " " + X.text, "EVENT")
                ents_label_list.append(new_ent)
            else:
                ents_label_list.append((X.text, X.label_))
                prev = X.label_

    ents_label_list = sorted(ents_label_list, key=lambda x: priorities[x[1]])
    return ents_label_list, doc  #


def reduced_text(wiki_page, doc, topics):
    text = wiki_page.content
    reduced_passage = ""
    doc_roots = []

    for chunk in doc.noun_chunks:
        doc_roots.append(chunk.root.text)
    # for nkey in topics:
    #    if nkey in doc_roots:
    #        doc_roots.remove(nkey)
    if topics != []:
        if topics[0] in doc_roots:
            doc_roots.remove(topics[0])
    text = text.split('\n')

    if "== See also ==" in text:
        text = text[:text.index("== See also ==")]
    if "== Notes ==" in text:
        text = text[:text.index("== Notes ==")]
    if "== References ==" in text:
        text = text[:text.index("== References ==")]

    for line in text:
        for root in doc_roots:
            if root in line:
                sen = line.split(".")
                for s in sen:
                    if root in s:
                        reduced_passage += s + "."

    return wiki_page.summary + reduced_passage


def get_context(question):
    for corpuskey in DATAKEYS:
        if corpuskey in question:
            text_file = open(LOCALINFO[corpuskey], "r")
            print("Local file used :", LOCALINFO[corpuskey])
            search_passage = text_file.read()
            return search_passage

    topic_list, doc = spacy_ner(question)

    for i in range(len(topic_list)):
        topic_list[i] = topic_list[i][0]

    if len(topic_list) == 0:
        for token in doc:
            if 'NN' in token.tag_:
                topic_list.append(token.lemma_)
        try:
            wiki_page = wikipedia.page(topic_list[0])
        except wikipedia.exceptions.DisambiguationError as err:
            wiki_page = wikipedia.page(err.options[0])

    else:
        try:
            wiki_page = wikipedia.page(topic_list[0])
        except wikipedia.exceptions.DisambiguationError as err:
            wiki_page = wikipedia.page(err.options[0])
    print("Page Used :", wiki_page.title)
    return reduced_text(wiki_page, doc, topic_list)


def get_context_via_search(question):
    for corpuskey in DATAKEYS:
        if corpuskey in question:
            text_file = open(LOCALINFO[corpuskey], "r")
            print("Local file used :", LOCALINFO[corpuskey])
            search_passage = text_file.read()
            return search_passage

    page_list = wikipedia.search(question)
    print("Page used:", page_list[0])
    wiki_page = wikipedia.page(page_list[0])
    # print(wiki_page.content)
    topic_list, doc = spacy_ner(question)
    return reduced_text(wiki_page, doc, topic_list)


directory_in_str = "test_audio/"
directory = os.fsencode(directory_in_str)


def generate_answer(question):
    try:
        context = get_context(question)
        # print(context)
        return bert_predict(context, question)
    except IndexError:
        return "Sorry, couldn't find any pages to search from!"


def test_aud_in():
    tstart = timer()
    audio = "py_rec.wav"
    fs = 44100
    duration = 5  # seconds
    myrecording = sd.rec(duration * fs, samplerate=fs, channels=2, dtype='float32')
    print("Recording Audio")
    sd.wait()
    print("Audio recording complete , Play Audio")
    sd.play(myrecording, fs)
    sd.wait()
    print("Play Audio Complete")
    sf.write(audio, myrecording, fs)

    fin = wave.open(audio, 'rb')
    fs = fin.getframerate()
    if fs != 16000:
        warn = 'Resampling from {}Hz to 16kHz'
        print(warn.format(fs), file=sys.stderr)
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
    print("Infered:", qasked)
    qasked = spell_check.parse(qasked)['result']
    print("Question:", qasked)
    print("Generating answer!")
    gen_start = timer()
    ans = generate_answer(qasked)
    print("Answer:", ans)
    print("Answer generated in {:.3}s.".format(timer() - gen_start))
    print("Generating audio out")
    if tts_choice == 0:
        aud_timer = timer()
        aud_out = synthesizer.synthesize(ans)
        print('Took {:.3}s for audio synthesis.'.format(timer() - aud_timer))
        tot_time = timer() - tstart
        aud_out = np.frombuffer(aud_out, dtype='int32')
        sd.play(aud_out, 10500)
        sd.wait()
        print("Time for sample: {:.3}s.".format(tot_time))
        save_path = f'Tacotron_TTS/Tacotron_outputs/__input_{ans[:10]}.wav'
        sf.write(save_path,aud_out, 10500)
    else:
        input_sequence = text_to_sequence(ans.strip(), hp.tts_cleaner_names)
        aud_timer = timer()
        _, m, attention = tts_model.generate(input_sequence)
        save_path = f'Vocoder_WaveRNN/WaveRNN_outputs/__input_{ans[:10]}.wav'
        m = torch.tensor(m).unsqueeze(0)
        m = (m + 4) / 8
        batched = 1
        op = voc_model.generate(m, save_path, batched, hp.voc_target, hp.voc_overlap, hp.mu_law)
        print('Took {:.3}s for audio synthesis.'.format(timer() - aud_timer))
        sample_time = timer() - aud_timer
        sd.play(op, 22050)
        sd.wait()
        print("Time for sample: {:.3}s.".format(sample_time))


def test_files():
    count = 0
    time_for_all_files = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"):
            start_time = timer()
            fn2 = directory_in_str + filename
            playsound(fn2)
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
            ans = generate_answer(qasked)
            print("Answer:", ans)
            print("Answer generated in {:.3}s.".format(timer() - gen_start))
            print("Generating audio out")
            if tts_choice == 0:
                aud_timer = timer()
                aud_out = synthesizer.synthesize(ans)
                print('Took {:.3}s for audio synthesis.'.format(timer() - aud_timer))
                sample_time = timer() - start_time
                aud_out = np.frombuffer(aud_out, dtype='int32')
                sd.play(aud_out, 10500)
                sd.wait()
                save_path = f'Tacotron_TTS/Tacotron_outputs/__input_{ans[:10]}.wav'
                sf.write(save_path, aud_out, 10500)

            else:
                input_sequence = text_to_sequence(ans.strip(), hp.tts_cleaner_names)
                aud_timer = timer()
                _, m, attention = tts_model.generate(input_sequence)
                save_path = f'Vocoder_WaveRNN/WaveRNN_outputs/__input_{ans[:10]}.wav'
                m = torch.tensor(m).unsqueeze(0)
                m = (m + 4) / 8
                batched = 1
                op = voc_model.generate(m, save_path, batched, hp.voc_target, hp.voc_overlap, hp.mu_law)
                print('Took {:.3}s for audio synthesis.'.format(timer() - aud_timer))
                sample_time = timer() - start_time
                sd.play(op, 22050)
                sd.wait()
            print("Time for sample: {:.3}s.\n".format(sample_time))
            time_for_all_files += sample_time
            count += 1
            print("******")
    print("Time for all samples :", time_for_all_files, "s")
    print("Average time: {:.3}s".format(time_for_all_files / count))


print()
print("############################")
print("0 - test all files in testing folder")
print("1 - use mic input")
print("########################")
print()
while True:
    choice = int(input(">>>"))
    if choice == 0:
        test_files()
    elif choice == 1:
        test_aud_in()
    else:
        print("Exiting...")
        break
