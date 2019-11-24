# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertConfig
import torch
import wikipedia
import spacy
import collections
import re

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

class ActionHelloWorld(Action):
    def name(self) -> Text:
        return "action_hello_world"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message("Hello World!")
        return []


class ActionAnswerQuestion(Action):
    def name(self) -> Text:
        return "action_answer_question"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        question, gotcontext = insert_context(tracker.latest_message['text'], tracker)
        anstype = answer_type(question)

        #dispatcher.utter_message("You said: "+tracker.latest_message['text'])
        #dispatcher.utter_message("The answer to that question is....")
        anstext, topic = generate_answer(question)
        dispatcher.utter_message(anstext)
        if anstext == "Sorry, couldn't find any pages to search from!":
            return []
        if anstype == "NNP":
            return [SlotSet("person_name", anstext)]
        if anstype == "NN":
            if len(anstext.split()) <= len(topic.split()):
                return [SlotSet("thing_name", anstext)]
            else:
                return [SlotSet("thing_name", topic)]

        return []


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
            return search_passage, corpuskey

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
    return reduced_text(wiki_page, doc, topic_list), topic_list[0]


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


def generate_answer(question):
    try:
        context, topic = get_context(question)
        # print(context)
        return bert_predict(context, question), topic
    except IndexError:
        return "Sorry, couldn't find any pages to search from!", "N/A"

def answer_type(question):
    question = question.lower().split()
    questionwords = {"who": "NNP", "what": "NN", "when": "DATE", "how": "method", "why": "reason"}
    for qw in questionwords.keys():
        if qw in question:
            return questionwords[qw]

def splitquestion(question):
    question = question.lower()
    questionwords = ["who", "what", "when", "how", "why"]
    words = question.split()
    qw_count = 0
    for w in words:
        if w in questionwords:
            qw_count +=1
    if qw_count <= 1:
        return [question] , qw_count

def insert_context(question, tracker):
        # question = original_question.lower()
        current_slots = tracker.current_slot_values()
        pronouns = [r"he", r"she", r"they", r"it", r"him", r"her", r"they", r"them", r"his"]
        for pn in pronouns:
            if re.search(r'\b'+pn + r'\b',question):
                if pn in ["he","she","him","her","they","them"]:
                    if current_slots['person_name'] != "None":
                        question = re.sub(r"\b"+pn +r"\b" , current_slots['person_name'], question,1)
                        print("Had oontext. Question is now:",question)
                    else:
                        print("Don't have context")
                        return question, 0
                if pn in ["his"]:
                    if current_slots['person_name'] != "None":
                        question = re.sub(r"\b"+pn +r"\b" , current_slots['person_name'] + "'s", question,1)
                        print("Had oontext. Question is now:",question)
                    else:
                        print("Don't have context")
                        return question, 0

                if pn in ["it"]:
                    if current_slots['thing_name'] != "None":
                        question = re.sub(r"\b"+pn +r"\b" , current_slots['thing_name'], question,1)
                        print("Had oontext. Question is now:",question)
                    else:
                        print("Don't have context")
                        return question, 0
        return question,1

