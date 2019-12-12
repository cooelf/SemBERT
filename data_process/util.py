# Do dataset converting & aligning
import sys
import os
import logging
from tqdm import tqdm
from typing import List

from data_process.datasets import ProcessedExample, InputTagFeatures, DocQueSequence, QueryTagSequence, SenTagSequence, \
    DocTagSequence, DocQueExample, QueryExample, DocExample, SenSequence
from tag_model.tagging import get_tags

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def detokenize(token_list):

    tok_text = " ".join(token_list)

    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())

    return tok_text


def get_tag_sequence(sen_sequence: SenSequence, srl_predictor, tag_vocab):
    """input sen_sequence, output sen_tag_sequence"""
    sen_tokens = sen_sequence.sen_tokens
    clean_sen_text = detokenize(sen_tokens)
    sen_words, sen_tags = get_tags(srl_predictor, clean_sen_text, tag_vocab)
    sen_tag_sequence = SenTagSequence(sen_words, sen_tags)

    return sen_tag_sequence


def convert_examples_to_clean_examples(examples, tokenizer):

    clean_examples = []

    for (example_index, example) in enumerate(tqdm(examples, desc="Converting")):
        question_tokens = tokenizer.tokenize(example.question_text)

        clean_question_text = detokenize(question_tokens)

        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                all_doc_tokens.append(sub_token)

        all_sent_tokens = []
        check_all_sent_tokens = []
        clean_doc_text = []
        for tokens in example.sent_token_list:
            sent_tokens = []
            for token in tokens:
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    sent_tokens.append(sub_token)
                    check_all_sent_tokens.append(sub_token)
            sent_text = detokenize(sent_tokens)
            clean_doc_text.append(sent_text)
            all_sent_tokens.append(sent_tokens)

        assert len(all_doc_tokens) == len(check_all_sent_tokens)

        clean_example = ProcessedExample(
                    doc_id=example.doc_id,
                    qas_id=example.qas_id,
                    processed_question_text=clean_question_text,
                    question_tokens=question_tokens,
                    processed_doc_text=clean_doc_text,
                    doc_tokens=all_doc_tokens,
                    sent_token_list=all_sent_tokens)
        clean_examples.append(clean_example)

    return clean_examples


def align_tag_with_token(sent_tag, token_list, sent_words):

    new_sent_tag = []
    cnt_sen_words = 0
    cnt = 0
    flag = False
    sen_word = ""
    while cnt < len(token_list):
        if flag:
            sen_word = sen_word + sent_words[cnt_sen_words]
        else:
            sen_word = sent_words[cnt_sen_words]
        token = token_list[cnt]
        new_token = token
        if len(token) > 1:
            new_token = token.strip('#')
        tmp_cnt = cnt
        cnt += 1
        if not flag:
            new_sent_tag.append(sent_tag[cnt_sen_words])
        else:
            flag = False
        tmp_new_sent_tag = new_sent_tag.copy()

        while (sen_word != new_token) and (cnt < len(token_list)):
            nxt_token = token_list[cnt]
            new_token = new_token.strip('#') + nxt_token.strip('#')
            cnt += 1
            if sent_tag[cnt_sen_words][0] == 'B':
                new_sent_tag.append('I' + sent_tag[cnt_sen_words][1:])
            else:
                new_sent_tag.append(sent_tag[cnt_sen_words])

        if sen_word != new_token:
            # print("=============================")
            # print(sen_word, new_token)
            # print(sent_words, token_list)
            # print("=============================")
            flag = True
            cnt = tmp_cnt
            new_sent_tag = tmp_new_sent_tag.copy()

        cnt_sen_words += 1

    assert len(new_sent_tag) == len(token_list)

    return new_sent_tag


def do_alignment(tag_examples):
    all_aligned_doc_tags = []
    all_aligned_question_tags = []
    prev_doc_id = None
    prev_aligned_doc_tags = []
    for (example_index, tag_example) in enumerate(tag_examples):
        doc_id = tag_example.doc_id
        aligned_question_tags = []
        for question_tag in tag_example.question_tags:
            aligned_question_tag = align_tag_with_token(
                question_tag, tag_example.question_tokens, tag_example.question_words
            )
            assert len(aligned_question_tag) == len(tag_example.question_tokens)
            aligned_question_tags.append(aligned_question_tag)

        if prev_doc_id != doc_id:
            prev_doc_id = doc_id
            aligned_doc_tags = []
            for sent_tags, sent_token, sent_word in zip(
                    tag_example.doc_tags, tag_example.doc_tokens, tag_example.doc_words):
                aligned_sent_tags = []
                for doc_tag in sent_tags:
                    aligned_sent_tag = align_tag_with_token(
                        doc_tag, sent_token, sent_word
                    )
                    aligned_sent_tags.append(aligned_sent_tag)
                aligned_doc_tags.append(aligned_sent_tags)
            prev_aligned_doc_tags = aligned_doc_tags
        else:
            aligned_doc_tags = prev_aligned_doc_tags
        all_aligned_question_tags.append(aligned_question_tags)
        all_aligned_doc_tags.append(aligned_doc_tags)

    return all_aligned_question_tags, all_aligned_doc_tags


def choose_tag_set(tag_sets):
    cnt_tag = 0
    tag_ix = 0
    for ix, tag_set in enumerate(tag_sets):
        cnt_tmp_tag = 0
        for tag in tag_set:
            if tag != 'O':
                cnt_tmp_tag = cnt_tmp_tag + 1
        if cnt_tmp_tag > cnt_tag:
            cnt_tag = cnt_tmp_tag
            tag_ix = ix
    chosen_tag_set = tag_sets[tag_ix]

    return chosen_tag_set


def do_padding(tag_sets, max_num_aspect, cnt):
    if len(tag_sets) >= 100:
        print("************")
        print(len(tag_sets))
        print(tag_sets)
        print("************")

    cnt[len(tag_sets)] += 1
    if len(tag_sets) > max_num_aspect:
        return tag_sets[:max_num_aspect]
    # choose tag_set with least number 'O' as the padding tag_set
    tag_set_to_copy = choose_tag_set(tag_sets)
    while len(tag_sets) < max_num_aspect:
        tag_sets.append(tag_set_to_copy)

    return tag_sets


def do_aspect_padding(question_tag_sets, doc_tag_sets, max_num_aspect, cnt):
    padding_question_tag_sets = do_padding(question_tag_sets, max_num_aspect, cnt)

    padding_doc_tag_sets = []
    for sent_tag_sets in doc_tag_sets:
        padding_sent_tag_sets = do_padding(sent_tag_sets, max_num_aspect, cnt)
        padding_doc_tag_sets.append(padding_sent_tag_sets)

    # merge all sent_tag_sets with same idx to one doc_tag_set, will get max_num_aspect doc_tag_sets

    all_merge_doc_tag_sets = []
    for idx in range(max_num_aspect):
        merge_doc_tag_sets = []
        for padding_doc_tag_set in padding_doc_tag_sets:
            merge_doc_tag_sets.extend(padding_doc_tag_set[idx])
        all_merge_doc_tag_sets.append(merge_doc_tag_sets)

    assert len(padding_question_tag_sets) == max_num_aspect
    assert len(all_merge_doc_tag_sets) == max_num_aspect

    return padding_question_tag_sets, all_merge_doc_tag_sets


def convert_tag_examples_to_tag_features(tag_examples, features, max_query_length, max_seq_length, max_num_aspect,
                                         tag_tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    cnt = [0]*100
    tag_features = []
    print("Do alignment...")
    all_aligned_question_tags, all_aligned_doc_tags = do_alignment(tag_examples)

    assert len(all_aligned_question_tags) == len(tag_examples)
    assert len(all_aligned_doc_tags) == len(tag_examples)

    for (feature_index, feature) in enumerate(tqdm(features, desc="Converting tags")):
        example_index = feature.example_index
        question_tags = all_aligned_question_tags[example_index]
        doc_tags = all_aligned_doc_tags[example_index]

        question_tag_sets, doc_tag_sets = do_aspect_padding(question_tags, doc_tags, max_num_aspect, cnt)

        input_tags = []
        input_tag_ids = []
        for (question_tag_set, doc_tag_set) in zip(question_tag_sets, doc_tag_sets):
            if len(question_tag_set) > max_query_length:
                question_tag_set = question_tag_set[0:max_query_length]
            split_doc_tag_set = []
            for doc_index in feature.doc_split_index:
                split_doc_tag_set.append(doc_tag_set[doc_index])
            input_tag = ["[CLS]"] + question_tag_set + ["[SEP]"] + split_doc_tag_set + ["[SEP]"]
            input_tags.append(input_tag)
            assert len(input_tag) == len(feature.tokens)

            input_tag_id = tag_tokenizer.convert_tags_to_ids(input_tag)
            # Zero-pad up to the sequence length.
            padding_id = [0] * (max_seq_length - len(input_tag_id))
            input_tag_id += padding_id
            assert len(input_tag_id) == len(feature.input_ids)
            input_tag_ids.append(input_tag_id)

        # input_tag_ids.shape should be (max_num_aspect, max_seq_length)

        if example_index < 20:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % unique_id)
            logger.info("example_index: %s" % example_index)
            logger.info("tokens: %s" % " ".join(feature.tokens))
            logger.info("tags: %s" % " ".join(input_tags[0]))
            logger.info("input_tag_ids: %s" % " ".join([str(x) for x in input_tag_ids[0]]))
            logger.info(
                "input_mask: %s" % " ".join([str(x) for x in feature.input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in feature.segment_ids]))

        tag_features.append(
            InputTagFeatures(
                unique_id=feature.unique_id,
                example_index=example_index,
                tags=input_tags,
                input_tag_ids=input_tag_ids,
                input_mask=feature.input_mask)
        )
        unique_id += 1

    print("===============")
    print(cnt)
    print("===============")

    return tag_features


def iter_data(datas, n_batch=128, truncate=False, verbose=False, max_batches=float("inf")):
    n = len(datas)
    if truncate:
        n = (n//n_batch)*n_batch
    n = min(n, max_batches*n_batch)
    n_batches = 0
    if verbose:
        f = sys.stderr
    else:
        f = open(os.devnull, 'w')
    for i in tqdm(range(0, n, n_batch), total=n//n_batch, file=f, ncols=80, leave=False):
        if n_batches >= max_batches: raise StopIteration

        yield datas[i:i+n_batch]

        n_batches += 1
