# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import pickle
import os
import logging
from os.path import join, exists
from os import makedirs


logger = logging.getLogger(__name__)
TAG_VOCAB = ["[PAD]",'[CLS]', '[SEP]', '[UNK]','B-V', 'I-V', 'B-ARG0', 'I-ARG0', 'B-ARG1', 'I-ARG1', 'B-ARG2', 'I-ARG2', 'B-ARG4', 'I-ARG4', 'B-ARGM-TMP', 'I-ARGM-TMP', 'B-ARGM-LOC', 'I-ARGM-LOC', 'B-ARGM-CAU', 'I-ARGM-CAU', 'B-ARGM-PRP', 'I-ARGM-PRP', 'O']
#or load the full vocab of SRL, this will not affect the performance too much.
# If using all the vocab based on each dataset, de-comment out the following line in the run_xxx.py file: TagTokenizer.make_tag_vocab("tag_vocab", tag_vocab)

def load_tag_vocab(tag_vocab_file):
    vocab_list = ["[PAD]", "[CLS]", "[SEP]"]
    with open(tag_vocab_file, 'rb') as f:
        vocab_list.extend(pickle.load(f))
    return vocab_list


class TagTokenizer(object):
    @staticmethod
    def make_tag_vocab(corpus_dir, tag_vocab):
        vocab_file = join(corpus_dir, "tag_vocab.pkl")

        if not exists(corpus_dir):
            makedirs(corpus_dir)

        with open(vocab_file, 'wb') as f:
            pickle.dump(tag_vocab, f)

    def __init__(self, corpus_dir):
        # vocab_file = join(corpus_dir, "tag_vocab.pkl")
        # if not os.path.isfile(vocab_file):
        #     self.tag_vocab = TAG_VOCAB
        # else:
        #     self.tag_vocab = load_tag_vocab(vocab_file)
        #     print("loading tag vocab")
        # self.ids_to_tags = collections.OrderedDict(
        #     [(ids, tag) for ids, tag in enumerate(TAG_VOCAB)])
        self.tag_vocab = TAG_VOCAB
        self.ids_to_tags = collections.OrderedDict(
            [(ids, tag) for ids, tag in enumerate(TAG_VOCAB)])

    def convert_tags_to_ids(self, tags):
        """Converts a sequence of tags into ids using the vocab."""
        ids = []
        for tag in tags:
            # may appear some tag not in the previous vocab due to alignment
            if tag not in self.tag_vocab:
                tag = '[UNK]'
            # if tag not in TAG_VOCAB:
            #     tag = 'O'
            ids.append(self.tag_vocab.index(tag))
            #ids.append(TAG_VOCAB.index(tag))

        return ids

    def convert_ids_to_tags(self, ids):
        """Converts a sequence of ids into tags using the vocab."""
        tags = []
        for i in ids:
            tags.append(self.ids_to_tags[i])
        return tags
