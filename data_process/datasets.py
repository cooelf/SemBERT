from typing import List
from tag_model.tag_tokenization import TagTokenizer


class SenSequence(object):
    def __init__(self, sen_ix, sen_text: str, sen_tokens: List[str]):
        self.sen_ix = sen_ix
        self.sen_text = sen_text
        self.sen_tokens = sen_tokens


class QuerySequence(SenSequence):
    def __init__(self, query_id, query_text: str, query_tokens: List[str]):
        super().__init__(query_id, query_text, query_tokens)


class DocSequence(object):
    def __init__(self, doc_id, doc_tokens, sen_sequence_list: List[SenSequence]):
        self.doc_id = doc_id
        self.doc_tokens = doc_tokens
        self.sen_sequence_list = sen_sequence_list


class SenTagSequence(object):
    def __init__(self, sen_words: List[str], sen_tags_list: List[List[str]]):
        self.sen_words = sen_words
        self.sen_tags_list = sen_tags_list

    def do_alignment(self, tokens: List[str]):
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

        all_aligned_sen_tags = []
        for sen_tags in self.sen_tags_list:
            # print("================")
            # print(self.sen_words)
            # print(sen_tags)
            # print("================")
            assert len(self.sen_words) == len(sen_tags)

            aligned_sen_tags = align_tag_with_token(
                sen_tags, tokens, self.sen_words
            )
            assert len(aligned_sen_tags) == len(tokens)
            all_aligned_sen_tags.append(aligned_sen_tags)

        self.sen_tags_list = all_aligned_sen_tags

    def aspect_padding(self, max_num_aspect):
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

        if len(self.sen_tags_list) > max_num_aspect:
            self.sen_tags_list = self.sen_tags_list[:max_num_aspect]
        # choose tag_set with least number 'O' as the padding tag_set
        tag_set_to_copy = choose_tag_set(self.sen_tags_list)
        while len(self.sen_tags_list) < max_num_aspect:
            self.sen_tags_list.append(tag_set_to_copy)

    def convert_to_ids(self, tokenizer: TagTokenizer):
        sen_tag_ids_list = []
        for sen_tags in self.sen_tags_list:
            sen_tag_ids = tokenizer.convert_tags_to_ids(sen_tags)
            sen_tag_ids_list.append(sen_tag_ids)
        return sen_tag_ids_list


class QueryTagSequence(SenTagSequence):
    def __init__(self, query_words, query_tags_list: List[List[str]]):
        super().__init__(query_words, query_tags_list)


class DocTagSequence(object):
    def __init__(self, sen_tag_sequence_list: List[SenTagSequence]):
        self.sen_tag_sequence_list = sen_tag_sequence_list
        self.doc_tags_list = None

    def do_alignment(self, doc_sequence: DocSequence):
        sen_sequence_list = doc_sequence.sen_sequence_list
        for sen_ix, sen_tag_sequence in enumerate(self.sen_tag_sequence_list):
            sen_sequence = sen_sequence_list[sen_ix]
            # here the sen_tokens of sen_sequence must be tokenized token list
            # print("sen_tag:", sen_tag_sequence.sen_tags_list[0])
            sen_tag_sequence.do_alignment(sen_sequence.sen_tokens)

    def aspect_padding(self, max_num_aspect):
        for sen_tag_sequence in self.sen_tag_sequence_list:
            sen_tag_sequence.aspect_padding(max_num_aspect)

    def convert_to_ids(self, tokenizer: TagTokenizer):
        # assume all sen_tags_lists has same len of sen_tags_list
        doc_tag_ids_list = []
        doc_tags_list = []
        sen_tag_ids_list = []
        all_sen_tag_ids_list = []
        for sen_tag_sequence in self.sen_tag_sequence_list:
            # sen_tag_ids_list: List[List[Int]]
            sen_tag_ids_list = sen_tag_sequence.convert_to_ids(tokenizer)
            all_sen_tag_ids_list.append(sen_tag_ids_list)

        num_aspect = len(sen_tag_ids_list)
        for idx in range(num_aspect):
            # merge all sent_tag_sets with same idx to one doc_tag_set, will get num_aspect doc_tag_sets
            merge_doc_tag_id = []
            merge_doc_tag = []
            for sen_ix, sen_tag_ids in enumerate(all_sen_tag_ids_list):
                merge_doc_tag_id.extend(sen_tag_ids[idx])
                merge_doc_tag.extend(self.sen_tag_sequence_list[sen_ix].sen_tags_list[idx])
            doc_tag_ids_list.append(merge_doc_tag_id)
            doc_tags_list.append(merge_doc_tag)

        self.doc_tags_list = doc_tags_list

        return doc_tag_ids_list


class DocExample(object):
    def __init__(self, doc_id, doc_sequence: DocSequence, doc_tag_sequence: DocTagSequence):
        self.doc_id = doc_id
        self.doc_sequence = doc_sequence
        self.doc_tag_sequence = doc_tag_sequence


class QueryExample(object):
    def __init__(self, query_id, query_sequence, query_tag_sequence):
        self.query_id = query_id
        self.query_sequence = query_sequence
        self.query_tag_sequence = query_tag_sequence


class DocQueExample(object):
    def __init__(self, doc_example: DocExample, query_example: QueryExample):
        self.doc_example = doc_example
        self.query_example = query_example


class DocQueSequence(object):
    def __init__(self, doc_sequence: DocSequence, query_sequence: QuerySequence):
        self.doc_sequence = doc_sequence
        self.query_sequence = query_sequence


class Example(object):
    def __init__(self,
                 doc_id,
                 qas_id,
                 question_text,
                 doc_tokens,
                 sent_token_list
                 ):
        self.doc_id = doc_id
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens

        # for aligning each sent
        self.sent_token_list = sent_token_list


class TagExample(object):
    def __init__(self,
                 doc_id,
                 qas_id,
                 question_tags,
                 question_words,
                 question_tokens,
                 doc_tags,
                 doc_words,
                 doc_tokens
                 ):
        self.doc_id = doc_id
        self.qas_id = qas_id
        self.question_tags = question_tags
        self.question_words = question_words
        self.question_tokens = question_tokens
        self.doc_tags = doc_tags
        self.doc_words = doc_words
        self.doc_tokens = doc_tokens


class ProcessedExample(Example):
    """A single training/test example for the detokenized dataset."""

    def __init__(self, doc_id, qas_id, processed_question_text,
                 question_tokens, processed_doc_text, doc_tokens, sent_token_list):
        super().__init__(doc_id, qas_id, processed_question_text,
                         doc_tokens, sent_token_list)

        self.question_tokens = question_tokens
        self.doc_text = processed_doc_text


class InputFeatures(object):
    def __init__(self,
                 unique_id,
                 example_index,
                 tokens: List[str],
                 input_ids: List[int],
                 input_mask,
                 segment_ids
                 ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class InputTagFeatures(object):

    def __init__(self,
                 unique_id,
                 example_index,
                 tags: List[str],
                 input_tag_ids: List[int],
                 input_mask
                 ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tags = tags
        self.input_tag_ids = input_tag_ids
        self.input_mask = input_mask


