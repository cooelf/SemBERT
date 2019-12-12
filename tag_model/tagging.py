from allennlp.predictors import Predictor
import json

class SRLPredictor(object):
    def __init__(self,SRL_MODEL_PATH):
        # use the model from allennlp for simlicity.
        self.predictor = Predictor.from_path(SRL_MODEL_PATH)
        self.predictor._model = self.predictor._model.cuda()  # this can only support GPU computation

    def predict(self, sent):
        return self.predictor.predict(sentence=sent)


def get_tags(srl_predictor, tok_text, tag_vocab):
    if srl_predictor == None:
        srl_result = json.loads(tok_text)  # can load a pre-tagger dataset for quick evaluation
    else:
        srl_result = srl_predictor.predict(tok_text)
    sen_verbs = srl_result['verbs']
    sen_words = srl_result['words']

    sent_tags = []
    if len(sen_verbs) == 0:
        sent_tags = [["O"] * len(sen_words)]
    else:
        for ix, verb_tag in enumerate(sen_verbs):
            sent_tag = sen_verbs[ix]['tags']
            for tag in sent_tag:
                if tag not in tag_vocab:
                    tag_vocab.append(tag)
            sent_tags.append(sent_tag)

    return sen_words, sent_tags
