# coding:utf-8
import json
from allennlp.predictors import Predictor
import csv
import sys
import time
from tqdm import tqdm


class srl_labeler():
    def __init__(self):
        #use the model from allennlp for simlicity.
        self.predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")
        self.predictor._model = self.predictor._model.cuda()

    def post_allen(self, sentence):
        results = self.predictor.predict_batch_json(sentence)
        return results

    def annoatate(self, seg_file, batch, index_a, index_b, index_label, has_b, is_test=False):
        print(index_a, index_b, index_label, has_b)
        with open(seg_file, 'r', encoding='utf-8') as data_file, open(seg_file + "_tag", 'w',
                                                                      encoding='utf-8') as new_file:
            reader = csv.reader(data_file, delimiter="\t", quotechar=None)
            all_data = []
            for line in reader:
                all_data.append(line)
            print(all_data[0])
            data_chunk = [all_data[i:i + batch] for i in range(0, len(all_data), batch)]
            for data_sent in tqdm(data_chunk):
                text_a = [{"sentence": sent[index_a]} for sent in data_sent]
                text_a_tags = self.post_allen(text_a)
                if has_b:
                    text_b = [{"sentence": sent[index_b]} for sent in data_sent]
                    text_b_tags = self.post_allen(text_b)
                    for raw_data, tag_a, tag_b in zip(data_sent, text_a_tags, text_b_tags):
                        line = str(raw_data[0]) + "\t" + raw_data[index_a] + "\t" + raw_data[
                            index_b] + "\t" + json.dumps(tag_a) + "\t" + json.dumps(tag_b)
                        if not is_test:
                            line = line + "\t" + str(raw_data[index_label]) + "\n"
                        else:
                            line = line + "\n"
                        new_file.write(line)
                else:
                    for raw_data, tag_a in zip(data_sent, text_a_tags):
                        line = str(raw_data[0]) + "\t" + raw_data[index_a] + "\t" + json.dumps(tag_a)
                        if not is_test:
                            line = line + "\t" + str(raw_data[index_label]) + "\n"
                        else:
                            line = line + "\n"
                        new_file.write(line)

    def seg_data(self, num_seg, file_path):
        with open(file_path, 'r', encoding='utf-8') as data_file:
            print("Reading instances from  MNLI dataset at: %s", file_path)
            all_data = data_file.readlines()[1:]  # for MNLI
        print("data length:" + str(len(all_data)))
        print("Seg count:" + str(num_seg))
        if (len(all_data) % num_seg):
            separate_length = int(len(all_data) / num_seg)
        else:
            separate_length = int(len(all_data) / num_seg) + 1
        print("separete length:" + str(separate_length))
        separate_data = [all_data[i:i + separate_length] for i in range(0, len(all_data), separate_length)]
        for i, separate in enumerate(separate_data):
            if len(separate_data) > 1 and i == len(separate_data) - 1:
                print(file_path + "_addseg_%d" % (i - 1))
                with open(file_path + "_seg_%d" % (i - 1), 'a', encoding="utf-8") as new_file:
                    new_file.writelines(separate)
            else:
                print(file_path + "_seg_%d" % (i))
                with open(file_path + "_seg_%d" % (i), 'w', encoding="utf-8") as new_file:
                    new_file.writelines(separate)


srl_labeler = srl_labeler()
folder = "RTE"
batch_size = 40
index_a = 1
index_b = 2
index_label = -1
has_b = True
data_path = "glue_data/" + folder + "/train.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b)
data_path = "glue_data/" + folder + "/dev.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b)
data_path = "glue_data/" + folder + "/test.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b, True)

folder = "WNLI"
batch_size = 80
index_a = 1
index_b = 2
index_label = -1
has_b = True
data_path = "glue_data/" + folder + "/train.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b)
data_path = "glue_data/" + folder + "/dev.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b)
data_path = "glue_data/" + folder + "/test.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b, True)

folder = "STS-B"
batch_size = 80
index_a = 7
index_b = 8
index_label = -1
has_b = True
data_path = "glue_data/" + folder + "/train.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b)
data_path = "glue_data/" + folder + "/dev.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b)
data_path = "glue_data/" + folder + "/test.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b, True)

folder = "SST-2"
batch_size = 80
index_a = 0
index_b = 0
index_label = -1
has_b = False
data_path = "glue_data/" + folder + "/train.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b)
data_path = "glue_data/" + folder + "/dev.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b)
data_path = "glue_data/" + folder + "/test.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b, True)

folder = "QQP"
batch_size = 40
index_a = 3
index_b = 4
index_label = -1
has_b = True
data_path = "glue_data/" + folder + "/train.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b)
data_path = "glue_data/" + folder + "/dev.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b)
data_path = "glue_data/" + folder + "/test.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b, True)

folder = "QNLI"
batch_size = 80
index_a = 1
index_b = 2
index_label = -1
has_b = True
data_path = "glue_data/" + folder + "/train.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b)
data_path = "glue_data/" + folder + "/dev.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b)
data_path = "glue_data/" + folder + "/test.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b, True)

folder = "MRPC"
batch_size = 80
index_a = 3
index_b = 4
index_label = 0
has_b = True
data_path = "glue_data/" + folder + "/train.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b)
data_path = "glue_data/" + folder + "/dev.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b)
data_path = "glue_data/" + folder + "/test.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b, True)

folder = "diagnostic"
batch_size = 80
index_a = 1
index_b = 2
index_label = -1
has_b = True
data_path = "glue_data/" + folder + "/diagnostic.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b, True)

folder = "CoLA"
batch_size = 80
index_a = 3
index_b = 0
index_label = 1
has_b = False
data_path = "glue_data/" + folder + "/train.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b)
data_path = "glue_data/" + folder + "/dev.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b)
data_path = "glue_data/" + folder + "/test.tsv"
srl_labeler.annoatate(data_path, batch_size, index_a, index_b, index_label, has_b, True)