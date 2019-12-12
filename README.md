# Semantics-aware BERT for Language Understanding

Pytorch codes for the paper **Semantics-aware BERT for Language Understanding** in AAAI 2020

### **Overview**

![](SemBERT.png)

## Requirements

(Our experiment environment for reference)

Python 3.6+
PyTorch (1.0.0)
AllenNLP (0.8.1)

## Datasets
GLUE data can be downloaded from [GLUE data](https://gluebenchmark.com/tasks) by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpack it to directory <u>glue_data</u>.
We provide an example data sample in <u>glue_data/MNLI</u> to show how SemBERT works.

## Instructions
This repo shows the example implementation of SemBERT for NLU tasks.
We basically used the pre-trained BERT uncased models so do not forget to pass the parameter `--do_lower_case`
An example script is

```shell
CUDA_VISIBLE_DEVICES=0 \
python run_classifier.py \
--data_dir glue_data/MNLI/ \
--eval_batch_size 32 \
--max_seq_length 200 \
--bert_model bert-base-uncased \
--do_lower_case \
--task_name mnli \
--do_train \
--do_eval \
--do_predict \
--output_dir glue/base_mnli \
--learning_rate 3e-5
```

The output pred file can be directly used for GLUE online submission and evaluation.

We provde two kinds of semantic labeling method, 

* **online**: each word sequence are passed to label module to obtain the tags which could be used for online prediction. This would be time-consuming for large corpus. See  *tag_model/tagging.py*

  If you want to use the online one, please specify the `--tagger_path` parameter in the run.py file.

* **offline**: the current one that pre-process the datasets and save them for later loading for training and evaluation. See *tag_model/tagger_offline.py*

  Our labeled data can be downloaded here for quick start.

  [https://drive.google.com/file/d/1B-_IRWRvR67eLdvT6bM0b2OiyvySkO-x/view?usp=sharing](https://drive.google.com/file/d/1B-_IRWRvR67eLdvT6bM0b2OiyvySkO-x/view?usp=sharing)

Note this repo is based on the offline version, so that the column id/index in the data-processor would be slightly different from the original, which is like this:

text_a = line[-3]
text_b = line[-2]
label = line[-1]

If you use the original data <u>instead of</u> our preprocessed one by tag_model/tagger_offline.py, please modify the index according to the dataset structure.

### SRL model

The SRL model in this implementation used the [ELMo-based SRL model](https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz)  from [AllenNLP](https://github.com/allenai/allennlp). 

Recently, there is a new [BERT-based model](https://github.com/allenai/allennlp), which is a nice alternative. 

### Reference

Please kindly cite this paper in your publications if it helps your research:

```
@inproceedings{zhang2020SemBERT,
	title={Semantics-aware {BERT} for language understanding},
	author={Zhang, Zhuosheng and Wu, Yuwei and Zhao, Hai and Li, Zuchao and Zhang, Shuailiang and Zhou, Xi and Zhou, Xiang},
  	booktitle={the Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-2020)},
	year={2020}
}
```