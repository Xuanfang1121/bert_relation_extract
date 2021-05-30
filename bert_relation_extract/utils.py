# -*- coding: utf-8 -*-
import os
import json
import shutil
import logging
import codecs

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report


models_path = "./models"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def evaluate_model_score(targets, pres, id2tag):
    """
    :param targets: 真实的label，list
    :param pres: 模型预测得到的label
    :param id2tag: dict
    :return: precision, recall, f1_score
    """
    recall = recall_score(np.array(targets), np.array(pres), average="macro", zero_division=0)
    precision = precision_score(np.array(targets), np.array(pres), average="macro", zero_division=0)
    f1 = f1_score(np.array(targets), np.array(pres), average="macro", zero_division=0)
    target_names = [id2tag[i] for i in range(len(id2tag))]
    print(classification_report(targets, pres, target_names=target_names))
    if np.isnan(recall):
        recall = 0.001
    if np.isnan(precision):
        precision = 0.001
    if np.isnan(f1):
        f1 = 0.001
    return recall, precision, f1


def save_predict_result(results, path):
    output_file = os.path.join(path, "model_predict.utf8")
    with open(output_file, "w", encoding='utf-8') as f:
        for line in results:
            f.write(line[0] + "\n")


def print_config(config, logger):
    """
    Print configuration of the model
    """
    for k, v in config.items():
        logger.info("{}:\t{}".format(k.ljust(15), v))


def make_path(params):
    """
    Make folders for training and evaluation
    """
    if not os.path.isdir(params["result_path"]):
        os.makedirs(params["result_path"])
    if not os.path.isdir(params["ckpt_path"]):
        os.makedirs(params["ckpt_path"])
    if not os.path.isdir(params["log_path"]):
        os.makedirs(params["log_path"])


def clean(params):
    """
    Clean current folder
    remove saved model and training log
    """
    if os.path.isfile(params.vocab_file):
        os.remove(params.vocab_file)

    if os.path.isfile(params.map_file):
        os.remove(params.map_file)

    if os.path.isdir(params.ckpt_path):
        shutil.rmtree(params.ckpt_path)

    if os.path.isdir(params.summary_path):
        shutil.rmtree(params.summary_path)

    if os.path.isdir(params.result_path):
        shutil.rmtree(params.result_path)

    if os.path.isdir("log"):
        shutil.rmtree("log")

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")

    if os.path.isfile(params.config_file):
        os.remove(params.config_file)

    if os.path.isfile(params.vocab_file):
        os.remove(params.vocab_file)


def save_config(config, config_file):
    """
    Save configuration of the model
    parameters are stored in json format
    """
    with open(config_file, "w", encoding="utf8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def load_config(config_file):
    """
    Load configuration of the model
    parameters are stored in json format
    """
    with open(config_file, encoding="utf8") as f:
        return json.load(f)


def convert_to_text(line):
    """
    Convert conll data to text
    """
    to_print = []
    for item in line:

        try:
            if item[0] == " ":
                to_print.append(" ")
                continue
            word, gold, tag = item.split(" ")
            if tag[0] in "SB":
                to_print.append("[")
            to_print.append(word)
            if tag[0] in "SE":
                to_print.append("@" + tag.split("-")[-1])
                to_print.append("]")
        except:
            print(list(item))
    return "".join(to_print)


def save_model(sess, model, path, logger, global_steps):
    checkpoint_path = os.path.join(path, "ner.ckpt")
    model.saver.save(sess, checkpoint_path, global_step = global_steps)
    logger.info("model saved")


def create_model(session, Model_class, path, config, logger):
    # create model, reuse parameters if exists
    model = Model_class(config)

    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        #saver = tf.train.import_meta_graph('ckpt/ner.ckpt.meta')
        #saver.restore(session, tf.train.latest_checkpoint("ckpt/"))
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def result_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item

def bio_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    iCount = 0
    entity_tag = ""
    #assert len(string)==len(tags), "string length is: {}, tags length is: {}".format(len(string), len(tags))

    for c_idx in range(len(tags)):
        c, tag = string[c_idx], tags[c_idx]
        if c_idx < len(tags)-1:
            tag_next = tags[c_idx+1]
        else:
            tag_next = ''

        if tag[0] == 'B':
            entity_tag = tag[2:]
            entity_name = c
            entity_start = iCount
            if tag_next[2:] != entity_tag:
                item["entities"].append({"word": c, "start": iCount, "end": iCount + 1, "type": tag[2:]})
        elif tag[0] == "I":
            if tag[2:] != tags[c_idx-1][2:] or tags[c_idx-1][2:] == 'O':
                tags[c_idx] = 'O'
                pass
            else:
                entity_name = entity_name + c
                if tag_next[2:] != entity_tag:
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": iCount + 1, "type": entity_tag})
                    entity_name = ''
        iCount += 1
    return item


def convert_single_example(char_line, tag_to_id, max_seq_length, tokenizer, label_line):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为lb
    """
    text_list = char_line.split(' ')
    labels = [label_line]  # me modify

    tokens = []
    for i, word in enumerate(text_list):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = [tag_to_id[labels[0]]]
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)

    # padding
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        # label_ids.append(0)
        ntokens.append("**NULL**")

    return input_ids, input_mask, segment_ids, label_ids