# -*- coding: utf-8 -*-

import os
import pickle
import tensorflow as tf
from utils import create_model, get_logger
from model import Model
from loader import input_from_line
from utils import load_config
from config.bert_rela_cls_config import bert_rela_cls_params


def main(args):
    config = load_config(args["config_file"])
    logger = get_logger(args["log_file"])
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(args["map_file"], "rb") as f:
        tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, args["ckpt_path"], config, logger)
        while True:
            line = input("input sentence, please:")
            result = model.evaluate_line(sess, input_from_line(line,
                                                               args["max_seq_len"],
                                                               tag_to_id), id_to_tag)
            print(result)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = bert_rela_cls_params()
    main(args)