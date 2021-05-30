# encoding=utf8
import math
import pickle
import os

import tensorflow as tf
import numpy as np
from model import Model

from loader import load_sentences
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, create_model, save_model
from utils import print_config, save_config, save_predict_result, evaluate_model_score
from data_utils import BatchManager
from config.bert_rela_cls_config import bert_rela_cls_params


def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    pred_results, rela_labels, pred_labels = model.evaluate(sess, data, id_to_tag)
    save_predict_result(pred_results, args["result_path"])
    recall, precision, f1 = evaluate_model_score(rela_labels, pred_labels, id_to_tag)

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train(args):
    # load data sets
    train_sentences = load_sentences(args["train_file"], args["lower"], args["zeros"])
    dev_sentences = load_sentences(args["dev_file"], args["lower"], args["zeros"])
    print("dev_sentences: ", dev_sentences[:5])
    test_sentences = load_sentences(args["test_file"], args["lower"], args["zeros"])

    # create maps if not exist
    if not os.path.isfile(args["map_file"]):
        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(args["map_file"], "wb") as f:
            pickle.dump([tag_to_id, id_to_tag], f)
    else:
        with open(args["map_file"], "rb") as f:
            tag_to_id, id_to_tag = pickle.load(f)
    print("tag2id: ", tag_to_id)
    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(
        train_sentences, args["max_seq_len"], tag_to_id, args["lower"]
    )
    dev_data = prepare_dataset(
        dev_sentences, args["max_seq_len"], tag_to_id, args["lower"]
    )
    test_data = prepare_dataset(
        test_sentences, args["max_seq_len"], tag_to_id, args["lower"]
    )
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_data), len(test_data)))

    train_manager = BatchManager(train_data, args["batch_size"])
    dev_manager = BatchManager(dev_data, args["batch_size"])
    test_manager = BatchManager(test_data, args["batch_size"])
    # make path for store log and model if not exist
    make_path(args)
    # save model config
    save_config(args, args["config_file"])

    log_path = os.path.join(args["log_path"], args["log_file"])
    logger = get_logger(log_path)
    print_config(args, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, args["ckpt_path"], args, logger)

        logger.info("start training")
        loss = []
        for i in range(args["max_epoch"]):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)

                loss.append(batch_loss)
                if step % args["steps_check"] == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "training loss:{:>9.6f}".format(
                        iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            if best:
                save_model(sess, model, args["ckpt_path"], logger, global_steps=step)
            evaluate(sess, model, "test", test_manager, id_to_tag, logger)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = bert_rela_cls_params()
    train(args)