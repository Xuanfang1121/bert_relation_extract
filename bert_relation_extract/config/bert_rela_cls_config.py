# -*- coding: utf-8 -*-
# @Time    : 2021/1/17 14:32
# @Author  : zxf
import os


def bert_rela_cls_params():
    config = dict()
    root_path = "D:/Spyder/plantdata/kgtext_ml_relation/output/output_bert/"
    base_bert_path = "D:/Spyder/pretrain_model/chinese_L-12_H-768_A-12/"

    config["num_tags"] = 8
    config["batch_size"] = 32
    config['max_seq_len'] = 64
    config["seg_dim"] = 20
    config["char_dim"] = 100

    config["clip"] = 5
    config["dropout_keep"] = 0.7
    config["optimizer"] = "adam"
    config["lr"] = 0.01
    config["zeros"] = False
    config["lower"] = True
    config["max_epoch"] = 1
    config["steps_check"] = 100
    config["init_checkpoint"] = os.path.join(base_bert_path, "bert_model.ckpt")
    config["bert_config"] = os.path.join(base_bert_path, "bert_config.json")
    config["bert_vocab"] = os.path.join(base_bert_path, "vocab.txt")

    config["ckpt_path"] = os.path.join(root_path, "ckpt")
    config["pb_path"] = os.path.join(root_path, "pb_model/")
    config["summary_path"] = os.path.join(root_path, "summary")
    config["map_file"] = os.path.join(root_path, "maps.pkl")
    config["vocab_file"] = os.path.join(root_path, "vocab.json")
    config["config_file"] = os.path.join(root_path, "config_file")
    config["result_path"] = os.path.join(root_path, "result/")
    config["log_path"] = os.path.join(root_path, "log/")
    config["log_file"] = "log.log"
    config["train_file"] = os.path.join(root_path, "pdRelation_train.data")
    config["dev_file"] = os.path.join(root_path, "pdRelation_val.data")
    config["test_file"] = os.path.join(root_path, "pdRelation_val.data")
    return config