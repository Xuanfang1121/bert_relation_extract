# -*- coding: utf-8 -*-
# @Time    : 2021/1/16 19:10
# @Author  : zxf
import os
import logging

import tensorflow as tf

from model import Model
from utils import get_logger
from utils import load_config
from utils import create_model
from config.bert_rela_cls_config import bert_rela_cls_params


def export_bert_pb_fun(args):

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    if not os.path.exists(args["pb_path"]):
        os.makedirs(args["pb_path"])
    logging.info("pb_path: {}".format(args["pb_path"]))
    files = os.listdir(args["pb_path"])
    if len(files) == 0:
        max_version = 1
        export_path = os.path.join(args["pb_path"], str(max_version))
    else:
        files = list(map(int, files))
        max_version = max(files) + 1
        export_path = os.path.join(args["pb_path"], str(max_version))
    log_path = os.path.join(args["log_path"], args["log_file"])
    logger = get_logger(log_path)
    graph = tf.Graph()
    with graph.as_default():
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_path)
        with tf.Session(config=tf_config) as sess:
            model = create_model(sess, Model, args["ckpt_path"], args, logger)
            input_ids = model.input_ids
            input_mask = model.input_mask
            segment_ids = model.segment_ids
            dropout = model.dropout
            logits = model.logits
            saver = tf.train.Saver()

        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            ckpt_file = tf.train.latest_checkpoint(args["ckpt_path"])
            saver.restore(sess, ckpt_file)

            model_tensor_input_ids = tf.compat.v1.saved_model.utils.build_tensor_info(input_ids)
            model_tensor_input_dropout = tf.compat.v1.saved_model.utils.build_tensor_info(dropout)
            model_tensor_input_mask = tf.compat.v1.saved_model.utils.build_tensor_info(input_mask)
            model_tensor_segment_ids = tf.compat.v1.saved_model.utils.build_tensor_info(segment_ids)
            model_tensor_output = tf.compat.v1.saved_model.utils.build_tensor_info(logits)

            prediction_signature = (
                tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                    inputs={'input_ids': model_tensor_input_ids,
                            'input_mask': model_tensor_input_mask,
                            'segment_ids': model_tensor_segment_ids,
                            "Dropout": model_tensor_input_dropout},
                    outputs={'predictions': model_tensor_output},
                    method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME
                )
            )
            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

            builder.add_meta_graph_and_variables(
                sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict':
                        prediction_signature,
                    tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        prediction_signature,
                },
                legacy_init_op=legacy_init_op)

            builder.save(as_text=False)
            logging.info('pb done exporting!')


if __name__ == "__main__":
    args = bert_rela_cls_params()
    export_bert_pb_fun(args)