# encoding = utf8
import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers.python.layers import initializers

from bert import modeling


class Model(object):
    def __init__(self, config):

        self.config = config
        self.lr = config["lr"]
        self.num_tags = config["num_tags"]

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # add placeholders for the model
        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
        self.input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask")
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="segment_ids")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32, name="Dropout")

        used = tf.sign(tf.abs(self.input_ids))
        self.batch_size = tf.shape(self.input_ids)[0]
        self.num_steps = tf.shape(self.input_ids)[-1]

        # embeddings for chinese character and segmentation representation
        # bert [batch_size, seq_length, 768]
        embedding = self.bert_embedding()

        # apply dropout before feed tp lstm layer
        fc_inputs = tf.nn.dropout(embedding, self.dropout)

        # logits for tags
        self.logits, self.pred = self.fc_layer(fc_inputs)

        # loss
        self.loss = self.losses_layer(self.logits)

        # bert模型参数初始化的地方
        init_checkpoint = self.config["init_checkpoint"]
        # 获取模型中所有的训练参数。
        tvars = tf.trainable_variables()
        # 加载BERT模型
        (assignment_map, initialized_variable_names) = \
            modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        print("**** Trainable Variables ****")
        # 打印加载模型的参数
        train_vars = []
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            else:
                train_vars.append(var)
            # print("  name = %s, shape = %s%s", var.name, var.shape,
            #       init_string)
        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            else:
                raise KeyError

            grads = tf.gradients(self.loss, train_vars)
            (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

            self.train_op = self.opt.apply_gradients(
                zip(grads, train_vars), global_step=self.global_step)
            #capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
            #                     for g, v in grads_vars if g is not None]
            #self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step, )

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def bert_embedding(self):
        # load bert embedding
        bert_config_file = self.config["bert_config"]
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)  # 配置文件地址。
        model = modeling.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)
        embedding = model.get_pooled_output()
        return embedding

    def fc_layer(self, fc_input):
        """
           fc_input: [batch_size, seq_length, hidden_units], bert model output
        """
        with tf.variable_scope("logits"):
            input_dim = fc_input.get_shape().as_list()[-1]
            W = tf.get_variable("W", shape=[input_dim, self.num_tags],
                                dtype=tf.float32, initializer=self.initializer)
            b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32, initializer=tf.zeros_initializer())
            logits = tf.nn.xw_plus_b(fc_input, W, b)
            probabilities = tf.nn.softmax(logits, axis=-1)
            predict = tf.argmax(probabilities, axis=1, output_type=tf.int32)
            # pred = tf.reshape(pred, [-1, self.num_steps, self.num_tags])
            return logits, predict

    def losses_layer(self, logits):

        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(self.targets, depth=self.num_tags, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return loss

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        _, segment_ids, chars, mask, tags = batch
        feed_dict = {
            self.input_ids: np.asarray(chars),
            self.input_mask: np.asarray(mask),
            self.segment_ids: np.asarray(segment_ids),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _, = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            logits, pred = sess.run([self.logits, self.pred], feed_dict)
            return logits, pred

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model 
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        rela_labels = []
        pred_labels = []
        for batch in data_manager.iter_batch(shuffle=True):
            strings = batch[0]
            labels = [item[0] for item in batch[-1]]
            rela_labels.extend(labels)
            logits, pred = self.run_step(sess, False, batch)
            pred_ = pred.tolist()
            pred_labels.extend(pred_)
            for i in range(len(strings)):
                result = []
                string = ''.join(strings[i])
                gold = id_to_tag[int(labels[i])]
                pred_label = id_to_tag[int(pred_[i])]
                result.append(" ".join([string, gold, pred_label]))
                results.append(result)
        return results, rela_labels, pred_labels

    def evaluate_line(self, sess, inputs, id_to_tag):
        logits, pred = self.run_step(sess, False, inputs)
        tags = [id_to_tag[pred[0]]]
        return tags