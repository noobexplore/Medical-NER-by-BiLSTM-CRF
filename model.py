#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/11 16:02
# @Author  : TheTao
# @Site    :
# @File    : model.py
# @Software: PyCharm
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers


class Model(object):
    def __init__(self, config):
        # 初始化配置文件
        self.config = config
        # 取出配置文件来初始化参数
        self.lr = config["lr"]
        # 字向量大小
        self.char_dim = config["char_dim"]
        # LSTM单元维度
        self.lstm_dim = config["lstm_dim"]
        # 分词维度
        self.seg_dim = config["seg_dim"]
        # 标签个数
        self.num_tags = config["num_tags"]
        # 字的长度
        self.num_chars = config["num_chars"]
        # 4个不同的特征
        self.num_segs = 4
        # 设置全局变量
        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        # 初始化器
        self.initializer = initializers.xavier_initializer()
        # 定义placeholder，[None, None]第一个维度是None代表一个batch大小第二个代表词的维度
        self.char_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="ChatInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="SegInputs")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Targets")
        # dropout
        self.dropout = tf.placeholder(dtype=tf.float32, name="Dropout")
        # 这里计算一句话中除了填充0之外词的个数，在计算的时候就只是计算对应的词
        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]
        # 获取model类型有bi_lstm和idcnn
        self.model_type = config['model_type']
        # idCNN参数设置
        self.layers = [{'dilation': 1}, {'dilation': 2}, {'dilation': 3}]
        # 过滤器宽度
        self.filter_width = 3
        # 过滤器数量
        self.num_filter = 100
        # 重复次数
        self.repeat_times = 4
        # 总共的输入维度
        self.embedding_dim = self.char_dim + self.seg_dim
        # 下面为第一层嵌入层
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)
        # 首先判断为那个模型
        if self.model_type == 'bilstm':
            # 加入dropout层
            model_inputs = tf.nn.dropout(embedding, self.dropout)
            # 初始化LSTM层
            model_outputs = self.biLSTM_layer(model_inputs, self.lstm_dim, self.lengths)
            # 得到logits得分
            self.logits = self.project_layer_bilstm(model_outputs)
        elif self.model_type == 'idcnn':
            # 加入dropout层
            model_inputs = tf.nn.dropout(embedding, self.dropout)
            # IDCNN层
            model_outputs = self.IDCNN_layer(model_inputs)
            # 映射层
            self.logits = self.project_layer_idcnn(model_outputs)
        # 根据映射层的输入计算得分
        self.loss = self.loss_layer(self.logits, self.lengths)
        # 开始定义优化器
        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError
        # 然后做一个梯度截断操作，防止梯度爆炸
        grads_vars = self.opt.compute_gradients(self.loss)
        # 根据截断比例去进行截断操作
        capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v] for g, v in grads_vars]
        # 然后再更新
        self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)
        # 最后保存模型
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    # 定义嵌入层，其实这里面可以映射任意特征
    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        """
        :param self:
        :param char_inputs:
        :param seg_inputs:
        :param config:
        :param name:
        :return:
        """
        embedding = []
        with tf.variable_scope("char_embedding" if not name else name):
            # 100维
            self.char_lookup = tf.get_variable(name="char_embedding", shape=[self.num_chars, self.char_dim],
                                               initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            # 如果需要有分词嵌入
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"):
                    self.seg_lookup = tf.get_variable(name="seg_embedding", shape=[self.num_segs, self.seg_dim],
                                                      initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            # 连接词向量与分词向量
            embed = tf.concat(embedding, axis=-1)
        return embed

    # BiLSTM层的定义
    def biLSTM_layer(self, model_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            # 双向分前向和后向
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(lstm_dim,
                                                                                         use_peepholes=True,
                                                                                         initializer=self.initializer,
                                                                                         state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_cell["forward"],
                                                                    lstm_cell["backward"],
                                                                    model_inputs,
                                                                    dtype=tf.float32,
                                                                    sequence_length=lengths)
        # 返回相连接
        return tf.concat(outputs, axis=2)

    # IDCNN层的定义
    def IDCNN_layer(self, model_inputs, name=None):
        """
        :param model_inputs:[batch_size, num_steps, emb_size]
        :return:[batch_size, num_steps, cnn_output_width]
        """
        # 先扩维 [batch, 1, length, dim]
        model_inputs = tf.expand_dims(model_inputs, 1)
        # 是否重用
        reuse = False
        if self.dropout == 1.0:
            reuse = True
        # 先构建图
        with tf.variable_scope("idcnn" if not name else name):
            # 卷积参数
            filter_Weights = tf.get_variable("idcnn_filter",
                                             shape=[1, self.filter_width, self.embedding_dim, self.num_filter],
                                             initializer=self.initializer)
            # 输入层是一个4维向量shape of input = [batch, in_height, in_width, in_channels]
            # 滤波器的shape of filter = [filter_height, filter_width, in_channels, out_channels=100]
            layerInput = tf.nn.conv2d(model_inputs, filter_Weights, strides=[1, 1, 1, 1], padding="SAME",
                                      name="init_layer", use_cudnn_on_gpu=True)
            # 最终输出列表
            finalOutFromLayers = []
            # 最终维度
            totalWidthForLastDim = 0
            # 分层级进行高阶卷积
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    # 获取打孔的参数
                    dilation = self.layers[i]['dilation']
                    # 判断是否是最后一层
                    isLast = True if i == (len(self.layers) - 1) else False
                    # 循环定义图
                    with tf.variable_scope("atrous-conv-layer-%d" % i, reuse=True if (reuse or j > 0) else False):
                        w = tf.get_variable("filterW", shape=[1, self.filter_width, self.num_filter, self.num_filter],
                                            initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("filterB", shape=[self.num_filter])
                        # 开始打孔层
                        conv = tf.nn.atrous_conv2d(layerInput, w, rate=dilation, padding="SAME")
                        conv = tf.nn.bias_add(conv, b)
                        # 激活层
                        conv = tf.nn.relu(conv)
                        # 判断是否是最后一层
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                        layerInput = conv
            # 最后进行拼接
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)  # 4个100拼接为400
            # 设置keep_prob
            keepProb = 1.0 if reuse else 0.5
            # 设置dropout
            finalOut = tf.nn.dropout(finalOut, keepProb)
            # 降维重塑为最终输出
            finalOut = tf.squeeze(finalOut, [1])  # [batch,length,400]
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])
            self.cnn_output_width = totalWidthForLastDim
            return finalOut  # [batch*length,400]

    # idcnn映射层
    def project_layer_idcnn(self, idcnn_outputs, name=None):  # [batch*length,400]
        """
        :param idcnn_outputs:[batch*length,400]
        :return:[batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            # 全连接映射层
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.cnn_output_width, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)
                b = tf.get_variable("b", initializer=tf.constant(0.001, shape=[self.num_tags]))
                # 预测
                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)
            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])  # [batch,length,58]

    # tf1.x层定义好了之后需要加入映射层来更新参数
    def project_layer_bilstm(self, lstm_outputs, name=None):
        """
        输入为LSTM的输出
        :param lstm_outputs:[batch_size, num_steps, emb_size]
        :param name:
        :return:shape[None, num_step, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                # LSTM中的参数
                W = tf.get_variable("W", shape=[self.lstm_dim * 2, self.lstm_dim], dtype=tf.float32,
                                    initializer=self.initializer)
                # 偏置项
                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32, initializer=tf.zeros_initializer())
                # 由于是双向LSTM
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim * 2])
                # 激活函数
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags], dtype=tf.float32,
                                    initializer=self.initializer)
                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                # 此处计算得分的时候不需要激活函数
                pred = tf.nn.xw_plus_b(hidden, W, b)
            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    # 此处定义loss函数
    def loss_layer(self, project_logits, lengths, name=None):
        """
        计算crfloss
        :param project_logits:[1, num_steps, num_tags]
        :param lengths:
        :param name:
        :return:
        """
        with tf.variable_scope("crf_loss" if not name else name):
            # 首先添加Start词除开start在59位为0其他全部为-1000
            small = -1000.0
            # 在三维tensor中去填补
            start_logits = tf.concat([small * tf.ones(shape=[self.batch_size, 1, self.num_tags]),
                                      tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            # 标签也需要填充
            targets = tf.concat([tf.cast(self.num_tags * tf.ones([self.batch_size, 1]), tf.int32),
                                 self.targets], axis=-1)
            # 定义crf中转移矩阵，即为CRF中参数矩阵
            self.trans = tf.get_variable("transitions", shape=[self.num_tags + 1, self.num_tags + 1],
                                         initializer=self.initializer)
            # 传入进行计算，其计算的为真实路径得分的目标函数
            log_likelihood, self.trans = crf_log_likelihood(inputs=logits,
                                                            tag_indices=targets,
                                                            transition_params=self.trans,
                                                            sequence_lengths=lengths + 1)
            # 最后返回平均值
            return tf.reduce_mean(-log_likelihood)

    # 投喂操作
    def create_feed_dict(self, is_train, batch):
        """
        将投喂到网络中的操作同一封装
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        _, chars, segs, tags = batch
        feed_dict = {self.char_inputs: np.asarray(chars),
                     self.seg_inputs: np.asarray(segs),
                     self.dropout: 1.0}
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    # 运行步骤
    def run_step(self, sess, is_train, batch):
        """

        :param sess:传入会话
        :param is_train:是否是训练过程
        :param batch:一个batch大小
        :return:
        """
        # 初始化投喂字典
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run([self.global_step, self.loss, self.train_op], feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    # 解码
    def decode(self, logits, lengths, matrix):
        """
        单步解码
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        paths = []
        small = -1000.0
        # 同样需要进行填充start
        start = np.asarray([[small] * self.num_tags + [0]])
        # 循环单步去解码
        for score, length in zip(logits, lengths):
            # 这里根据真实长度去获取的得分
            score = score[:length]
            # 填充开始标志
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            # 使用viterbi_decode对最大序列长度进行解码
            path, _ = viterbi_decode(logits, matrix)
            # 不要start标志
            paths.append(path[1:])
        return paths

    # 评估函数批量评估
    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess:
        :param data_manager:
        :param id_to_tag:
        :return:
        """
        results = []
        # 拿到转移矩阵
        trans = self.trans.eval()
        # 拿到batch
        for batch in data_manager.iter_batch():
            # 拿到测试句子
            strings = batch[0]
            # 拿到真实的标签
            tags = batch[-1]
            # 单步的去运行，得到句子真实长度和预测分值
            lengths, scores = self.run_step(sess, False, batch)
            # 再去解码得到最佳路径
            batch_paths = self.decode(scores, lengths, trans)
            # 每个字每个字的去组装为[词，标签，预测标签]
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                # 真实标签和预测标签
                gold = [id_to_tag[int(x)] for x in tags[i][:lengths[i]]]
                pred = [id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]]
                # 循环去加入
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        # 返回最终一个batch的结果集为后面做准备
        return results

    # 预测单个句子
    def evaluate_line(self, sess, inputs, id_to_tag):
        # 只要是测试必须拿到转移矩阵
        trans = self.trans.eval(session=sess)
        # 单步运行
        lengths, scores = self.run_step(sess, False, inputs)
        # 解码最佳路径
        batch_paths = self.decode(scores, lengths, trans)
        # 单步只是这里不同，直接依次拿出对应词的预测的标签
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        # 返回对应的结果
        return inputs[0][0], tags
