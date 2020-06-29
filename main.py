#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/11 9:14
# @Author  : TheTao
# @Site    : 
# @File    : main.py
# @Software: PyCharm
import os
import time
import pickle
import itertools
import json as js
import numpy as np
import tensorflow as tf
from model import Model
from data_utils import BatchManager, load_word2vec, input_from_line
from collections import OrderedDict
from loader import load_sentences, char_mapping, augment_with_pretrained, tag_mapping, prepare_dataset
from utils import make_path, load_config, save_config, get_logger, print_config, create_model, save_model, clean, \
    test_ner, result_to_json
import warnings

warnings.filterwarnings("ignore")

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

# 主目录路径
root = os.getcwd() + os.sep
# 参数管理
flags = tf.app.flags
flags.DEFINE_boolean("clean", False, "clean train folder")
flags.DEFINE_boolean("train", False, "Whether train the model")
flags.DEFINE_boolean("server", False, "if not run server on flask")
# 配置模型有关参数
flags.DEFINE_integer("seg_dim", 20, "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim", 100, "Embedding size for characters")
flags.DEFINE_integer("lstm_dim", 100, "Num of hidden units in LSTM, or num of filters in IDCNN")
# 标注模式
flags.DEFINE_string("tag_schema", "iobes", "tagging schema iobes or iob")
# 梯度裁剪
flags.DEFINE_float("clip", 5, "Gradient clip")
flags.DEFINE_float("dropout", 0.5, "Dropout rate")
flags.DEFINE_float("batch_size", 10, "batch size")
flags.DEFINE_float("lr", 0.001, "Initial learning rate")
flags.DEFINE_string("optimizer", "adam", "Optimizer for training")
# 是否使用预训练模型
flags.DEFINE_boolean("pre_emb", True, "Wither use pre-trained embedding")
# 替换数字都为0
flags.DEFINE_boolean("zeros", True, "Wither replace digits with zero")
flags.DEFINE_boolean("lower", False, "Wither lower case")
# 训练周期
flags.DEFINE_integer("max_epoch", 200, "maximum training epochs")
# 模型保存有关
flags.DEFINE_integer("steps_check", 5, "steps per checkpoint")
flags.DEFINE_string("ckpt_path", "./check/ckpt", "Path to save model")
flags.DEFINE_string("summary_path", "summary", "Path to store summaries")
# 一些文件路径
flags.DEFINE_string("log_file", "train.log", "File for log")
flags.DEFINE_string("test_log_file", "test.log", "File for log")
flags.DEFINE_string("map_file", "./pickle/maps.pkl", "file for maps")
flags.DEFINE_string("vocab_file", "vocab.json", "File for vocab")
flags.DEFINE_string("config_file", "./config/config_file", "File for config")
flags.DEFINE_string("script", "conlleval", "evaluation script")
flags.DEFINE_string("result_path", "result", "Path for results")
# 数据集有关
flags.DEFINE_string("emb_file", os.path.join(root + "datas/emb_file", "vec.txt"), "Path for pre_trained embedding")
flags.DEFINE_string("train_file", os.path.join(root + "datas/dataset", "example_1.train"), "Path for train data")
flags.DEFINE_string("dev_file", os.path.join(root + "datas/dataset", "example_1.dev"), "Path for dev data")
flags.DEFINE_string("test_file", os.path.join(root + "datas/dataset", "example_1.test"), "Path for test data")
# 模型类型看是使用CNN还是LSTM
flags.DEFINE_string("model_type", "bilstm", "Model type, can be idcnn or bilstm")

# 这样初始化参数管理对象
FLAGS = tf.app.flags.FLAGS
# 抛出一些错误
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# 配置一些参数
def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["model_type"] = FLAGS.model_type
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size
    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config


# 批量评估
def evaluate(sess, model, name, data, id_to_tag, logger):
    # 拿到对应的一个批次测试结果集
    ner_results = model.evaluate(sess, data, id_to_tag)
    # 预测结果保存到结果集
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    # 这里是打印报告结果
    for line in eval_lines:
        logger.info(line)
    # 这里就拿到F1值
    f1 = float(eval_lines[1].strip().split()[-1])
    # 这里返回最佳的F1值
    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1


# 评估单个句子
def predict_line():
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    tf_config = tf.ConfigProto()
    # 读取词典
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        # 根据保存的模型读取模型
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        while True:
            # 反复输入句子进行预测
            line = input("请输入测试句子:")
            inputs, tag = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            if FLAGS.server:
                # 如果是命令行测试就打印json数据
                result = result_to_json(inputs, tag)
                result = js.dumps(result, ensure_ascii=False, indent=4, separators=(',', ': '))
                with open('./result/result.json', 'w', encoding='utf-8') as f:
                    f.write(result)
                print("预测结果为：{}".format(result))


# 训练函数
def train():
    # 加载数据集
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    # 判断是否已经存在有映射好的词典
    if not os.path.isfile(FLAGS.map_file):
        # 如果需要预训练向量
        if FLAGS.pre_emb:
            dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                FLAGS.emb_file,
                list(itertools.chain.from_iterable([[w[0] for w in s] for s in test_sentences])))
        # 如果不用就直接返回
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)
        # 为标签构建词典映射并保存
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences, dev_sentences, test_sentences)
        # 保存词典
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    # 如果有处理好的词典就直接读取
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    # 格式化输入到网络中的标准数据
    train_data = prepare_dataset(train_sentences, char_to_id, tag_to_id, FLAGS.lower)
    dev_data = prepare_dataset(dev_sentences, char_to_id, tag_to_id, FLAGS.lower)
    # 打印统计结果
    print("len{} - len{} sentences in train - dev.".format(len(train_data), len(dev_data)))
    # 初始化batch管理器
    train_manager = BatchManager(train_data, FLAGS.batch_size)
    dev_manager = BatchManager(dev_data, 10)
    # 创建对应的文件夹，确保不出错
    make_path(FLAGS)
    # 读取配置文件
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id)
        save_config(config, FLAGS.config_file)
    # 配置日志
    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)
    # 在训练开始前首先配置GPU
    tf_config = tf.ConfigProto()
    # 获取总共训练周期
    steps_per_epoch = train_manager.len_data
    # 在一个会话中
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        logger.info("start training")
        # 开始迭代训练
        for i in range(FLAGS.max_epoch):
            loss = []
            total_loss = 0
            start = time.time()
            # 获取batch
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                # 将计算好的loss加入
                loss.append(batch_loss)
                total_loss += batch_loss
                iteration = step // steps_per_epoch
                # 5步开始打印
                if (step + 1) % FLAGS.steps_check == 0:
                    logger.info("iteration:{} step:{}/{}, NER loss:{:>9.6f}"
                                .format(iteration + 1, (step % steps_per_epoch) + 1, steps_per_epoch, np.mean(loss)))
            # 每1个epoch打印出验证集的F1值，有点浪费训练时间
            # evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            # 每两个周期保存一次
            if (i + 1) % 2 == 0:
                save_model(sess, model, FLAGS.ckpt_path, logger)
            logger.info('Epoch {} total Loss {:.4f}'.format(i + 1, total_loss / steps_per_epoch))
            logger.info('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# 批量test
def test():
    # 加载配置文件
    config = load_config(FLAGS.config_file)
    # 加载日志管理器
    log_path = os.path.join("log", FLAGS.test_log_file)
    logger = get_logger(log_path)
    # 配置GPU
    tf_config = tf.ConfigProto()
    # 加载数据集
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)
    # 读取词典
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    # 格式化test
    test_data = prepare_dataset(test_sentences, char_to_id, tag_to_id, FLAGS.lower)
    # 加载batch
    test_manager = BatchManager(test_data, 20)
    with tf.Session(config=tf_config) as sess:
        logger.info("start testing...")
        start = time.time()
        # 根据保存的模型读取模型
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        # 获取testbatch
        evaluate(sess, model, "test", test_manager, id_to_tag, logger)
        logger.info("The best_f1 on test_dataset is {}".format(model.best_test_f1.eval()))
        logger.info('Time test for 10 batch is {} sec\n'.format(time.time() - start))


def main():
    if FLAGS.train:
        # 需要进行清洗之后训练
        if FLAGS.clean:
            clean(FLAGS)
        # 或者继续训练
        train()
    elif FLAGS.server:
        # 进行单个预测
        predict_line()
    else:
        test()


if __name__ == '__main__':
    main()
