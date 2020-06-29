#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/13 0:34
# @Author : TheTAO
# @Site :
# @File : flaskserver.py
# @Software: PyCharm
import os
import pickle
import json as js
from flask import jsonify
from flask import Flask, render_template
from flask import request
import tensorflow as tf
from model import Model
from collections import OrderedDict
from utils import get_logger, load_config, create_model, save_config, make_path, result_to_json, result_to_show, \
    result_to_str
from data_utils import load_word2vec, input_from_line
import time
import threading
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
flags.DEFINE_boolean("server", True, "if not run server on flask")
# 配置模型有关参数
flags.DEFINE_integer("seg_dim", 20, "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim", 100, "Embedding size for characters")
flags.DEFINE_integer("lstm_dim", 100, "Num of hidden units in LSTM, or num of filters in IDCNN")
# 标注模式
flags.DEFINE_string("tag_schema", "iobes", "tagging schema iobes or iob")
# 梯度裁剪
flags.DEFINE_float("clip", 5, "Gradient clip")
flags.DEFINE_float("dropout", 0.5, "Dropout rate")
flags.DEFINE_float("batch_size", 20, "batch size")
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


# 加载字典
with open(FLAGS.map_file, "rb") as f:
    char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
# 创建相应文件夹
make_path(FLAGS)
# 配置文件读取
if os.path.isfile(FLAGS.config_file):
    config = load_config(FLAGS.config_file)
else:
    config = config_model(char_to_id, tag_to_id)
    save_config(config, FLAGS.config_file)
# 初始化Flask，并且渲染静态页面
app = Flask(__name__, static_url_path="/static")

# 加载配置文件
log_path = os.path.join("log", FLAGS.log_file)
logger = get_logger(log_path)
# 配置GPU
tf_config = tf.ConfigProto()
sess = tf.Session(config=tf_config)
# 加载训练好的模型
model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)


def heartbeat():
    print(time.strftime('%Y-%m-%d %H:%M:%S - heartbeat', time.localtime(time.time())))
    timer = threading.Timer(60, heartbeat)
    timer.start()


timer = threading.Timer(60, heartbeat)
timer.start()

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


# 渲染最初界面
@app.route("/")
def index():
    return render_template("index.html")


# 启动Flask服务器
@app.route('/message', methods=['POST'])
def get_text_input():
    # 获取输入的文本
    req_msg = request.form['msg']
    if req_msg == "":
        res_msg = "请输入需要测试的句子"
    else:
        inputs, tag = model.evaluate_line(sess, input_from_line(req_msg, char_to_id), id_to_tag)
        if FLAGS.server:
            # 如果是命令行测试就打印json数据
            result = result_to_str(inputs, tag)
            return jsonify({'text': result})


if __name__ == "__main__":
    app.config['JSON_AS_ASCII'] = False
    app.run(host='192.168.0.79', port=10112)
