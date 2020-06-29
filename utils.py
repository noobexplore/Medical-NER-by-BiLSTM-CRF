#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/11 15:07
# @Author  : TheTao
# @Site    : 
# @File    : utils.py
# @Software: PyCharm
import os
import json
import logging
import shutil
import tensorflow as tf
from conlleval import return_report
from data_process import entities_dict_chinese


def make_path(params):
    """
    构建对应的文件夹函数
    Make folders for training and evaluation
    """
    if not os.path.isdir(params.result_path):
        os.makedirs(params.result_path)
    if not os.path.isdir(params.ckpt_path):
        os.makedirs(params.ckpt_path)
    if not os.path.isdir("log"):
        os.makedirs("log")


# 存储配置文件
def save_config(config, config_file):
    """

    Save configuration of the model
    parameters are stored in json format
    """
    with open(config_file, "w", encoding="utf8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


# 读取配置文件
def load_config(config_file):
    """
    Load configuration of the model
    parameters are stored in json format
    """
    with open(config_file, encoding="utf8") as f:
        return json.load(f)


def print_config(config, logger):
    """
    Print configuration of the model
    """
    for k, v in config.items():
        logger.info("{}:\t{}".format(k.ljust(15), v))


# 获取日志文件
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


# 创建模型
def create_model(session, Model_class, path, load_vec, config, id_to_char, logger):
    model = Model_class(config)
    # 加载模型
    ckpt = tf.train.get_checkpoint_state(path)
    # 看是否存在训练好的模型
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        # 如果存在就进行重新加载
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created model with fresh parameters.")
        # 此步骤非常重要，不初始化的话就无法读取到权重
        session.run(tf.global_variables_initializer())
        # 读取预训练模型
        if config["pre_emb"]:
            # 先取得随机初始化的权重
            emb_weights = session.run(model.char_lookup.read_value())
            # 然后再加载预训练好的词向量
            emb_weights = load_vec(config["emb_file"], id_to_char, config["char_dim"], emb_weights)
            # 进行分配，后面训练的时候还是会修改
            session.run(model.char_lookup.assign(emb_weights))
            logger.info("Load pre-trained embedding.")
    return model


# 保存模型
def save_model(sess, model, path, logger):
    checkpoint_path = os.path.join(path, "ner.ckpt")
    model.saver.save(sess, checkpoint_path)
    logger.info("model saved")


# 清除相关文件以及缓存函数
def clean(params):
    """
    重新训练前清除函数
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


# 测试结果写入文件
def test_ner(results, path):
    """
    Run perl script to evaluate model
    """
    output_file = os.path.join(path, "ner_predict.utf8")
    with open(output_file, "w", encoding='utf8') as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + "\n")
            to_write.append("\n")
        f.writelines(to_write)
    # 返回评估报告
    eval_lines = return_report(output_file)
    return eval_lines


# 将结果写成JSON文件
def result_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx + 1, "type": entities_dict_chinese[tag[2:]]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": entities_dict_chinese[tag[2:]]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item


# 将结果正常展示
def result_to_show(string, tags):
    # 显示列表
    show_list = []
    entity_name = ''
    # 还是要循环去拼
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            show_list.append([char, idx, idx + 1, entities_dict_chinese[tag[2:]]])
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            show_list.append([entity_name, entity_start, idx + 1, entities_dict_chinese[tag[2:]]])
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return show_list


def result_to_str(string, tags):
    show_str = ''
    entity_name = ''
    # 还是要循环去拼
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            show_str = show_str + '实体：' + char + " " + '实体类别：' + entities_dict_chinese[tag[2:]] + '<br>'
        elif tag[0] == "B":
            entity_name += char
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            show_str = show_str + '实体：' + entity_name + " " + '实体类别：' + entities_dict_chinese[tag[2:]] + '<br>'
            entity_name = ""
        else:
            entity_name = ""
        idx += 1
    return show_str
