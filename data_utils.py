#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 17:14
# @Author  : TheTao
# @Site    : 
# @File    : data_utils.py
# @Software: PyCharm
import re
import math
import codecs
import random
import jieba
import numpy as np

# 结巴初始化
jieba.initialize()


# 数字正则
def zero_digits(s):
    """
    替换掉字符串中间的数字为全部为0
    """
    return re.sub(r'\d', '0', s)


def create_dico(item_list):
    """
    构建一个频率字典，频率就相当于id
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


# 映射函数
def create_mapping(dico):
    """
    创造一个词典映射
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


# 获取到分词特征
def get_seg_features(string):
    """
    这里给我的感觉就是加入附加特征信息
    :param string:
    :return:
    """
    seg_feature = []
    # 遍历分词的结果列表
    for word in jieba.cut(string):
        # 如果分词为一个词
        if len(word) == 1:
            seg_feature.append(0)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)
    return seg_feature


# 填充函数
def pad_data(data):
    strings = []
    chars = []
    segs = []
    targets = []
    max_length = max([len(sentence[0]) for sentence in data])
    for line in data:
        string, char, seg, target = line
        # 这里注意此处的0对应的为pad特殊字符的id
        padding = [0] * (max_length - len(string))
        strings.append(string + padding)
        chars.append(char + padding)
        segs.append(seg + padding)
        targets.append(target + padding)
    return [strings, chars, segs, targets]


# 创建一个batch类
class BatchManager(object):
    def __init__(self, data, batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        # 总共batch
        num_batch = int(math.ceil(len(data) / batch_size))
        # 安装词的长度升序排序
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        # 初始化
        batch_data = list()
        # 一个batch一个batch去填充
        for i in range(num_batch):
            batch_data.append(pad_data(sorted_data[i * int(batch_size): (i + 1) * int(batch_size)]))
        return batch_data

    # 打乱生成迭代器
    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


# 读取预训练的词向量将其替换成新的
def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """
    加载预训练词向量
    :param emb_path:
    :param id_to_word:
    :param word_dim:
    :param old_weights:
    :return:
    """
    # 获取随机向量
    new_weights = old_weights
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    # 无效词的统计
    emb_invalid = 0
    # 先遍历将值转化为浮点数
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array([float(x) for x in line[1:]]).astype(np.float32)
        else:
            # 统计无效值
            emb_invalid += 1
    if emb_invalid > 0:
        # 打印无效向量
        print('WARNING: %i invalid lines' % emb_invalid)
    # 开始替换对应词典中存在的词对应的向量
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    for i in range(n_words):
        # 从词典中取出词去找
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            # 寻找小写词，估计对应英文字母
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub(r'\d', '0', word.lower()) in pre_trained:
            # 寻找数字，对应数值且需要全部转化为0
            new_weights[i] = pre_trained[re.sub(r'\d', '0', word.lower())]
            c_zeros += 1
    print('Loaded %i pretrained embeddings.' % len(pre_trained))
    # 打印统计信息
    print('%i / %i (%.4f%%) words have been initialized with pretrained embeddings.' % (
        c_found + c_lower + c_zeros, n_words, 100. * (c_found + c_lower + c_zeros) / n_words))
    print('%i found directly, %i after lowercasing, %i after lowercasing + zero.' % (c_found, c_lower, c_zeros))
    # 返回新参数
    return new_weights


# 对单个句子的处理，结果返回为模型能接受的输入
def input_from_line(line, char_to_id):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    line = full_to_half(line)
    line = replace_html(line)
    inputs = list()
    inputs.append([line])
    line.replace(" ", "$")
    inputs.append([[char_to_id[char] if char in char_to_id else char_to_id["<UNK>"] for char in line]])
    inputs.append([get_seg_features(line)])
    inputs.append([[]])
    return inputs


def full_to_half(s):
    """
    Convert full-width character to half-width one
    """
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)


def replace_html(s):
    s = s.replace('&quot;', '"')
    s = s.replace('&amp;', '&')
    s = s.replace('&lt;', '<')
    s = s.replace('&gt;', '>')
    s = s.replace('&nbsp;', ' ')
    s = s.replace("&ldquo;", "")
    s = s.replace("&rdquo;", "")
    s = s.replace("&mdash;", "")
    s = s.replace("\xa0", " ")
    return (s)


def create_input(data):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    inputs = list()
    inputs.append(data['chars'])
    inputs.append(data["segs"])
    inputs.append(data['tags'])
    return inputs


if __name__ == '__main__':
    s = '黑龙江在中国是我家'
    sf = get_seg_features(s)
    print(sf)
