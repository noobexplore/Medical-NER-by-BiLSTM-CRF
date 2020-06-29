#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 17:14
# @Author  : TheTao
# @Site    : 
# @File    : loader.py
# @Software: PyCharm
import os
import re
import codecs
from data_utils import zero_digits, create_mapping, create_dico, get_seg_features


def load_sentences(path, lower, zero):
    """
    加载数据集，一行至少包含一个词和对应的标签
    :param path:
    :param lower:
    :param zero:
    :return:
    """
    sentences = []
    sentence = []
    # 每一行去读
    for line in codecs.open(path, 'r', encoding='utf-8'):
        # 此处一定要将各种数字转化为0，这样就能泛化识别
        line = zero_digits(line.rstrip()) if zero else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
                # word[0] = " "
            else:
                word = line.split()
            if len(word) == 2:
                sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def char_mapping(sentences, lower):
    """
    构建一个词典和每个词的映射，通过频率排序
    :param sentences:
    :param lower:
    :return:
    """
    # 将每个词转化为小写
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(chars)
    # 定义特殊字符
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    # 返回正反向词典
    char_to_id, id_to_char = create_mapping(dico)
    return dico, char_to_id, id_to_char


def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    用预先训练好的词向量来扩充词典
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    assert os.path.isfile(ext_emb_path)
    # 加载已经预训练好的词向量
    pretrained = set([line.rstrip().split()[0].strip()
                      for line in codecs.open(ext_emb_path, 'r', 'utf-8') if len(ext_emb_path) > 0])
    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    # 这里应该是在判断词在字典中没有，如果没有就分配为0
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            if any(x in pretrained for x in [char, char.lower(), re.sub(r'\d', '0', char.lower())]) \
                    and char not in dictionary:
                dictionary[char] = 0
    # 重新生成词典映射
    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


# 标签映射函数
def tag_mapping(train_senc, dev_senc, test_senc):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tag_to_id_file = open('./dictdata/tag_to_id.txt', 'w', encoding='utf8')
    id_to_tag_file = open('./dictdata/id_to_tag.txt', 'w', encoding='utf8')
    tags = []
    # 迭代分别或训练集测试集以及验证集的标签
    for s in train_senc:
        ts = []
        for char in s:
            tag = char[-1]
            ts.append(tag)
        tags.append(ts)
    for s in dev_senc:
        ts = []
        for char in s:
            tag = char[-1]
            ts.append(tag)
        tags.append(ts)
    for s in test_senc:
        ts = []
        for char in s:
            tag = char[-1]
            ts.append(tag)
        tags.append(ts)
    # 然后构建频率映射
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    # 保存为文件
    for k, v in tag_to_id.items():
        tag_to_id_file.write(k + ":" + str(v) + "\n")
    for k, v in id_to_tag.items():
        id_to_tag_file.write(str(k) + ":" + str(v) + "\n")
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    """
    格式化输入到网络中数据
    :param sentences:
    :param char_to_id:
    :param tag_to_id:
    :param lower:
    :param train:
    :return:
    """
    none_index = tag_to_id["O"]

    # 英文小写转化函数
    def to_lower(x):
        return x.lower() if lower else x

    data = []
    for s in sentences:
        string = [w[0] for w in s]
        # 这里根据vocab转化ID，没有就用unk填充
        chars = [char_to_id[to_lower(w) if to_lower(w) in char_to_id else '<UNK>'] for w in string]
        # 这个位置提取分词特征，这传入的是之前一个一个词连接好的一句话
        segs = get_seg_features("".join(string))
        # 如果是训练就需要将标签转化
        if train:
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            tags = [none_index for _ in chars]
        # 这里添加这4个输入
        data.append([string, chars, segs, tags])
    return data

