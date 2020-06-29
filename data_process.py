#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 10:20
# @Author  : TheTao
# @Site    : 
# @File    : data_process.py
# @Software: PyCharm
import os
import codecs

# 实体字典
entities_dict = {
    'Level': 'LEV',
    'Test_Value': 'TSV',
    'Test': 'TES',
    'Anatomy': 'ANT',
    'Amount': 'AMO',
    'Disease': 'DIS',
    'Drug': 'DRU',
    'Treatment': 'TRE',
    'Reason': 'REA',
    'Method': 'MET',
    'Duration': 'DUR',
    'Operation': 'OPE',
    'Frequency': 'FRE',
    'Symptom': 'SYM',
    'SideEff': 'SID'
}
entities_dict_chinese = {
    'LEV': '等级-Level',
    'TSV': '检测值-Test_Value',
    'TES': '测试类-Test',
    'ANT': '解剖类-Anatomy',
    'AMO': '程度-Amount',
    'DIS': '疾病类-Disease',
    'DRU': '药物类-Drug',
    'TRE': '治疗方法-Treatment',
    'REA': '原因-Reason',
    'MET': '方法类-Method',
    'DUR': '持续时间-Duration',
    'OPE': '手术类-Operation',
    'FRE': '频率-Frequency',
    'SYM': '症状类-Symptom',
    'SID': '副作用-SideEff'
}
# 停用词表
stopwords = ['  ', '\n', ' ']

# 一些路径
train_dir = './datas/ruijin_round1_train2_20181022'
test_dir = './datas/ruijin_round1_test_a_20181022'
train_file = './datas/dataset/example_1.train'
test_file = './datas/dataset/example_1.test'
dev_file = './datas/dataset/example_1.dev'


# 一些函数
def readfile(filepath):
    """
    读取每一篇文章，返回文章中每个字构成的列表
    :param filepath:txt文件路径
    :return:文章构成的字级别的列表
    """
    sentence = []
    for line in codecs.open(filepath, 'r', encoding='utf-8'):
        sentence.extend([line[i] for i in range(len(line))])
    return sentence


# 获取实体的类别
def getEntities(dir):
    """
    获取train中的实体类别
    :param dir:传入训练集数据路径
    :return:去重实体类别
    """
    entities = []
    # 列出当前目录下所有文件与目录
    files = os.listdir(dir)
    # 每个目录和文件遍历
    for i in range(len(files)):
        # 将文件名以.分开取后面的文件名
        if files[i].split(".")[1] == 'ann':
            # 合并路径
            path = os.path.join(dir, files[i])
            # 然后再打开对应的文件
            for line in codecs.open(path, 'r', 'utf-8'):
                # 以\t分开
                parts = line.split("\t")
                # 取中间的第一个即为类别
                entities.append(parts[1].split(" ")[0])
    print('total of entities is {}'.format(len(entities)))
    return list(set(entities))


# 获取标注信息
def getNotations(filepath):
    """
    从ann文件中获取文件路径
    :param filepath:ann文件路径
    :return:标注信息对列表[[实体类别，起始位置，结束位置]]
    """
    pairs = []
    for line in codecs.open(filepath, 'r', 'utf-8'):
        # 首先先将每行以\t分开，然后取中间部分，最后再以空格分开
        parts = line.split('\t')[1].split(' ')
        # 然后对应的去取之后形成实体对
        pair = []
        pair.append(parts[0])
        pair.append(int(parts[1]))
        # 这里主要是会有类似于如下的情况：T62	Symptom 1487 1488;1489 1491	低 血糖，所以直接就取最后一个就为结束字符
        pair.append(int(parts[-1]))
        pairs.append(pair)
    return pairs


# 开始打标注
def getPairs(senc_file, label_file):
    """
    该函数读取两个文件的内容，一个是文档文件，一个是标注文件。
    目的是遍历两个文件去对应的打标注。
    :param senc_file:原文本文档
    :param label_file:标签文档
    :return:
    """
    # 读取原文文档转化为字列表
    sentence = readfile(senc_file)
    # 先将上面的字列表扩展为[w, O]，提前先打好标签
    sentence = [[w, 'O'] for w in sentence]
    # 再读取标签文件获取到对应标签位置列表[[实体类别，起始位置，结束位置]]
    labels = getNotations(label_file)

    # 替换字符串第一个字符
    def replace_l(index, char):
        s = list(sentence[index][1])
        s[0] = char
        sentence[index][1] = "".join(s)

    # 处理单个字符标注的情况，其实究其原因：由于标注人员标注失误造成重复和嵌套标注，所以需要处理
    def check_single(index, lab):
        # 如果在最开始或最后，那么直接标为S
        if index < 2 or index > len(sentence) - 2:
            sentence[index][1] = 'S-' + lab
        # 如果不是处于开始或者结尾
        if sentence[index][1] != 'O':
            # 这里打好标签之后还需要考虑前面的情况
            sentence[index][1] = 'S-' + lab
            # 这里是往前看两个位置如果都有标记，那说明需要给S前面加一个结尾E
            if sentence[index - 1][1] != 'O' and sentence[index - 2][1] != 'O':
                replace_l(index - 1, 'E')
            # 否则如果只是一个O那么就直接加为S
            elif sentence[index - 1][1] != 'O':
                replace_l(index - 1, 'S')
            # 这里看完前面再看后面两个位置一样的道理
            if sentence[index + 1][1] != 'O' and sentence[index + 2][1] != 'O':
                replace_l(index + 1, 'B')
            elif sentence[index + 1][1] != 'O':
                replace_l(index + 1, 'S')
        else:
            # 如果还未标注过，则就直接打上标注
            sentence[index][1] = 'S-' + lab

    # 处理正常多个字符标记的时候，也需要像上面处理一些情况，只不过处理之后，打标记的过程在下面一起处理
    def check(start, end, lab):
        if sentence[start][1] != 'O' and start > 2:
            if sentence[start - 1][1] != 'O' and sentence[start - 2][1] != 'O':
                replace_l(start - 1, 'E')
            elif sentence[start - 1][1] != 'O':
                replace_l(start - 1, 'S')
        if sentence[end][1] != 'O' and end < len(sentence) - 2:
            if sentence[end + 1][1] != 'O' and sentence[end + 2][1] != 'O':
                replace_l(end + 1, 'B')
            elif sentence[end + 1][1] != 'O':
                replace_l(end + 1, 'S')

    # 开始迭代打标记
    for label in labels:
        # 首先取出对应的类别
        lab = entities_dict[label[0]]
        # 获取起始或结束的位置
        start = label[1]
        end = label[2]
        # 开始判断是否是单个字的标注
        if end - start == 1:
            check_single(start, lab)
            continue
        # 处理不是单个字的情况
        check(start, end - 1, lab)
        # 开始标注
        sentence[start][1] = 'B-' + lab
        sentence[end - 1][1] = 'E-' + lab
        # 标注中间字
        for i in range(start + 1, end - 1):
            sentence[i][1] = 'I-' + lab
    # 去掉停用词
    sentence = [w for w in sentence if w[0] not in stopwords]
    return sentence


# 文件写入函数
def writefile(filename, list, sep=" "):
    """
    该函数负责将处理好的标注数据文件保存
    一个字的一个字的去写
    :param filename: 保存的文件名
    :param list: 要保存的列表
    :param sep: 字符和标注之间的分隔符
    :return:
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for item in list:
            if item == '\n':
                f.write('\n')
            else:
                f.write(sep.join(item) + '\n')


def get_files(dir):
    """
    获取所有文件名
    :param dir: 目录
    :return: 目录下所有去重文件名的列表
    """
    file_list = []
    for roots, dirs, files in os.walk(dir):
        for file in files:
            file_list.append(file.split('.')[0])
    return list(set(file_list))


def due_files(dir, files, filename, sep=' '):
    """
    批量处理文件，最后写入文件
    :param dir: 目录
    :param filename: 处理后保存的文件名
    :param sep: 分隔符
    :return:
    """
    sentences = []
    for name in files:
        t_file = os.path.join(dir, name + '.txt')
        a_file = os.path.join(dir, name + '.ann')
        sentence = getPairs(t_file, a_file)
        for word in sentence:
            # 这里主要是将一篇文章分层几个句子，为后面好处理
            sentences.append(word)
            if word[0] == '。' and word[1] == 'O':
                sentences.append('\n')
    # 写入文件
    writefile(filename, sentences)


# 按照一定的比例进行分离训练集和测试集以及验证集
def split_data(dir, train_name, dev_name, test_name, split_ratio=[0.8, 0.1], sep=' '):
    file_list = get_files(dir)
    total = len(file_list)
    p1 = int(total * split_ratio[0])
    p2 = int(total * (split_ratio[0] + split_ratio[1]))
    due_files(dir, file_list[:p1], train_name)
    due_files(dir, file_list[p1:p2], dev_name)
    due_files(dir, file_list[p2:], test_name)


if __name__ == '__main__':
    # 获取实体数量
    getEntities(train_dir)
    # 测试路径
    # filepath_ann = './datas/ruijin_round1_train2_20181022/0.ann'
    # filepath_txt = './datas/ruijin_round1_train2_20181022/0.txt'
    # sen = readfile(filepath2)
    # pairslist = getNotations(filepath1)
    # print('sen list is {}'.format(sen))
    # print('pairslist is {}'.format(pairslist))
    # getPairs(filepath_txt, filepath_ann)
    # 标注测试
    # notation_sentence = getPairs(filepath_txt, filepath_ann)
    # print(notation_sentence)
    # 文件测试
    # files_list = get_files('./datas/ruijin_round1_train2_20181022')
    # print(files_list)
    # 最终测试
    # split_data(train_dir, train_name=train_file, test_name=test_file, dev_name=dev_file)
