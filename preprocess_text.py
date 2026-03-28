# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: text_preprocess.py
@Time: 2021/3/31 22:14
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:44:08 2020

@author: hx301
"""
import unicodedata
import re
from w3lib.html import remove_tags
import pickle
import argparse
import os
import json
from tqdm import tqdm

# 创建命令行参数解析器
parser = argparse.ArgumentParser()
parser.add_argument('--input-data-dir', default='europarl/en', type=str, help='输入数据目录')
parser.add_argument('--output-train-dir', default='europarl/train_data.pkl', type=str, help='训练数据输出路径')
parser.add_argument('--output-test-dir', default='europarl/test_data.pkl', type=str, help='测试数据输出路径')
parser.add_argument('--output-vocab', default='europarl/vocab.json', type=str, help='词汇表输出路径')

# 定义特殊标记及其ID
SPECIAL_TOKENS = {
  '<PAD>': 0,  # 填充标记
  '<START>': 1,  # 开始标记
  '<END>': 2,  # 结束标记
  '<UNK>': 3,  # 未知标记
}

# Unicode转ASCII
# s = [w1, w2, ..., wL]
#第三页右下角寒色标亮……
def unicode_to_ascii(s):
    """
    将Unicode字符串转换为ASCII字符串
     
    参数：
    - s: Unicode字符串
    
    返回：
    - ASCII字符串
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s)  # 将字符分解为基本字符和变音符号
                   if unicodedata.category(c) != 'Mn')  # 过滤掉非间距标记

# 文本预处理，清理和标准化文本数据
def normalize_string(s):
    """
    标准化字符串
    
    参数：
    - s: 原始字符串
    
    返回：
    - 标准化后的字符串
    """
    # 标准化Unicode字符
    s = unicode_to_ascii(s)
    # 移除XML标签
    s = remove_tags(s)
    # 在标点符号前添加空格
    s = re.sub(r'([!.?])', r' \1', s)
    # 移除非字母和标点符号的字符
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    # 合并多个空格为单个空格
    s = re.sub(r'\s+', r' ', s)
    # 转换为小写
    s = s.lower()
    return s

# 文本截断，保留长度在MIN_LENGTH和MAX_LENGTH之间的句子
def cutted_data(cleaned, MIN_LENGTH=4, MAX_LENGTH=30):
    """
    过滤并截断数据
    
    参数：
    - cleaned: 清洗后的句子列表
    - MIN_LENGTH: 句子最小长度
    - MAX_LENGTH: 句子最大长度
    
    返回：
    - 过滤后的句子列表
    """
    cutted_lines = list()
    for line in cleaned:
        length = len(line.split())  # 计算句子长度
        if length > MIN_LENGTH and length < MAX_LENGTH:  # 过滤长度合适的句子
            line = [word for word in line.split()]  # 分词
            cutted_lines.append(' '.join(line))  # 重新组合为句子
    return cutted_lines

# 将处理后的句子保存为pickle文件
def save_clean_sentences(sentence, save_path):
    """
    保存清洗后的句子
    
    参数：
    - sentence: 句子列表
    - save_path: 保存路径
    """
    pickle.dump(sentence, open(save_path, 'wb'))  # 使用pickle序列化保存
    print('Saved: %s' % save_path)

# 处理单个文本文件
def process(text_path):
    """
    处理单个文本文件
    
    参数：
    - text_path: 文本文件路径
    
    返回：
    - 处理后的句子列表
    """
    fop = open(text_path, 'r', encoding='utf8')  # 打开文件
    raw_data = fop.read()  # 读取文件内容
    sentences = raw_data.strip().split('\n')  # 按行分割成句子
    raw_data_input = [normalize_string(data) for data in sentences]  # 对每个句子进行标准化处理
    raw_data_input = cutted_data(raw_data_input)  # 过滤句子长度
    fop.close()  # 关闭文件

    return raw_data_input

# 分词和词汇表构建函数
# 一个句子变成了字典
def tokenize(s, delim=' ', add_start_token=True, add_end_token=True,
             punct_to_keep=None, punct_to_remove=None):
    """
    对序列进行分词
    
    参数：
    - s: 输入字符串
    - delim: 分隔符
    - add_start_token: 是否添加开始标记
    - add_end_token: 是否添加结束标记
    - punct_to_keep: 需要保留的标点符号
    - punct_to_remove: 需要移除的标点符号
    
    返回：
    - 分词后的标记列表
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))  # 在保留的标点前添加空格

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')  # 移除指定的标点符号

    tokens = s.split(delim)  # 分词
    if add_start_token:
        tokens.insert(0, '<START>')  # 添加开始标记
    if add_end_token:
        tokens.append('<END>')  # 添加结束标记
    return tokens

# 构建词汇表
# 构建了以频率为下标的字典。
def build_vocab(sequences, token_to_idx={}, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None):
    """
    构建词汇表
    
    参数：
    - sequences: 序列列表
    - token_to_idx: 初始标记到索引的映射
    - min_token_count: 最小标记出现次数
    - delim: 分隔符
    - punct_to_keep: 需要保留的标点符号
    - punct_to_remove: 需要移除的标点符号
    
    返回：
    - 标记到索引的映射
    """
    token_to_count = {}  # 标记到出现次数的映射

    for seq in sequences:
        # 对每个句子进行分词
        seq_tokens = tokenize(seq, delim=delim, punct_to_keep=punct_to_keep,
                        punct_to_remove=punct_to_remove,
                        add_start_token=False, add_end_token=False)

        # 统计每个标记的出现次数
        for token in seq_tokens:
            if token not in token_to_count:
                token_to_count[token] = 0
            token_to_count[token] += 1

        # 为每个标记分配唯一ID
    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count:  # 过滤出现次数低于阈值的标记
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx

# 将标记序列转换为ID序列
def encode(seq_tokens, token_to_idx, allow_unk=False):
    """
    将标记序列编码为ID序列
    
    参数：
    - seq_tokens: 标记序列
    - token_to_idx: 标记到索引的映射
    - allow_unk: 是否允许未知标记
    
    返回：
    - ID序列
    """
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'  # 替换为未知标记
            else:
                raise KeyError('Token "%s" not in vocab' % token)  # 抛出错误
        seq_idx.append(token_to_idx[token])  # 添加标记对应的ID
    return seq_idx

# 将ID序列转换回标记序列
def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    """
    将ID序列解码为标记序列
    
    参数：
    - seq_idx: ID序列
    - idx_to_token: 索引到标记的映射
    - delim: 分隔符
    - stop_at_end: 是否在遇到结束标记时停止
    
    返回：
    - 标记序列或字符串
    """
    tokens = []
    for idx in seq_idx:
        tokens.append(idx_to_token[idx])  # 添加ID对应的标记
        if stop_at_end and tokens[-1] == '<END>':  # 遇到结束标记时停止
            break
    if delim is None:
        return tokens  # 返回标记列表
    else:
        return delim.join(tokens)  # 返回用分隔符连接的字符串

# 主函数
def main(args):
    """
    主函数
    
    参数：
    - args: 命令行参数
    """
    # 数据目录
    data_dir = '/import/antennas/Datasets/hx301/'
    args.input_data_dir = data_dir + args.input_data_dir
    args.output_train_dir = data_dir + args.output_train_dir
    args.output_test_dir = data_dir + args.output_test_dir
    args.output_vocab = data_dir + args.output_vocab

    print(args.input_data_dir)
    sentences = []
    print('Preprocess Raw Text')
    # 遍历输入目录中的所有文件
    for fn in tqdm(os.listdir(args.input_data_dir)):
        if not fn.endswith('.txt'): continue  # 只处理txt文件
        process_sentences = process(os.path.join(args.input_data_dir, fn))  # 处理文件
        sentences += process_sentences  # 收集处理后的句子

    # 去重
    a = {}
    for set in sentences:
        if set not in a:
            a[set] = 0
        a[set] += 1
    sentences = list(a.keys())  # 去重后的句子列表
    print('Number of sentences: {}'.format(len(sentences)))
    
    # 构建词汇表
    print('Build Vocab')
    token_to_idx = build_vocab(
        sentences, SPECIAL_TOKENS,
        punct_to_keep=[';', ','], punct_to_remove=['?', '.']
    )

    vocab = {'token_to_idx': token_to_idx}  # 词汇表字典
    print('Number of words in Vocab: {}'.format(len(token_to_idx)))

    # 保存词汇表
    if args.output_vocab != '':
        with open(args.output_vocab, 'w') as f:
            json.dump(vocab, f)  # 将词汇表保存为JSON文件

    # 编码句子为ID序列
    print('Start encoding txt')
    results = []
    for seq in tqdm(sentences):
        words = tokenize(seq, punct_to_keep=[';', ','], punct_to_remove=['?', '.'])  # 分词
        tokens = [token_to_idx[word] for word in words]  # 转换为ID
        results.append(tokens)  # 收集ID序列

    # 保存数据
    print('Writing Data')
    train_data = results[:round(len(results) * 0.9)]  # 训练集（90%）
    test_data = results[round(len(results) * 0.9):]  # 测试集（10%）
    # 保存训练集
    with open(args.output_train_dir, 'wb') as f:
        pickle.dump(train_data, f)
    # 保存测试集
    with open(args.output_test_dir, 'wb') as f:
        pickle.dump(test_data, f)

# 程序入口
if __name__ == '__main__':
    args = parser.parse_args()  # 解析命令行参数
    main(args)  # 执行主函数

# preprocess_text.py 总结：
# 1. 该脚本用于预处理文本数据，为模型训练做准备
# 2. 主要功能包括：
#    - 文本清洗和标准化：移除XML标签、转换为小写、处理标点符号
#    - 句子长度过滤：保留长度在指定范围内的句子
#    - 词汇表构建：统计词频并生成标记到索引的映射
#    - 数据编码：将文本转换为ID序列
#    - 数据分割：将数据分为训练集和测试集
# 3. 处理流程：
#    - 读取文本文件
#    - 清洗和标准化文本
#    - 过滤句子长度
#    - 构建词汇表
#    - 编码句子为ID序列
#    - 保存处理后的数据和词汇表