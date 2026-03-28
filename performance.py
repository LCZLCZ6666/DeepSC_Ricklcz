# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: performance.py
@Time: 2021/4/1 11:48
"""
import os
import json
import torch
import argparse
import numpy as np
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm
from sklearn.preprocessing import normalize
# from bert4keras.backend import keras
# from bert4keras.models import build_bert_model
# from bert4keras.tokenizers import Tokenizer
from w3lib.html import remove_tags

# 创建命令行参数解析器
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='europarl/train_data.pkl', type=str, help='数据目录')
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str, help='词汇表文件')
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str, help='模型检查点路径')
parser.add_argument('--channel', default='Rayleigh', type=str, help='信道类型')
parser.add_argument('--MAX-LENGTH', default=30, type=int, help='最大序列长度')
parser.add_argument('--MIN-LENGTH', default=4, type=int, help='最小序列长度')
parser.add_argument('--d-model', default=128, type=int, help='模型维度')
parser.add_argument('--dff', default=512, type=int, help='前馈网络隐藏层维度')
parser.add_argument('--num-layers', default=4, type=int, help='编码器和解码器层数')
parser.add_argument('--num-heads', default=8, type=int, help='注意力头数')
parser.add_argument('--batch-size', default=64, type=int, help='批处理大小')
parser.add_argument('--epochs', default=2, type=int, help='评估轮数')
parser.add_argument('--bert-config-path', default='bert/cased_L-12_H-768_A-12/bert_config.json', type=str, help='BERT配置路径')
parser.add_argument('--bert-checkpoint-path', default='bert/cased_L-12_H-768_A-12/bert_model.ckpt', type=str, help='BERT检查点路径')
parser.add_argument('--bert-dict-path', default='bert/cased_L-12_H-768_A-12/vocab.txt', type=str, help='BERT词典路径')

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 使用预训练模型计算句子相似度
# class Similarity():
#     def __init__(self, config_path, checkpoint_path, dict_path):
#         self.model1 = build_bert_model(config_path, checkpoint_path, with_pool=True)
#         self.model = keras.Model(inputs=self.model1.input,
#                                  outputs=self.model1.get_layer('Encoder-11-FeedForward-Norm').output)
#         # build tokenizer
#         self.tokenizer = Tokenizer(dict_path, do_lower_case=True)
#
#     def compute_similarity(self, real, predicted):
#         token_ids1, segment_ids1 = [], []
#         token_ids2, segment_ids2 = [], []
#         score = []
#
#         for (sent1, sent2) in zip(real, predicted):
#             sent1 = remove_tags(sent1)
#             sent2 = remove_tags(sent2)
#
#             ids1, sids1 = self.tokenizer.encode(sent1)
#             ids2, sids2 = self.tokenizer.encode(sent2)
#
#             token_ids1.append(ids1)
#             token_ids2.append(ids2)
#             segment_ids1.append(sids1)
#             segment_ids2.append(sids2)
#
#         token_ids1 = keras.preprocessing.sequence.pad_sequences(token_ids1, maxlen=32, padding='post')
#         token_ids2 = keras.preprocessing.sequence.pad_sequences(token_ids2, maxlen=32, padding='post')
#
#         segment_ids1 = keras.preprocessing.sequence.pad_sequences(segment_ids1, maxlen=32, padding='post')
#         segment_ids2 = keras.preprocessing.sequence.pad_sequences(segment_ids2, maxlen=32, padding='post')
#
#         vector1 = self.model.predict([token_ids1, segment_ids1])
#         vector2 = self.model.predict([token_ids2, segment_ids2])
#
#         vector1 = np.sum(vector1, axis=1)
#         vector2 = np.sum(vector2, axis=1)
#
#         vector1 = normalize(vector1, axis=0, norm='max')
#         vector2 = normalize(vector2, axis=0, norm='max')
#
#         dot = np.diag(np.matmul(vector1, vector2.T))  # a*b
#         a = np.diag(np.matmul(vector1, vector1.T))  # a*a
#         b = np.diag(np.matmul(vector2, vector2.T))
#
#         a = np.sqrt(a)
#         b = np.sqrt(b)
#
#         output = dot / (a * b)
#         score = output.tolist()
#
#         return score

# 测试模型在不同信噪比下的通信效果
def performance(args, SNR, net):
    """
    评估模型性能
    
    参数：
    - args: 命令行参数
    - SNR: 信噪比列表
    - net: 模型
    
    返回：
    - 不同SNR下的BLEU分数
    """
    # similarity = Similarity(args.bert_config_path, args.bert_checkpoint_path, args.bert_dict_path)
    # 初始化BLEU评分器（1-gram）
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    # 加载测试数据集
    test_eur = EurDataset('test')
    # 创建数据加载器
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)

    # 初始化序列到文本的转换器
    StoT = SeqtoText(token_to_idx, end_idx)

    # 存储多轮测试的BLEU分数
    score = []
    score2 = []

    # 切换模型到评估模式
    net.eval()
    with torch.no_grad():  # 不计算梯度，节省显存、加速推理
        # 循环跑多轮测试
        for epoch in range(args.epochs):
            Tx_word = []  # 存储模型输出的文本
            Rx_word = []  # 存储目标文本

            # 在不同信号强弱下分别测试准确率
            for snr in tqdm(SNR):
                word = []  # 存储当前SNR下模型输出的文本
                target_word = []  # 存储当前SNR下的目标文本
                # 将SNR转换为噪声标准差
                noise_std = SNR_to_noise(snr)

                # 遍历测试集里的每一批句子
                for sents in test_iterator:
                    # 将数据移动到设备
                    sents = sents.to(device)
                    target = sents

                    # 使用贪婪解码生成输出
                    out = greedy_decode(net, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                        start_idx, args.channel)

                    # 将模型输出的数字序列转换为文本
                    sentences = out.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    word = word + result_string

                    # 将目标数字序列转换为文本
                    target_sent = target.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string

                Tx_word.append(word)
                Rx_word.append(target_word)

            # 计算BLEU分数
            bleu_score = []
            sim_score = []
            for sent1, sent2 in zip(Tx_word, Rx_word):
                # 计算1-gram BLEU分数
                bleu_score.append(bleu_score_1gram.compute_blue_score(sent1, sent2))
                # sim_score.append(similarity.compute_similarity(sent1, sent2))
            # 计算每个SNR下的平均BLEU分数
            bleu_score = np.array(bleu_score)
            bleu_score = np.mean(bleu_score, axis=1)
            score.append(bleu_score)

            # sim_score = np.array(sim_score)
            # sim_score = np.mean(sim_score, axis=1)
            # score2.append(sim_score)
    # 计算多轮测试的平均BLEU分数
    score1 = np.mean(np.array(score), axis=0)
    # score2 = np.mean(np.array(score2), axis=0)

    return score1  # , score2

# 程序入口
if __name__ == '__main__':
    args = parser.parse_args()  # 解析命令行参数
    # 定义SNR值列表
    SNR = [0, 3, 6, 9, 12, 15, 18]

    # 初始化阶段
    args.vocab_file = '/import/antennas/Datasets/hx301/' + args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))  # 加载词汇表
    token_to_idx = vocab['token_to_idx']  # 标记到索引的映射
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))  # 索引到标记的映射
    num_vocab = len(token_to_idx)  # 词汇表大小
    pad_idx = token_to_idx["<PAD>"]  # 填充标记的索引
    start_idx = token_to_idx["<START>"]  # 开始标记的索引
    end_idx = token_to_idx["<END>"]  # 结束标记的索引

    # 定义模型
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                        num_vocab, num_vocab, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)

    # 收集模型检查点路径
    model_paths = []
    for fn in os.listdir(args.checkpoint_path):
        if not fn.endswith('.pth'): continue  # 只处理.pth文件
        idx = int(os.path.splitext(fn)[0].split('_')[-1])  # 读取模型索引
        model_paths.append((os.path.join(args.checkpoint_path, fn), idx))

    # 按索引排序模型路径
    model_paths.sort(key=lambda x: x[1])

    # 加载最新的模型
    model_path, _ = model_paths[-1]
    checkpoint = torch.load(model_path)
    deepsc.load_state_dict(checkpoint)
    print('model load!')

    # 评估模型性能
    bleu_score = performance(args, SNR, deepsc)
    print(bleu_score)

    # similarity.compute_similarity(sent1, real)

# performance.py 总结：
# 1. 该脚本用于评估DeepSC模型在不同信噪比下的性能
# 2. 主要功能包括：
#    - 加载训练好的模型
#    - 在不同SNR条件下测试模型性能
#    - 使用贪婪解码生成输出文本
#    - 计算BLEU分数评估生成文本的质量
# 3. 评估流程：
#    - 加载测试数据
#    - 对每个SNR值，使用模型生成输出
#    - 将模型输出和目标转换为文本
#    - 计算BLEU分数
#    - 输出不同SNR下的平均BLEU分数
# 4. 评估指标：
#    - BLEU分数：衡量生成文本与参考文本的相似度
#    - 可选的语义相似度：使用BERT模型计算语义相似度（当前已注释）