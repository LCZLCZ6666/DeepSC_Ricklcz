# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:59:14 2020

@author: HQ Xie
"""
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import SNR_to_noise, initNetParams, train_step, val_step, train_mi
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
#parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=80, type=int)

# 设备配置
# - 知识点 ：根据是否有可用的GPU，选择使用GPU或CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置随机种子
# - 知识点 ：确保实验结果可重复，避免随机性导致的差异
def setup_seed(seed):
    torch.manual_seed(seed)          # 设置CPU随机种子
    torch.cuda.manual_seed_all(seed) # 设置GPU随机种子
    np.random.seed(seed)             # 设置NumPy随机种子
    random.seed(seed)               # 设置Python随机种子 内置的随机函数
    torch.backends.cudnn.deterministic = True # 设置CUDNN为确定性模式，确保结果可重复性

# 验证函数
# - 知识点 ：在验证集上评估模型性能，计算损失
def validate(epoch, args, net):
    test_eur = EurDataset('test') # 加载验证集
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    net.eval() # 切换到评估模式 --禁用dropout
    pbar = tqdm(test_iterator) # 创建进度条，显示验证进度
    total = 0  #初始化累计损失变量。
    with torch.no_grad(): # 验证时不需要反向传播
        '''遍历测试数据集的每个批次'''
        for sents in pbar:
            sents = sents.to(device) # 将数据移动到指定设备（CPU或GPU）
            loss = val_step(net, sents, sents, 0.1, pad_idx,
                             criterion, args.channel)

            total += loss # 累加当前批次的损失
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )

    return total/len(test_iterator) # 返回验证损失的平均值

# 功能 ：训练模型

# - 加载训练数据集
# - 创建数据加载器
# - 生成噪声标准差（在 SNR 5-10dB 范围内）
# - 遍历训练数据批次
# - 计算互信息（如果使用 mi_net）
# - 执行训练步骤
# - 显示训练进度、损失和互信息（如果使用）
def train(epoch, args, net, mi_net=None):
    #- 有 mi_net ：同时训练主网络和互信息网络（信息论优化）
    #- 无 mi_net ：只训练主网络（标准自编码器）
    train_eur= EurDataset('train') # 加载训练集
    # 创建数据加载器
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    #进度条
    pbar = tqdm(train_iterator)
    # 生成噪声标准差（在 SNR 5-10dB 范围内）
    # 数据增强策略 - 让模型在不同噪声水平下训练，增强泛化能力
    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))

    for sents in pbar: #带进度条的数据迭代器
        sents = sents.to(device) # 将数据移动到指定设备（CPU或GPU）

        if mi_net is not None:
            mi = train_mi(net, mi_net, sents, 0.1, pad_idx, mi_opt, args.channel)
            loss = train_step(net, sents, sents, 0.1, pad_idx,
                              optimizer, criterion, args.channel, mi_net)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}; MI {:.5f}'.format(
                    epoch + 1, loss, mi
                )
            )
#         - 使用随机噪声 noise_std[0] （而非固定0.1）
#         - 不传入 mi_net
        else:
            loss = train_step(net, sents, sents, noise_std[0], pad_idx,
                              optimizer, criterion, args.channel)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )


if __name__ == '__main__':
    # setup_seed(10)
    args = parser.parse_args()
    args.vocab_file = '/import/antennas/Datasets/hx301/' + args.vocab_file
    """ preparing the dataset """
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]


    """ define optimizer and loss function """
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                        num_vocab, num_vocab, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)
    mi_net = Mine().to(device)
    criterion = nn.CrossEntropyLoss(reduction = 'none')
    optimizer = torch.optim.Adam(deepsc.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay = 5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)
    #opt = NoamOpt(args.d_model, 1, 4000, optimizer)
    initNetParams(deepsc)
    for epoch in range(args.epochs):
        start = time.time()
        record_acc = 10

        train(epoch, args, deepsc)
        avg_acc = validate(epoch, args, deepsc)

        if avg_acc < record_acc:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            with open(args.checkpoint_path + '/checkpoint_{}.pth'.format(str(epoch + 1).zfill(2)), 'wb') as f:
                torch.save(deepsc.state_dict(), f)
            record_acc = avg_acc
    record_loss = []


    

        


