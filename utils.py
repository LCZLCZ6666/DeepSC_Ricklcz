# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:47:54 2020

@author: HQ Xie
utils.py
"""
import os 
import math
import torch
import time
import torch.nn as nn
import numpy as np
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
from models.mutual_info import sample_batch, mutual_information

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 计算机器生成文本的BLEU分数（NLP翻译/生成任务最核心的评价指标）
class BleuScore():
    """BLEU分数计算类"""
    def __init__(self, w1, w2, w3, w4):
        """
        初始化BLEU评分器
        
        参数：
        - w1: 1-gram权重
        - w2: 2-gram权重
        - w3: 3-gram权重
        - w4: 4-gram权重
        """
        self.w1 = w1  # 1-gram权重
        self.w2 = w2  # 2-gram权重
        self.w3 = w3  # 3-gram权重
        self.w4 = w4  # 4-gram权重
    
    # 输入真实句子和预测句子，输出BLEU评分
    def compute_blue_score(self, real, predicted):
        """
        计算BLEU分数
        
        参数：
        - real: 真实句子列表
        - predicted: 预测句子列表
        
        返回：
        - BLEU分数列表
        """
        score = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()  # 移除标签并分词
            sent2 = remove_tags(sent2).split()  # 移除标签并分词
            score.append(sentence_bleu([sent1], sent2, 
                          weights=(self.w1, self.w2, self.w3, self.w4)))  # 计算BLEU分数
        return score
            
# 标签平滑（深度学习防过拟合技术）
class LabelSmoothing(nn.Module):
    """标签平滑类"""
    def __init__(self, size, padding_idx, smoothing=0.0):
        """
        初始化标签平滑
        
        参数：
        - size: 词汇表大小
        - padding_idx: 填充标记的索引
        - smoothing: 平滑系数
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.CrossEntropyLoss()  # 交叉熵损失
        self.padding_idx = padding_idx  # 填充标记的索引
        self.confidence = 1.0 - smoothing  # 置信度
        self.smoothing = smoothing  # 平滑系数
        self.size = size  # 词汇表大小
        self.true_dist = None  # 真实分布
        
    def forward(self, x, target):
        """
        前向传播
        
        参数：
        - x: 模型输出 [batch_size, seq_len, vocab_size]
        - target: 目标序列 [batch_size, seq_len]
        
        返回：
        - 平滑后的损失
        """
        assert x.size(1) == self.size  # 确保输出维度正确
        true_dist = x.data.clone()  # 克隆输出数据
        # 将数组全部填充为平滑值
        true_dist.fill_(self.smoothing / (self.size - 2)) 
        # 按照目标索引设置置信度
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 
        # 填充位置的概率设为0
        true_dist[:, self.padding_idx] = 0 
        # 找到填充位置并将其概率设为0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)

# Noam学习率调度器（Transformer专用优化器包装器）
class NoamOpt:
    """Noam学习率调度器"""
    def __init__(self, model_size, factor, warmup, optimizer):
        """
        初始化Noam学习率调度器
        
        参数：
        - model_size: 模型维度
        - factor: 学习率因子
        - warmup: 预热步数
        - optimizer: 优化器
        """
        self.optimizer = optimizer  # 优化器
        self._step = 0  # 当前步数
        self.warmup = warmup  # 预热步数
        self.factor = factor  # 学习率因子
        self.model_size = model_size  # 模型维度
        self._rate = 0  # 当前学习率
        self._weight_decay = 0  # 当前权重衰减
        
    def step(self):
        """
        更新参数和学习率
        """
        self._step += 1  # 步数加1
        rate = self.rate()  # 计算当前学习率
        weight_decay = self.weight_decay()  # 计算当前权重衰减
        for p in self.optimizer.param_groups:
            p['lr'] = rate  # 设置学习率
            p['weight_decay'] = weight_decay  # 设置权重衰减
        self._rate = rate
        self._weight_decay = weight_decay
        # 更新权重
        self.optimizer.step()
        
    def rate(self, step=None):
        """
        计算学习率
        
        参数：
        - step: 步数
        
        返回：
        - 学习率
        """
        if step is None:
            step = self._step
            
        # 计算学习率
        lr = self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
  
        return lr
    
    def weight_decay(self, step=None):
        """
        计算权重衰减
        
        参数：
        - step: 步数
        
        返回：
        - 权重衰减
        """
        if step is None:
            step = self._step
            
        if step <= 3000 :
            weight_decay = 1e-3
            
        if step > 3000 and step <=9000:
            weight_decay = 0.0005
              
        if step>9000:
            weight_decay = 1e-4

        weight_decay = 0
        return weight_decay

# 序列转文本（模型输出数字→可读句子）            
class SeqtoText:
    """序列转文本类"""
    def __init__(self, vocb_dictionary, end_idx):
        """
        初始化序列转文本转换器
        
        参数：
        - vocb_dictionary: 词汇表字典
        - end_idx: 结束标记的索引
        """
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))  # 索引到标记的映射
        self.end_idx = end_idx  # 结束标记的索引
        
    def sequence_to_text(self, list_of_indices):
        """
        将索引序列转换为文本
        
        参数：
        - list_of_indices: 索引序列
        
        返回：
        - 文本字符串
        """
        # 查找词汇表中的单词
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:  # 遇到结束标记时停止
                break
            else:
                words.append(self.reverse_word_map.get(idx))  # 添加单词
        words = ' '.join(words)  # 组合为句子
        return words 

# 无线通信信道模拟（核心物理层模块）
# AWGN：加性高斯白噪声（基础噪声信道）
# Rayleigh：瑞利信道（多径衰落，无直射信号）
# Rician：莱斯信道（有直射信号，更接近现实）
class Channels():
    """信道模拟类"""

    def AWGN(self, Tx_sig, n_var):
        """
        加性高斯白噪声信道
        
        参数：
        - Tx_sig: 发送信号
        - n_var: 噪声方差
        
        返回：
        - 接收信号
        """
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)  # 添加高斯噪声
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var):
        """
        瑞利衰落信道
        
        参数：
        - Tx_sig: 发送信号
        - n_var: 噪声方差
        
        返回：
        - 接收信号
        """
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)  # 实部信道增益
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)  # 虚部信道增益
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)  # 信道矩阵
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)  # 应用信道增益
        Rx_sig = self.AWGN(Tx_sig, n_var)  # 添加噪声
        # 信道估计
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)  # 信道均衡

        return Rx_sig

    def Rician(self, Tx_sig, n_var, K=1):
        """
        莱斯衰落信道
        
        参数：
        - Tx_sig: 发送信号
        - n_var: 噪声方差
        - K: 莱斯因子
        
        返回：
        - 接收信号
        """
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))  # 均值
        std = math.sqrt(1 / (K + 1))  # 标准差
        H_real = torch.normal(mean, std, size=[1]).to(device)  # 实部信道增益
        H_imag = torch.normal(mean, std, size=[1]).to(device)  # 虚部信道增益
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)  # 信道矩阵
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)  # 应用信道增益
        Rx_sig = self.AWGN(Tx_sig, n_var)  # 添加噪声
        # 信道估计
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)  # 信道均衡

        return Rx_sig

# 网络参数初始化
def initNetParams(model):
    """
    初始化网络参数
    
    参数：
    - model: 模型
    
    返回：
    - 初始化后的模型
    """
    '''Init net parameters.'''
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)  # 使用Xavier均匀初始化
    return model

# Transformer掩码矩阵
def subsequent_mask(size):
    """
    生成后续掩码
    
    参数：
    - size: 序列长度
    
    返回：
    - 掩码矩阵
    """
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    # 产生下三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)

# 创建编码器掩码和解码器掩码    
def create_masks(src, trg, padding_idx):
    """
    创建掩码
    
    参数：
    - src: 源序列
    - trg: 目标序列
    - padding_idx: 填充标记的索引
    
    返回：
    - 源掩码和目标掩码
    """

    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]

    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)  # 前瞻掩码
    combined_mask = torch.max(trg_mask, look_ahead_mask)  # 组合掩码
    
    return src_mask.to(device), combined_mask.to(device)

# 带掩码的损失计算
def loss_function(x, trg, padding_idx, criterion):
    """
    计算带掩码的损失
    
    参数：
    - x: 模型输出 [batch_size * seq_len, vocab_size]
    - trg: 目标序列 [batch_size * seq_len]
    - padding_idx: 填充标记的索引
    - criterion: 损失函数
    
    返回：
    - 平均损失
    """
    
    loss = criterion(x, trg)  # 计算损失
    mask = (trg != padding_idx).type_as(loss.data)  # 生成掩码
    loss *= mask  # 应用掩码
    
    return loss.mean()  # 返回平均损失

# 信号功率归一化
def PowerNormalize(x):
    """
    信号功率归一化
    
    参数：
    - x: 输入信号
    
    返回：
    - 归一化后的信号
    """
    
    x_square = torch.mul(x, x)  # 计算信号平方
    power = torch.mean(x_square).sqrt()  # 计算信号功率
    if power > 1:  # 如果功率大于1，进行归一化
        x = torch.div(x, power)  # 归一化信号
    
    return x

# 信噪比转噪声标准差
def SNR_to_noise(snr):
    """
    将信噪比转换为噪声标准差
    
    参数：
    - snr: 信噪比（dB）
    
    返回：
    - 噪声标准差
    """
    snr = 10 ** (snr / 10)  # 将dB转换为线性值
    noise_std = 1 / np.sqrt(2 * snr)  # 计算噪声标准差

    return noise_std

# 模型单步训练
def train_step(model, src, trg, n_var, pad, opt, criterion, channel, mi_net=None):
    """
    训练单步
    
    参数：
    - model: 模型
    - src: 源序列
    - trg: 目标序列
    - n_var: 噪声方差
    - pad: 填充标记的索引
    - opt: 优化器
    - criterion: 损失函数
    - channel: 信道类型
    - mi_net: 互信息网络（可选）
    
    返回：
    - 损失值
    """
    model.train()  # 切换到训练模式

    trg_inp = trg[:, :-1]  # 目标输入（去掉最后一个token）
    trg_real = trg[:, 1:]  # 目标真实值（去掉第一个token）

#输入给模型看：  <SOS>   我     爱    编程
#希望模型预测：   我     爱    编程   <EOS>

    channels = Channels()  # 初始化信道
    opt.zero_grad()  # 清零梯度
    
    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)  # 创建掩码
    
    enc_output = model.encoder(src, src_mask)  # 编码
    channel_enc_output = model.channel_encoder(enc_output)  # 信道编码
    Tx_sig = PowerNormalize(channel_enc_output)  # 功率归一化

    # 通过信道
    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)  # AWGN信道
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)  # Rayleigh信道
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)  # Rician信道
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output = model.channel_decoder(Rx_sig)  # 信道解码
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)  # 解码
    pred = model.dense(dec_output)  # 线性变换到词汇表空间
    
    ntokens = pred.size(-1)  # 词汇表大小
    
    # 计算损失
    loss = loss_function(pred.contiguous().view(-1, ntokens), 
                         trg_real.contiguous().view(-1), 
                         pad, criterion)

    # 如果使用互信息网络
    if mi_net is not None:
        mi_net.eval()  # 切换到评估模式
        joint, marginal = sample_batch(Tx_sig, Rx_sig)  # 采样批次
        mi_lb, _, _ = mutual_information(joint, marginal, mi_net)  # 计算互信息下界
        loss_mine = -mi_lb  # 互信息损失
        loss = loss + 0.0009 * loss_mine  # 总损失

    loss.backward()  # 反向传播
    opt.step()  # 更新参数

    return loss.item()  # 返回损失值

# 训练互信息网络
def train_mi(model, mi_net, src, n_var, padding_idx, opt, channel):
    """
    训练互信息网络
    
    参数：
    - model: 主模型
    - mi_net: 互信息网络
    - src: 源序列
    - n_var: 噪声方差
    - padding_idx: 填充标记的索引
    - opt: 优化器
    - channel: 信道类型
    
    返回：
    - 互信息损失
    """
    mi_net.train()  # 切换到训练模式
    opt.zero_grad()  # 清零梯度
    channels = Channels()  # 初始化信道
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  # [batch, 1, seq_len]
    enc_output = model.encoder(src, src_mask)  # 编码
    channel_enc_output = model.channel_encoder(enc_output)  # 信道编码
    Tx_sig = PowerNormalize(channel_enc_output)  # 功率归一化

    # 通过信道
    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)  # AWGN信道
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)  # Rayleigh信道
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)  # Rician信道
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    joint, marginal = sample_batch(Tx_sig, Rx_sig)  # 采样批次
    mi_lb, _, _ = mutual_information(joint, marginal, mi_net)  # 计算互信息下界
    loss_mine = -mi_lb  # 互信息损失

    loss_mine.backward()  # 反向传播
    torch.nn.utils.clip_grad_norm_(mi_net.parameters(), 10.0)  # 梯度裁剪
    opt.step()  # 更新参数

    return loss_mine.item()  # 返回损失值

# 评估模型
def val_step(model, src, trg, n_var, pad, criterion, channel):
    """
    评估单步
    
    参数：
    - model: 模型
    - src: 源序列
    - trg: 目标序列
    - n_var: 噪声方差
    - pad: 填充标记的索引
    - criterion: 损失函数
    - channel: 信道类型
    
    返回：
    - 损失值
    """
    channels = Channels()  # 初始化信道
    trg_inp = trg[:, :-1]  # 目标输入（去掉最后一个token）
    trg_real = trg[:, 1:]  # 目标真实值（去掉第一个token）

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)  # 创建掩码

    enc_output = model.encoder(src, src_mask)  # 编码
    channel_enc_output = model.channel_encoder(enc_output)  # 信道编码
    Tx_sig = PowerNormalize(channel_enc_output)  # 功率归一化

    # 通过信道
    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)  # AWGN信道
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)  # Rayleigh信道
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)  # Rician信道
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output = model.channel_decoder(Rx_sig)  # 信道解码
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)  # 解码
    pred = model.dense(dec_output)  # 线性变换到词汇表空间

    ntokens = pred.size(-1)  # 词汇表大小
    # 计算损失
    loss = loss_function(pred.contiguous().view(-1, ntokens), 
                         trg_real.contiguous().view(-1), 
                         pad, criterion)
    
    return loss.item()  # 返回损失值
    

# 贪婪解码（模型推理生成句子）
def greedy_decode(model, src, n_var, max_len, padding_idx, start_symbol, channel):
    """
    贪婪解码
    
    参数：
    - model: 模型
    - src: 源序列
    - n_var: 噪声方差
    - max_len: 最大序列长度
    - padding_idx: 填充标记的索引
    - start_symbol: 开始标记
    - channel: 信道类型
    
    返回：
    - 生成的序列
    """
    """
    这里采用贪婪解码器，如果需要更好的性能情况下，可以使用beam search decode
    """
    # 创建源掩码
    channels = Channels()  # 初始化信道
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  # [batch, 1, seq_len]

    enc_output = model.encoder(src, src_mask)  # 编码
    channel_enc_output = model.channel_encoder(enc_output)  # 信道编码
    Tx_sig = PowerNormalize(channel_enc_output)  # 功率归一化

    # 通过信道
    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)  # AWGN信道
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)  # Rayleigh信道
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)  # Rician信道
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
            
    memory = model.channel_decoder(Rx_sig)  # 信道解码
    
    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)  # 初始化输出序列

    for i in range(max_len - 1):
        # 创建解码掩码
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)  # 前瞻掩码
        combined_mask = torch.max(trg_mask, look_ahead_mask)  # 组合掩码
        combined_mask = combined_mask.to(device)  # 移动到设备

        # 解码接收的信号
        dec_output = model.decoder(outputs, memory, combined_mask, None)  # 解码
        pred = model.dense(dec_output)  # 线性变换到词汇表空间
        
        # 预测单词
        prob = pred[:, -1:, :]  # (batch_size, 1, vocab_size)

        # 返回最大概率的索引
        _, next_word = torch.max(prob, dim=-1)  # 贪婪选择
        
        outputs = torch.cat([outputs, next_word], dim=1)  # 拼接输出序列

    return outputs  # 返回生成的序列

# utils.py 总结：
# 1. 该文件包含了DeepSC模型的辅助函数和工具类
# 2. 主要功能包括：
#    - BLEU分数计算：评估生成文本的质量
#    - 标签平滑：防止过拟合
#    - Noam学习率调度器：优化学习率
#    - 序列转文本：将模型输出的数字序列转换为可读文本
#    - 信道模拟：模拟不同类型的通信信道
#    - 网络参数初始化：初始化模型参数
#    - 掩码生成：生成Transformer所需的掩码
#    - 损失计算：计算带掩码的损失
#    - 信号处理：功率归一化、信噪比转换
#    - 模型训练和评估：训练步骤、互信息训练、评估步骤
#    - 贪婪解码：模型推理时生成文本
# 3. 这些工具函数和类为DeepSC模型的训练、评估和推理提供了必要的支持
# 4. 特别重要的组件：
#    - Channels类：模拟不同的通信信道，是物理层模拟的核心
#    - train_step函数：实现模型的训练逻辑
#    - greedy_decode函数：实现模型的推理逻辑
#    - SNR_to_noise函数：将信噪比转换为噪声标准差，用于信道模拟