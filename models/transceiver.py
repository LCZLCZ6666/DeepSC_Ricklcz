# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:33:53 2020

@author: HQ Xie
这是一个Transformer的网络结构
"""
"""
Transformer includes:
    Encoder
        1. Positional coding
        2. Multihead-attention
        3. PositionwiseFeedForward
    Decoder
        1. Positional coding
        2. Multihead-attention
        3. Multihead-attention
        4. PositionwiseFeedForward
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math

# 位置编码类：为输入序列添加位置信息，使模型能够捕捉序列的顺序特征
class PositionalEncoding(nn.Module):
    """实现位置编码功能"""
    def __init__(self, d_model, dropout, max_len=5000):
        """
        初始化位置编码
        
        参数：
        - d_model: 模型维度
        - dropout: dropout概率
        - max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)  # 初始化dropout层
        
        # 预计算位置编码
        pe = torch.zeros(max_len, d_model)  # 初始化位置编码矩阵
        position = torch.arange(0, max_len).unsqueeze(1)  # 生成位置索引 [max_len, 1]
        # 计算除数项，用于生成不同频率的正弦和余弦函数
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦函数
        pe = pe.unsqueeze(0)  # 添加批次维度 [1, max_len, d_model]
        self.register_buffer('pe', pe)  # 将位置编码注册为缓冲区，确保在模型加载时加载
        
    def forward(self, x):
        """
        前向传播
        
        参数：
        - x: 输入张量 [batch_size, seq_len, d_model]
        
        返回：
        - 添加位置编码后的张量 [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]  # 添加位置编码
        x = self.dropout(x)  # 应用dropout
        return x

# 位置编码类总结：
# 1. 为每个位置生成唯一的编码，使模型能够区分不同位置的token
# 2. 使用正弦和余弦函数生成位置编码，具有周期性和单调性
# 3. 位置编码与输入嵌入相加，不增加模型参数
# 4. 预计算位置编码并注册为缓冲区，提高计算效率

# 多头注意力类：实现多头注意力机制，允许模型同时关注序列的不同位置和不同表示子空间
class MultiHeadedAttention(nn.Module):
    """实现多头注意力机制"""
    def __init__(self, num_heads, d_model, dropout=0.1):
        """
        初始化多头注意力
        
        参数：
        - num_heads: 注意力头数
        - d_model: 模型维度
        - dropout: dropout概率
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0  # 确保模型维度能被注意力头数整除
        # 假设d_v等于d_k
        self.d_k = d_model // num_heads  # 每个注意力头的维度
        self.num_heads = num_heads  # 注意力头数
        
        # 定义线性变换层
        self.wq = nn.Linear(d_model, d_model)  # 查询线性层
        self.wk = nn.Linear(d_model, d_model)  # 键线性层
        self.wv = nn.Linear(d_model, d_model)  # 值线性层
        self.dense = nn.Linear(d_model, d_model)  # 输出线性层
        
        self.attn = None  # 存储注意力权重
        self.dropout = nn.Dropout(p=dropout)  # dropout层
        
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        参数：
        - query: 查询张量 [batch_size, seq_len, d_model]
        - key: 键张量 [batch_size, seq_len, d_model]
        - value: 值张量 [batch_size, seq_len, d_model]
        - mask: 掩码张量 [batch_size, 1, seq_len]
        
        返回：
        - 注意力加权后的张量 [batch_size, seq_len, d_model]
        """
        if mask is not None:
            # 对所有注意力头应用相同的掩码
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)  # 批次大小
        
        # 1) 执行线性投影并分割成多个头
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)  # [batch, heads, seq_len, d_k]
        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)  # [batch, heads, seq_len, d_k]
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)  # [batch, heads, seq_len, d_k]
        
        # 2) 对所有投影向量应用注意力
        x, self.attn = self.attention(query, key, value, mask=mask)
        
        # 3) 拼接多头结果并应用最终线性层
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)  # 拼接多头
        x = self.dense(x)  # 线性变换
        x = self.dropout(x)  # 应用dropout
        
        return x
    
    def attention(self, query, key, value, mask=None):
        """
        计算缩放点积注意力
        
        参数：
        - query: 查询张量 [batch, heads, seq_len, d_k]
        - key: 键张量 [batch, heads, seq_len, d_k]
        - value: 值张量 [batch, heads, seq_len, d_k]
        - mask: 掩码张量 [batch, heads, seq_len, seq_len]
        
        返回：
        - 注意力加权值 [batch, heads, seq_len, d_k]
        - 注意力权重 [batch, heads, seq_len, seq_len]
        """
        d_k = query.size(-1)  # 每个注意力头的维度
        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 缩放点积
        if mask is not None:
            # 根据掩码，将掩码位置填充为-1e9
            scores += (mask * -1e9)
        p_attn = F.softmax(scores, dim=-1)  # 计算注意力权重
        return torch.matmul(p_attn, value), p_attn  # 返回注意力加权值和注意力权重

# 多头注意力类总结：
# 1. 将输入向量投影到多个子空间，每个子空间由一个注意力头处理
# 2. 每个注意力头关注序列的不同位置和不同特征
# 3. 拼接多头结果，增强模型的表达能力
# 4. 使用缩放点积注意力，防止分数过大导致softmax梯度消失

# 前馈神经网络类：对每个位置的表示进行非线性变换
class PositionwiseFeedForward(nn.Module):
    """实现前馈神经网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化前馈神经网络
        
        参数：
        - d_model: 模型维度
        - d_ff: 前馈网络隐藏层维度
        - dropout: dropout概率
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # 第一层线性变换
        self.w_2 = nn.Linear(d_ff, d_model)  # 第二层线性变换
        self.dropout = nn.Dropout(dropout)  # dropout层

    def forward(self, x):
        """
        前向传播
        
        参数：
        - x: 输入张量 [batch_size, seq_len, d_model]
        
        返回：
        - 变换后的张量 [batch_size, seq_len, d_model]
        """
        x = self.w_1(x)  # 线性变换到高维空间
        x = F.relu(x)  # 非线性激活
        x = self.w_2(x)  # 线性变换回原始维度
        x = self.dropout(x)  # 应用dropout
        return x

# 前馈神经网络类总结：
# 1. 对每个位置的表示独立进行非线性变换
# 2. 增加模型的非线性表达能力
# 3. 通常使用较大的隐藏层维度（如d_model的4倍）

# 编码器层类：编码器的基本组成单元，包含自注意力层和前馈网络层
class EncoderLayer(nn.Module):
    """编码器层"""
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        """
        初始化编码器层
        
        参数：
        - d_model: 模型维度
        - num_heads: 注意力头数
        - dff: 前馈网络隐藏层维度
        - dropout: dropout概率
        """
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)  # 自注意力层
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=0.1)  # 前馈网络层
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)  # 层归一化1
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)  # 层归一化2
        

    def forward(self, x, mask):
        """
        前向传播
        
        参数：
        - x: 输入张量 [batch_size, seq_len, d_model]
        - mask: 掩码张量 [batch_size, 1, seq_len]
        
        返回：
        - 编码后的张量 [batch_size, seq_len, d_model]
        """
        attn_output = self.mha(x, x, x, mask)  # 自注意力计算
        x = self.layernorm1(x + attn_output)  # 残差连接+层归一化
        
        ffn_output = self.ffn(x)  # 前馈网络计算
        x = self.layernorm2(x + ffn_output)  # 残差连接+层归一化
        
        return x

# 编码器层类总结：
# 1. 包含自注意力层，捕捉输入序列内部的依赖关系
# 2. 包含前馈网络层，对每个位置的表示进行非线性变换
# 3. 使用残差连接和层归一化，加速训练收敛

# 解码器层类：解码器的基本组成单元，包含自注意力层、编码器-解码器注意力层和前馈网络层
class DecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, d_model, num_heads, dff, dropout):
        """
        初始化解码器层
        
        参数：
        - d_model: 模型维度
        - num_heads: 注意力头数
        - dff: 前馈网络隐藏层维度
        - dropout: dropout概率
        """
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)  # 自注意力层
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)  # 编码器-解码器注意力层
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=0.1)  # 前馈网络层
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)  # 层归一化1
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)  # 层归一化2
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)  # 层归一化3
        
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        """
        前向传播
        
        参数：
        - x: 解码器输入 [batch_size, seq_len, d_model]
        - memory: 编码器输出 [batch_size, seq_len, d_model]
        - look_ahead_mask: 前瞻掩码，防止关注未来位置
        - trg_padding_mask: 目标序列的填充掩码
        
        返回：
        - 解码后的张量 [batch_size, seq_len, d_model]
        """
        attn_output = self.self_mha(x, x, x, look_ahead_mask)  # 自注意力计算
        x = self.layernorm1(x + attn_output)  # 残差连接+层归一化
        
        src_output = self.src_mha(x, memory, memory, trg_padding_mask)  # 编码器-解码器注意力计算
        x = self.layernorm2(x + src_output)  # 残差连接+层归一化
        
        fnn_output = self.ffn(x)  # 前馈网络计算
        x = self.layernorm3(x + fnn_output)  # 残差连接+层归一化
        return x

# 解码器层类总结：
# 1. 包含自注意力层，捕捉输出序列内部的依赖关系
# 2. 包含编码器-解码器注意力层，关注编码器的输出
# 3. 包含前馈网络层，对每个位置的表示进行非线性变换
# 4. 使用残差连接和层归一化，加速训练收敛

# 编码器类：将输入文本转换为连续的语义表示
class Encoder(nn.Module):
    """编码器"""
    def __init__(self, num_layers, src_vocab_size, max_len, 
                 d_model, num_heads, dff, dropout=0.1):
        """
        初始化编码器
        
        参数：
        - num_layers: 编码器层数
        - src_vocab_size: 源词汇表大小
        - max_len: 最大序列长度
        - d_model: 模型维度
        - num_heads: 注意力头数
        - dff: 前馈网络隐藏层维度
        - dropout: dropout概率
        """
        super(Encoder, self).__init__()
        
        self.d_model = d_model  # 模型维度
        self.embedding = nn.Embedding(src_vocab_size, d_model)  # 词嵌入层
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)  # 位置编码层
        # 创建多层编码器层
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout) 
                                            for _ in range(num_layers)])
        
    def forward(self, x, src_mask):
        """
        前向传播
        
        参数：
        - x: 输入序列 [batch_size, seq_len]
        - src_mask: 源序列的填充掩码
        
        返回：
        - 编码后的语义表示 [batch_size, seq_len, d_model]
        """
        x = self.embedding(x) * math.sqrt(self.d_model)  # 词嵌入并缩放
        x = self.pos_encoding(x)  # 添加位置编码
        
        # 依次通过每个编码器层
        for enc_layer in self.enc_layers:
            x = enc_layer(x, src_mask)
        
        return x

# 编码器类总结：
# 1. 将输入文本通过词嵌入转换为向量表示
# 2. 添加位置编码，捕捉序列的顺序信息
# 3. 通过多层编码器层，逐步提取文本的语义信息
# 4. 输出语义表示，供后续处理

# 解码器类：将语义表示转换回文本
class Decoder(nn.Module):
    """解码器"""
    def __init__(self, num_layers, trg_vocab_size, max_len, 
                 d_model, num_heads, dff, dropout=0.1):
        """
        初始化解码器
        
        参数：
        - num_layers: 解码器层数
        - trg_vocab_size: 目标词汇表大小
        - max_len: 最大序列长度
        - d_model: 模型维度
        - num_heads: 注意力头数
        - dff: 前馈网络隐藏层维度
        - dropout: dropout概率
        """
        super(Decoder, self).__init__()
        
        self.d_model = d_model  # 模型维度
        self.embedding = nn.Embedding(trg_vocab_size, d_model)  # 词嵌入层
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)  # 位置编码层
        # 创建多层解码器层
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, dropout) 
                                            for _ in range(num_layers)])
    
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        """
        前向传播
        
        参数：
        - x: 解码器输入 [batch_size, seq_len]
        - memory: 编码器输出 [batch_size, seq_len, d_model]
        - look_ahead_mask: 前瞻掩码
        - trg_padding_mask: 目标序列的填充掩码
        
        返回：
        - 解码后的表示 [batch_size, seq_len, d_model]
        """
        x = self.embedding(x) * math.sqrt(self.d_model)  # 词嵌入并缩放
        x = self.pos_encoding(x)  # 添加位置编码
        
        # 依次通过每个解码器层
        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, look_ahead_mask, trg_padding_mask)
            
        return x

# 解码器类总结：
# 1. 将输入序列通过词嵌入转换为向量表示
# 2. 添加位置编码，捕捉序列的顺序信息
# 3. 通过多层解码器层，利用编码器的输出和自身的输入生成文本
# 4. 输出解码表示，供后续线性层映射到词汇表

# 信道解码器类：从受噪声干扰的信号中恢复语义表示
class ChannelDecoder(nn.Module):
    """信道解码器"""
    def __init__(self, in_features, size1, size2):
        """
        初始化信道解码器
        
        参数：
        - in_features: 输入特征维度（信道编码器的输出维度）
        - size1: 中间层维度（与模型维度d_model一致）
        - size2: 隐藏层维度
        """
        super(ChannelDecoder, self).__init__()
        
        self.linear1 = nn.Linear(in_features, size1)  # 第一层线性变换
        self.linear2 = nn.Linear(size1, size2)  # 第二层线性变换
        self.linear3 = nn.Linear(size2, size1)  # 第三层线性变换
        self.layernorm = nn.LayerNorm(size1, eps=1e-6)  # 层归一化
        
    def forward(self, x):
        """
        前向传播
        
        参数：
        - x: 受噪声干扰的信号 [batch_size, seq_len, in_features]
        
        返回：
        - 恢复的语义表示 [batch_size, seq_len, size1]
        """
        x1 = self.linear1(x)  # 线性变换到size1维度
        x2 = F.relu(x1)  # 非线性激活
        x3 = self.linear2(x2)  # 线性变换到size2维度
        x4 = F.relu(x3)  # 非线性激活
        x5 = self.linear3(x4)  # 线性变换回size1维度
        
        output = self.layernorm(x1 + x5)  # 残差连接+层归一化

        return output

# 信道解码器类总结：
# 1. 从受噪声干扰的信号中恢复语义表示
# 2. 使用多层线性变换和非线性激活，增强恢复能力
# 3. 使用残差连接和层归一化，加速训练收敛

# DeepSC模型类：集成了编码器、信道编码器、信道解码器和解码器
class DeepSC(nn.Module):
    """DeepSC模型"""
    def __init__(self, num_layers, src_vocab_size, trg_vocab_size, src_max_len, 
                 trg_max_len, d_model, num_heads, dff, dropout=0.1):
        """
        初始化DeepSC模型
        
        参数：
        - num_layers: 编码器和解码器的层数
        - src_vocab_size: 源词汇表大小
        - trg_vocab_size: 目标词汇表大小
        - src_max_len: 源序列最大长度
        - trg_max_len: 目标序列最大长度
        - d_model: 模型维度
        - num_heads: 注意力头数
        - dff: 前馈网络隐藏层维度
        - dropout: dropout概率
        """
        super(DeepSC, self).__init__()
        
        # 初始化编码器
        self.encoder = Encoder(num_layers, src_vocab_size, src_max_len, 
                               d_model, num_heads, dff, dropout)
        
        # 初始化信道编码器：将语义表示映射到低维向量
        self.channel_encoder = nn.Sequential(
            nn.Linear(d_model, 256),  # 线性变换到256维度
            nn.ReLU(inplace=True),  # 非线性激活
            nn.Linear(256, 16)  # 线性变换到16维度
        )

        # 初始化信道解码器：从受噪声干扰的信号中恢复语义表示
        self.channel_decoder = ChannelDecoder(16, d_model, 512)
        
        # 初始化解码器
        self.decoder = Decoder(num_layers, trg_vocab_size, trg_max_len, 
                               d_model, num_heads, dff, dropout)
        
        # 初始化输出线性层：将解码表示映射到词汇表
        self.dense = nn.Linear(d_model, trg_vocab_size)

# DeepSC模型类总结：
# 1. 集成了完整的语义通信系统，包括编码器、信道编码器、信道解码器和解码器
# 2. 编码器将输入文本转换为语义表示
# 3. 信道编码器将语义表示压缩为低维向量，适合信道传输
# 4. 信道解码器从受噪声干扰的信号中恢复语义表示
# 5. 解码器将语义表示转换回文本
# 6. 输出线性层将解码表示映射到词汇表，生成最终输出

# DeepSC模型工作流程：
# 1. 输入文本 → 词嵌入 → 位置编码 → 编码器 → 语义表示
# 2. 语义表示 → 信道编码器 → 低维向量 → 功率归一化 → 信道传输（加噪声）
# 3. 受噪声干扰的信号 → 信道解码器 → 恢复的语义表示
# 4. 恢复的语义表示 → 解码器 → 解码表示 → 线性层 → 词汇表分布 → 输出文本