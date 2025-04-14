#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class CNN(nn.Module):
    def __init__(self, word_vec, class_num, config):
        super().__init__()
        self.word_vec = word_vec
        self.class_num = class_num

        # hyper parameters and others
        self.max_len = config.max_len
        self.word_dim = config.word_dim
        self.pos_dim = config.pos_dim
        self.pos_dis = config.pos_dis

        self.dropout_value = config.dropout
        self.filter_num = config.filter_num
        self.window = config.window
        self.hidden_size = config.hidden_size

        self.dim = self.word_dim + 2 * self.pos_dim

        self.use_pf = False
        
        if self.use_pf:
            self.dim = self.word_dim + 2 * self.pos_dim
        else:
            self.dim = self.word_dim

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        if self.use_pf:
            self.pos1_embedding = nn.Embedding(
                num_embeddings=2 * self.pos_dis + 3,
                embedding_dim=self.pos_dim
            )
            self.pos2_embedding = nn.Embedding(
                num_embeddings=2 * self.pos_dis + 3,
                embedding_dim=self.pos_dim
            )

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.filter_num,
            kernel_size=(self.window, self.dim),
            stride=(1, 1),
            bias=True,
            padding=(1, 0),  # same padding
            padding_mode='zeros'
        )
        self.maxpool = nn.MaxPool2d((self.max_len, 1))
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(self.dropout_value)
        self.linear = nn.Linear(
            in_features=self.filter_num,
            out_features=self.hidden_size,
            bias=True
        )
        self.dense = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.class_num,
            bias=True
        )

        # initialize weight
        if self.use_pf:
            init.xavier_normal_(self.pos1_embedding.weight)
            init.xavier_normal_(self.pos2_embedding.weight)
        init.xavier_normal_(self.conv.weight)
        init.constant_(self.conv.bias, 0.)
        init.xavier_normal_(self.linear.weight)
        init.constant_(self.linear.bias, 0.)
        init.xavier_normal_(self.dense.weight)
        init.constant_(self.dense.bias, 0.)

    def encoder_layer(self, token, pos1, pos2):
        word_emb = self.word_embedding(token)  # B*L*word_dim
        if self.use_pf and pos1 is not None and pos2 is not None:
            # 得到位置嵌入：形状 [B, L, pos_dim]
            pos1_emb = self.pos1_embedding(pos1)
            pos2_emb = self.pos2_embedding(pos2)
            # 拼接为最终表示：[word_emb, pos1_emb, pos2_emb]，形状 [B, L, word_dim+2*pos_dim]
            emb = torch.cat(tensors=[word_emb, pos1_emb, pos2_emb], dim=-1)
        else:
            # 如果不使用 PF，只保留词嵌入
            emb = word_emb
        return emb  # B*L*D, D=word_dim+2*pos_dim

    def conv_layer(self, emb, mask):
        emb = emb.unsqueeze(dim=1)  # B*1*L*D
        conv = self.conv(emb)  # B*C*L*1

        # mask, remove the effect of 'PAD'
        conv = conv.view(-1, self.filter_num, self.max_len)  # B*C*L
        mask = mask.unsqueeze(dim=1)  # B*1*L
        mask = mask.expand(-1, self.filter_num, -1)  # B*C*L
        conv = conv.masked_fill_(mask.eq(0), float('-inf'))  # B*C*L
        conv = conv.unsqueeze(dim=-1)  # B*C*L*1
        return conv

    def single_maxpool_layer(self, conv):
        pool = self.maxpool(conv)  # B*C*1*1
        pool = pool.view(-1, self.filter_num)  # B*C
        return pool

    def forward(self, data):
        token = data[:, 0, :].view(-1, self.max_len)
        if self.use_pf:
            pos1 = data[:, 1, :].view(-1, self.max_len)   # [B, max_len]
            pos2 = data[:, 2, :].view(-1, self.max_len)   # [B, max_len]
        else:
            pos1, pos2 = None, None
        mask = data[:, 3, :].view(-1, self.max_len)
        #todo 根据原论文补充
        # 1. 词表示层：将 token 与位置向量拼接得到句子中每个词的向量表示
        emb = self.encoder_layer(token, pos1, pos2)  # [B, L, D]，其中 D=word_dim+2*pos_dim

        # 2. 卷积层：先扩展维度，再通过卷积操作并应用掩码消除 pad 影响
        conv_out = self.conv_layer(emb, mask)  # [B, filter_num, L, 1]

        # 3. 最大池化层：对每个滤波器在所有时间步上取最大值，得到固定长度的特征向量
        pool = self.single_maxpool_layer(conv_out)  # [B, filter_num]

        # 4. 非线性映射：全连接层加 tanh 激活，将池化特征映射到隐藏层 (即句子层特征)
        sentence_feature = self.tanh(self.linear(pool))  # [B, hidden_size]
        sentence_feature = self.dropout(sentence_feature)
        logits = self.dense(sentence_feature)
        return logits
