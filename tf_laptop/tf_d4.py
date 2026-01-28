# 开始写编码器小模块，主要是编码，多头注意力，残差和归一化，线形层，残差残差和归一化。
# 需要注意的是，多头输出后，先将其转化为emd，也就是输入时候的大小，所以需要一个线性层

"""
1. 首先继承nn.Module
2. 利用MultiHeadAttention构建多头注意力机制，然后定义Linear层、LayerNorm层、feedforward层
ps：LayerNorm称为层归一化，适用于NLP。归一化分为两种，另一种为BatchNorm，适用于CV。两者的区别在于BatchNorm是对于Channel个维度求均值和方差，LayerNorm则是对每一个样本求均值和方差。公式参考概率论中的0-1标准正态分布。
ref: https://www.bilibili.com/video/BV1hqpjzrEmT、https://www.bilibili.com/video/BV14K411m7Pf
3. 在forward实例方法中，根据transformer模型中Encoder的架构图，先通过muli_atten，然后通过Linear层，同时将该tensor z1与原先的tensor x相加后进行层归一化。随后通过feedforward层，变为z2，最后再将该z2与z1相加后进行层归一化。
4. 在main函数中，emb_result经过5次EncoderBlock再输出
"""

'''
Part1引入相关的库
'''
import torch
from torch import nn
from tf_d1_5090 import de_vocab,de_preprocess,train_dataset
from tf_d2 import EmbeddingWithPosition
from tf_d3 import MultiHeadAttention
import math

# from tf_config import DEVICE
# torch.set_default_device(DEVICE)

class EncoderBlock(nn.Module):
    def __init__(self,emd_size,f_size,head,v_size,q_k_size,dropout_rate):
        super().__init__()

        # 第一个定义多头注意力机制
        self.muli_atten=MultiHeadAttention(head=head,emd_size=emd_size,q_k_size=q_k_size,v_size=v_size,dropout_rate=dropout_rate)

        # 第二个定义线性层转化为emd_size

        self.Wz=nn.Linear(head*v_size,emd_size)
        # 归一化(需要输入维度)
        self.norm1=nn.LayerNorm(emd_size)
        # 最终还是输入的大小,所以需要一个中间维度f_size,最终还是(batch_size,q_seq_len,emd_size)
        self.feedforward=nn.Sequential(
            nn.Linear(emd_size,f_size),
            nn.ReLU(),
            nn.Linear(f_size,emd_size),
        )
        # 归一化
        self.norm2 = nn.LayerNorm(emd_size)

    def forward(self,x,mask_pad): # (batch_size,q_seq_len,emd)
        # 多头
        z=self.muli_atten(x_q=x,x_k_v=x,mask_pad=mask_pad) # (batch_size,q_seq_len,head*v_size)

        # 回大小
        z=self.Wz(z) # (batch_size,q_seq_len,emd)
        # 残差和归一化(主要是最后一层，所以初始化输入的大小为emd)
        output1=self.norm1(z+x) # (batch_size,q_seq_len,emd)

        # 前向
        z=self.feedforward(output1) # (batch_size,q_seq_len,emd)
        # 残差
        output2=self.norm2(z+output1) # (batch_size,q_seq_len,emd)
        return output2


if __name__ == '__main__':
    # 准备1个batch
    emb = EmbeddingWithPosition(len(de_vocab), 128)
    de_tokens, de_ids = de_preprocess(train_dataset[0][0])  # 取de句子转词ID序列
    de_ids_tensor = torch.tensor(de_ids, dtype=torch.long)
    emb_result = emb(de_ids_tensor.unsqueeze(0))  # 转batch再输入模型
    print('emb_result:', emb_result.size())

    attn_mask = torch.zeros((1, de_ids_tensor.size()[0], de_ids_tensor.size()[0]))  # batch中每个样本对应1个注意力矩阵

    # 用于module初始化嵌套
    encoder_list = []
    for i in range(5):
        encoder_list.append(EncoderBlock(emd_size=128, f_size=256, head=8, v_size=512, q_k_size=256,dropout_rate=0.1))

    # forward输出
    output = emb_result
    for i in range(5):
        output = encoder_list[i](output, attn_mask)
    print('encoder_outputs:', output.size())
