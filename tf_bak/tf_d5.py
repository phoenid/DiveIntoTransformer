# encoder,主要组成部分是输入的嵌入，encoder_block的组成，最后得到多个编码器组合的输出z

"""
1. 在Encoder中，init中首先定义encoder block的数量，然后对位置编码实例化，最后定义encoder block list
2. 在forward中，首先根据x与PAD_IDX比对，与PAD_IDX相等的置为true，其余置为false，构造mask
3. x经过位置编码后，经过encoder block list中的所有encoder block，输出output
4. 在main函数中，在train dataset中抽出两个德语句子，经过分词和编码后使用padding对齐组成batch放入encoder中
"""

'''
# Part1 进行库的导入

'''
import torch
from torch import nn
from tf_d2 import EmbeddingWithPosition
from tf_d1 import de_vocab,train_dataset,de_preprocess,PAD_IDX
from tf_d4 import EncoderBlock

'''
# Part2 定义编码器的这个类
'''
class Encoder(nn.Module):
    def __init__(self,vocab_size,emd_size,head,q_k_size,v_size,f_size,nums_encoderblock=5):
        super().__init__()
        self.nums_encoderblock=nums_encoderblock
        # 定义编码器
        self.emd=EmbeddingWithPosition(vocab_size=vocab_size,emd_size=emd_size)
        # encoder block
        self.encoder_block_list=[]
        for i in range(nums_encoderblock):
            self.encoder_block_list.append(EncoderBlock(emd_size=emd_size,f_size=f_size,head=head,v_size=v_size,q_k_size=q_k_size))

    def forward(self,x): #输出是原始的没编码过的list，不定长，(batch_size,q_seq_len)都形成不了矩阵？
        # 前提此时的x已经是PAD过的矩阵。为(batch_size,q_seq_len)

        mask_pad=(x==PAD_IDX).unsqueeze(1) # (batch_size,1,q_seq_len)
        mask_pad=mask_pad.expand(-1,x.size()[1],-1) # (batch_size,q_seq_len,q_seq_len)
        # mask_pad=mask_pad
        # 进行编码
        x = self.emd(x) # (batch_size,seq_len,emd_size)
        output=x # (batch_size,seq_len,emd_size)
        for i in range(self.nums_encoderblock):
            output=self.encoder_block_list[i](output,mask_pad) # (batch_size,seq_len,emd_size)
        return output


if __name__ == '__main__':
    # 取2个de句子转词ID序列
    de_tokens1, de_ids1 = de_preprocess(train_dataset[0][0])
    de_tokens2, de_ids2 = de_preprocess(train_dataset[1][0])

    # 组成batch并padding对齐
    if len(de_ids1) < len(de_ids2):
        de_ids1.extend([PAD_IDX] * (len(de_ids2) - len(de_ids1)))
    elif len(de_ids1) > len(de_ids2):
        de_ids2.extend([PAD_IDX] * (len(de_ids1) - len(de_ids2)))

    batch = torch.tensor([de_ids1, de_ids2], dtype=torch.long)
    print('batch:', batch.size())  # (2, de_ids2.size())

    # Encoder编码
    encoder = Encoder(vocab_size=len(de_vocab), emd_size=128, q_k_size=256, v_size=512, f_size=512, head=8, nums_encoderblock=3)
    z = encoder.forward(batch)
    print('encoder outputs:', z.size())
