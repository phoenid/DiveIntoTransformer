# 该模块主要是transformer模块，主要由编码器和解码器组成

"""
1. 首先在main函数中，生成送入模型的batch，包括德语句子和英语句子的token以及id，并组成batch、padding对齐
2. 然后将上述batch送入transformer中。在init部分，定义encoder和decoder；feedforward中，将encoder_x先通过encoder，再通过decoder即可
3. 除此之外，定义encode和decode实例方法（就是类Encoder的实例对象），用于后续使用
"""

'''
# Part1,库函数的引入
'''
import torch
from torch import nn
from tf_d5 import Encoder
from tf_d7 import Decoder
from tf_d1 import en_preprocess,de_preprocess,train_dataset,en_vocab,de_vocab,PAD_IDX

'''
# Part2 定义一个类，实现先编码后面解码
'''
class Transformer(nn.Module):
    def __init__(self,de_vocab_size,emd_size,head,q_k_size,v_size,f_size,en_vocab_size ,nums_encoder_block=6,nums_decoder_block=6,dropout=0.1, seq_max_len=5000):
        super().__init__()

        # 编码器
        self.encoder=Encoder(vocab_size=de_vocab_size,emd_size=emd_size,head=head,q_k_size=q_k_size,v_size=v_size,f_size=f_size,nums_encoderblock=nums_encoder_block,dropout_rate=dropout, seq_max_len=seq_max_len)
        self.decoder=Decoder(en_vocab_size=en_vocab_size, emd_size=emd_size, nums_decoder_block=nums_decoder_block, head=head, q_k_size=q_k_size, v_size=v_size, f_size=f_size,dropout_rate=dropout, seq_max_len=seq_max_len)

    def forward(self,encoder_x,decoder_x):
        encoder_z=self.encoder(encoder_x)
        ouput=self.decoder(x=decoder_x, encoder_z=encoder_z,encoder_x=encoder_x)
        return ouput #(batch_size,q_seq_len,en_vocab_len)

    def encode(self,encoder_x):
        return self.encoder(encoder_x)

    def decode(self,decoder_x):
        return self.decoder(decoder_x)


# 测试

if __name__ == '__main__':
    transformer = Transformer(en_vocab_size=len(en_vocab), de_vocab_size=len(de_vocab), emd_size=128, q_k_size=256,
                              v_size=512, f_size=512, head=8, nums_encoder_block=3, nums_decoder_block=3, dropout=0.1, seq_max_len=5000)

    # 取2个de句子转词ID序列，输入给encoder
    de_tokens1, de_ids1 = de_preprocess(train_dataset[0][0])
    de_tokens2, de_ids2 = de_preprocess(train_dataset[1][0])
    # 对应2个en句子转词ID序列，再做embedding，输入给decoder
    en_tokens1, en_ids1 = en_preprocess(train_dataset[0][1])
    en_tokens2, en_ids2 = en_preprocess(train_dataset[1][1])

    # de句子组成batch并padding对齐
    if len(de_ids1) < len(de_ids2):
        de_ids1.extend([PAD_IDX] * (len(de_ids2) - len(de_ids1)))
    elif len(de_ids1) > len(de_ids2):
        de_ids2.extend([PAD_IDX] * (len(de_ids1) - len(de_ids2)))

    enc_x_batch = torch.tensor([de_ids1, de_ids2], dtype=torch.long)
    print('enc_x_batch batch:', enc_x_batch.size())

    # en句子组成batch并padding对齐
    if len(en_ids1) < len(en_ids2):
        en_ids1.extend([PAD_IDX] * (len(en_ids2) - len(en_ids1)))
    elif len(en_ids1) > len(en_ids2):
        en_ids2.extend([PAD_IDX] * (len(en_ids1) - len(en_ids2)))

    dec_x_batch = torch.tensor([en_ids1, en_ids2], dtype=torch.long)
    print('dec_x_batch batch:', dec_x_batch.size())

    # 输出每个en词的下一个词概率
    decoder_z = transformer(enc_x_batch, dec_x_batch)
    print('decoder outputs:', decoder_z.size())
