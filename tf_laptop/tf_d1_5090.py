# 手撕transformer下载数据集（去 torchtext 版）
import re
import tarfile
import io
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional

# ====== tokenizer: 替代 get_tokenizer('basic_english') ======
def tokenizer_keep_dot(s: str):
    pattern = r"""
        \d+(?:[”"“»])?-+[^\W\d_]+(?:-[^\W\d_]+)* |  # 08“-Schild 这一类：数字 + (右引号可选) + 连字符 + 词(可带内部连字符)
        [„"“']+[^\W\d_]+                          |  # „Obama 这一类：左引号开头 + 词
        [^\W\d_]+(?:-[^\W\d_]+)*                  |  # 普通单词：允许内部连字符
        \d+                                       |  # 纯数字
        [.,!?;:()\[\]{}]                             # 标点单独成 token
    """
    s = s.lower()
    return [s for s in re.findall(pattern, s, flags=re.VERBOSE | re.UNICODE)]

en_tokenizer = tokenizer_keep_dot
de_tokenizer = tokenizer_keep_dot  # 你原来也是这样用的 :contentReference[oaicite:2]{index=2}

# ====== vocab: 替代 build_vocab_from_iterator ======
# dataclass自己生成init方法
# https://www.bilibili.com/video/BV15zSVYtE13
@dataclass
class Vocab:
    stoi: Dict[str, int]  # {单词: 对应itos中的索引, 单词: 对应itos中的索引, ...}
    itos: List[str]  # [按序号存入的单词列表]
    default_index: int = 0

    def set_default_index(self, idx: int):
        self.default_index = idx

    def __len__(self):
        return len(self.itos)

    def __call__(self, tokens: List[str]) -> List[int]:
        di = self.default_index
        return [self.stoi.get(t, di) for t in tokens]
        # tokens是['<bos>', 'zwei', 'junge', 'weiße', 'männer', 'sind', 'im', 'freien', 'in', 'der', 'nähe', 'vieler', 'büsche', '.', '<eos>']
        # stoi为{'单词': 对应itos中的索引, '单词': 对应itos中的索引, ...}


def build_vocab_from_iterator(
    token_lists: Iterable[List[str]],
    specials: Optional[List[str]] = None,
    special_first: bool = True,
    min_freq: int = 1,
) -> Vocab:
    counter = Counter()
    for toks in token_lists:  # token_lists为[[一句话分词后的], ['str', 'str', ...], ...]，没有加<bos>
        # 累计统计toks中出现的字符个数，生成字典
        # https://www.bilibili.com/video/BV12M4y1m7QH
        counter.update(toks)

    specials = specials or []  # '<unk>', '<pad>', '<bos>', '<eos>'
    # 先放 specials
    """
    如果 specials 是“假值”(falsy)（比如 None、空列表 []、空字符串 ""、0 等），就把它设成一个新的空列表 []
    否则（specials 里已经有内容），就保持原来的 specials
    """
    # 定义一个变量 itos，它的类型期望是 List[str]（字符串列表），并且初始值是空列表 []
    itos: List[str] = []
    seen = set()  # 去重用
    def add(tok: str):
        # 在内层函数里声明“itos 和 seen 不是本地变量，而是来自外层函数作用域的变量”，并且我要修改它们。
        nonlocal itos, seen
        if tok not in seen:
            itos.append(tok)
            seen.add(tok)  # 去重用

    if special_first:
        for sp in specials:
            add(sp)

    # 再放普通词（按频率降序，再按字典序稳定一下）
    # 把 counter 里的 (词, 频次) 按“频次从高到低”排序；如果频次相同，再按“词的字典序从小到大”排序，然后依次遍历。
    # https://www.bilibili.com/video/BV1Jgf6YvE8e/?p=36
    for tok, freq in sorted(counter.items(), key=lambda x: (-x[1], x[0])):  # (词, 频次)
        if freq >= min_freq:
            add(tok)

    if not special_first:
        for sp in specials:
            add(sp)

    stoi = {tok: i for i, tok in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)

# ====== dataset: 替代 torchtext.datasets.Multi30k ======
TRAIN_URL = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
VALID_URL = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

def _download_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url) as r:
        return r.read()

def _read_tar_text(tar_bytes: bytes, member_name: str) -> List[str]:
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        m = tf.getmember(member_name)
        f = tf.extractfile(m)
        assert f is not None
        return f.read().decode("utf-8").splitlines()

def load_multi30k_train(language_pair=("de", "en")) -> List[Tuple[str, str]]:
    # small repo 的 tar 里一般是 train.de / train.en（如果名字不一致，下面会报 KeyError，告诉我我再适配）
    tar_bytes = _download_bytes(TRAIN_URL)
    src, tgt = language_pair
    src_lines = _read_tar_text(tar_bytes, f"train.{src}")
    tgt_lines = _read_tar_text(tar_bytes, f"train.{tgt}")
    return list(zip(src_lines, tgt_lines))

# ====== specials（你原来这段 그대로）=====
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM = "<unk>", "<pad>", "<bos>", "<eos>"

# ====== load train_dataset（替代 train_dataset=list(Multi30k(...))）=====
train_dataset = load_multi30k_train(language_pair=("de", "en"))

# ====== build vocab（对应你原来的 Part5/6）=====
de_tokens, en_tokens = [], []
for de, en in train_dataset:
    de_tokens.append(de_tokenizer(de))
    en_tokens.append(en_tokenizer(en))

# de_vocab是Vocab类的实例对象
de_vocab = build_vocab_from_iterator(
    de_tokens, specials=[UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM], special_first=True
)
de_vocab.set_default_index(UNK_IDX)

en_vocab = build_vocab_from_iterator(
    en_tokens, specials=[UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM], special_first=True
)
en_vocab.set_default_index(UNK_IDX)

def de_preprocess(de_sentence: str):
    tokens = de_tokenizer(de_sentence)
    tokens = [BOS_SYM] + tokens + [EOS_SYM]
    ids = de_vocab(tokens)
    return tokens, ids

def en_preprocess(en_sentence: str):
    tokens = en_tokenizer(en_sentence)
    tokens = [BOS_SYM] + tokens + [EOS_SYM]
    ids = en_vocab(tokens)
    return tokens, ids

if __name__ == "__main__":
    print("de vocab:", len(de_vocab))
    print("en vocab:", len(en_vocab))
    de_sentence, en_sentence = train_dataset[0]
    print("de preprocess:", *de_preprocess(de_sentence))
    print("en preprocess:", *en_preprocess(en_sentence))
