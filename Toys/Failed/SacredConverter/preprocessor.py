from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from verse_iterator import VerseIterator

# -------상수--------------------------------------------------------

SRC_EDITION = "AGAPE_EASY"  # 이거 번역 원문 못찾음
TGT_EDITION = "NKRV"
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]
BATCH_SIZE = 16 # 데이터가 16 * 29 * 67 사이즈라 이게 2^n인 최대

# -----------------------------------------------------------------------

# 소스나 타겟이나 언어가 같다
token_transform = get_tokenizer("spacy", language="ko_core_news_sm")


def generate_tokens(text_iter, edition):
    edition_index = {SRC_EDITION: 0, TGT_EDITION: 1}

    for text in text_iter:
        yield token_transform(text[edition_index[edition]])


vocab_transform = {}
for edition in [SRC_EDITION, TGT_EDITION]:
    train_iter = VerseIterator(split="train")
    vocab_transform[edition] = build_vocab_from_iterator(
        generate_tokens(train_iter, edition),
        min_freq=1,
        specials=special_symbols,
        special_first=True,
    )

for edition in [SRC_EDITION, TGT_EDITION]:
    vocab_transform[edition].set_default_index(UNK_IDX)

'''print("Vocab Transform:")
print(vocab_transform)'''


def input_transform(token_ids):
    return torch.cat(
        (torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX]))
    )

# ---------위로 전처리 함수 셋, 아래로 전처리 파이프 제작-----------------------------------------------------------
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


text_transform = {}
for edition in [SRC_EDITION, TGT_EDITION]:
    text_transform[edition] = sequential_transforms(
        token_transform, vocab_transform[edition], input_transform
    )

# --------------------------------------------------------------------
def collator(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_EDITION](src_sample))
        tgt_batch.append(text_transform[TGT_EDITION](tgt_sample))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)

    return src_batch, tgt_batch

'''data_iter = VerseIterator(split="valid")
dataloader = DataLoader(data_iter, batch_size=BATCH_SIZE, collate_fn=collator)
source_tensor, target_tensor = next(iter(dataloader))'''

'''
print("(source, target):")
print(next(iter(data_iter)))

print("source_batch:", source_tensor.shape)
print(source_tensor)

print("target_batch:", target_tensor.shape)
print(target_tensor)
'''

# 2
