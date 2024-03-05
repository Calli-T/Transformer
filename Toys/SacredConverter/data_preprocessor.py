from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from verse_iterator import VerseIterator

SRC_EDITION = "AGAPE_EASY" # 이거 번역 원문 못찾음
TGT_EDITION = "NKRV"
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

# 소스나 타겟이나 언어가 같다
token_transform = get_tokenizer("spacy", language="ko_core_news_sm")


def generate_tokens(text_iter, edition):
    edition_index = {SRC_EDITION: 0, TGT_EDITION: 1}

    for text in text_iter:
        yield token_transform(text[edition_index[edition]])

vocab_transform = {}
for edition in [SRC_EDITION, TGT_EDITION]:
    train_iter = VerseIterator(split="valid")
    vocab_transform[edition] = build_vocab_from_iterator(
        generate_tokens(train_iter, edition),
        min_freq=1,
        specials=special_symbols,
        special_first=True,
    )

for edition in [SRC_EDITION, TGT_EDITION]:
    vocab_transform[edition].set_default_index(UNK_IDX)

print("Vocab Transform:")
print(vocab_transform)