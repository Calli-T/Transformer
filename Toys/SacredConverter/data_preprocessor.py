from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

SRC_EDITION = "AGAPE_EASY" # 이거 번역 원문 못찾음
TGT_EDITION = "NKRV"
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

# 소스나 타겟이나 언어가 같다
token_transform = get_tokenizer("spacy", language="ko_core_news_sm")


'''
def generate_tokens(text_iter, edition):
    edition_index = {SRC_EDITION: 0, TGT_EDITION: 1}

    for text in text_iter:
        yield token_transform(text[language_index[language]])
'''