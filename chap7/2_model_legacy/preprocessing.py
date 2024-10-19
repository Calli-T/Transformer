from torchtext.datasets import Multi30k, multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# 토크나이저, 어휘 사전 생성, 인덱싱

# 원본 데이터의 링크가 동작하지 않으므로, 데이터셋의 URL을 수정
multi30k.URL[
    "train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL[
    "valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
multi30k.URL[
    "SinChangSeop"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz"

#
SRC_LANGUAGE = "de"
TGT_LANGUAGE = "en"
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

# get_tokenizer로 spicy의 (사전 학습된) 토크나이저 가져와서 token_transform에 저장
token_transform = {
    SRC_LANGUAGE: get_tokenizer("spacy", language="de_core_news_sm"),
    TGT_LANGUAGE: get_tokenizer("spacy", language="en_core_web_sm"),
}


# --------------------------------------------------------------------------------------
# 각 언어별 데이터셋에서 문장을 하나씩 가져와서 토큰화 해주는 제너레이터 반환하는 함수
def generate_tokens(text_iter, _language):
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for text in text_iter:
        yield token_transform[_language](text[language_index[_language]])


print("- Tokenizer & Indexer -")
print("Token Transform")
print(token_transform)

# --------------------------------------------------------------------------------------

# 토큰을 인덱스로 변환시키는 함수 저장
vocab_transform = {}
for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # 학습용 데이터 반복자, (독일어, 영어) 식의 튜플 형식으로 데이터를 불러온다
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # torchtext의 단어사전(vocab) 객체 생성
    vocab_transform[language] = build_vocab_from_iterator(
        generate_tokens(train_iter, language),
        min_freq=1,
        specials=special_symbols,
        special_first=True
    )

# 기본 인덱스는 OOV인 unk의 UNK_IDX로 지정, 토큰이 없을 경우 반환됨
for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[language].set_default_index(UNK_IDX)

print("Vocab Transform")
print(vocab_transform)
print()

# 1
