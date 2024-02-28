from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from model_architecture import *


def sequential_transform(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def input_transform(token_ids):
    return torch.cat(  # list를 줘야하나?
        ([torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])])
    )


# 세 개의 transform은 각각 문장->토큰 / 토큰->인덱스 / 인덱스->입력(시작과 끝 토큰 추가)를 위한것
text_transform = {}
for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[language] = sequential_transform(
        token_transform[language], vocab_transform[language], input_transform
    )


def collator(batch):
    src_batch, tgt_batch = [], []
    # 문장 -> 띄워쓰기 제거된 문장 -> 토큰 -> 인덱스 -> BOS와 EOS가 추가된 인덱스
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[SRC_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    
    # 손질다하면 그대로 소스와 타깃의 배치로
    return src_batch, tgt_batch


#
'''
반복자를 데이터로더에 집어넣어 데이터로더 정의
collate_fn은 batch로 묶어주는 내부함수로
사용자 정의로 만든것으로 대체하며
전처리를 진행한다
'''
data_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
dataloader = DataLoader(data_iter, batch_size=BATCH_SIZE, collate_fn=collator)
source_tensor, target_tensor = next(iter(dataloader))


print("- Dataloader sample -")
print("(source, target): ")
print(next(iter(data_iter)))

print("source_batch: ", source_tensor.shape)
print(source_tensor)

print("target_batch: ", target_tensor.shape)
print(target_tensor)
print()

# 4