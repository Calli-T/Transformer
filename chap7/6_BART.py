import numpy as np
from datasets import load_dataset

# ---

import torch
from transformers import BartTokenizer
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_directml import device

# ---

from torch import optim
from transformers import BartForConditionalGeneration

# - get dataset -

news = load_dataset("argilla/news-summary", split="test")
df = news.to_pandas().sample(5000, random_state=42)[["text", "prediction"]]
df["prediction"] = df["prediction"].map(lambda x: x[0]["text"])
train, valid, test = np.split(
    df.sample(frac=1, random_state=42), [int(0.6 * len(df)), int(0.8 * len(df))]  # 이거 나눈 패러미터 기준이 대체 뭔?
)

'''
print(f"Source News : {train.text.iloc[0][:200]}")
print(f"Summarization : {train.prediction.iloc[0][:50]}")
print(f"Training Data Size : {len(train)}")
print(f"Validation Data Size : {len(valid)}")
print(f"Testing Data Size : {len(test)}")
'''


# - input tensor preprocessing -

# BART 토크나이저 사용
def make_dataset(data, tokenizer, device):
    tokenized = tokenizer(
        text=data.text.tolist(),
        padding="longest",
        truncation=True,
        return_tensors='pt'
    )
    labels = []
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    for target in data.prediction:
        labels.append(tokenizer.encode(target, return_tensors='pt').squeeze())
    labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)

    return TensorDataset(input_ids, attention_mask, labels)


def get_dataloader(dataset, sampler, batch_size):
    data_sampler = sampler(dataset)
    dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=batch_size)

    return dataloader


device = device()
tokenizer = BartTokenizer.from_pretrained(pretrained_model_name_or_path="facebook/bart-base")
batch_size = 8
epochs = 3

train_dataset = make_dataset(train, tokenizer, device)
train_dataloader = get_dataloader(train_dataset, RandomSampler, batch_size)

valid_dataset = make_dataset(valid, tokenizer, device)
valid_dataloader = get_dataloader(valid_dataset, RandomSampler, batch_size)

test_dataset = make_dataset(test, tokenizer, device)
test_dataloader = get_dataloader(test_dataset, RandomSampler, batch_size)

# print(train_dataset[0])

# - model declaration -

model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path="facebook/bart-base").to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8)

'''
for main_name, main_module in model.named_children():
    print(main_name)
    for sub_name, sub_module in main_module.named_children():
        print("└", sub_name)
        for ssub_name, ssub_module in sub_module.named_children():
            print("└└", ssub_name)
            for sssub_name, sssub_module in ssub_module.named_children():
                print("└└└", sssub_name)
'''