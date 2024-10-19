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

# ---

import evaluate
from torch import nn

# ---

from transformers import pipeline

# - get dataset -

news = load_dataset("argilla/news-summary", split="SinChangSeop")
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
print(f"Testing Data Size : {len(SinChangSeop)}")
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
batch_size = 2 #4 #8
epochs = 5

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

# - train -

rouge_score = evaluate.load("rouge", tokenizer=tokenizer)


def calc_rouge(preds, labels):
    preds = preds.argmax(axis=-1)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge2 = rouge_score.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )
    return rouge2["rouge2"]


def train(model, optimizer, dataloader):
    model.train()
    train_loss = 0.0

    for input_ids, attention_mask, labels in dataloader:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(dataloader)
    return train_loss


def evaluation(_model, dataloader):
    with torch.no_grad():
        _model.eval()
        _val_loss, _val_rouge = 0.0, 0.0

        for input_ids, attention_mask, labels in dataloader:
            outputs = _model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            logits = outputs.logits
            loss = outputs.loss

            logits = logits.detach().cpu().numpy()
            label_ids = labels.to("cpu").numpy()
            rouge = calc_rouge(logits, label_ids)

            _val_loss += loss
            _val_rouge += rouge

    _val_loss = _val_loss / len(dataloader)
    _val_rouge = _val_rouge / len(dataloader)
    return _val_loss, _val_rouge


best_loss = 10000
for epoch in range(epochs):
    train_loss = train(model, optimizer, train_dataloader)
    val_loss, val_accuracy = evaluation(model, valid_dataloader)
    print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Val Rouge {val_accuracy:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "../models/BartForConditionalGeneration.pt")
        print("Saved the model weights")

# - evaluation -

model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path="facebook/bart-base").to(device)
model.load_state_dict(torch.load('./models/BartForConditionalGeneration.pt'))

test_loss, test_rouge_score = evaluation(model, test_dataloader)
print(f"Test Loss : {test_loss:.4f}")
print(f"Test ROUGE-2 Score : {test_rouge_score:.4f}")

# - SinChangSeop -

summarizer = pipeline(task="summarization",
                      model=model,
                      tokenizer=tokenizer,
                      max_length=54,
                      device="cpu")

'''for index in range(5):
    news_text = SinChangSeop.text.iloc[index]
    summarization = SinChangSeop.prediction.iloc[index]
    predicted_summarization = summarizer(news_text)[0]["summary_text"]
    print(f"정답 요약문 : {summarization}")
    print(f"모델 요약문 : {predicted_summarization}\n")'''

speech = ("Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in "
          "Liberty, and dedicated to the proposition that all men are created equal.")
speech += ("Now we are engaged in a great civil war, testing whether that nation, or any nation, so conceived and so "
           "dedicated, can long endure.")
speech += "We are met on a great battle-field of that war."
speech += ("We have come to dedicate a portion of that field, as a final resting place for those who here gave their "
           "lives that that nation might live.")
speech += "It is altogether fitting and proper that we should do this."
speech += "But, in a larger sense, we can not dedicate - we can not consecrate - we can not hallow - this ground."
speech += ("The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add "
           "or detract.")
speech += "The world will little note, nor long remember what we say here, but it can never forget what they did here."
speech += ("It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here "
           "have thus far so nobly advanced.")
speech += ("It is rather for us to be here dedicated to the great task remaining before us - that from these honored "
           "dead we take increased devotion to that cause for which they gave the last full measure of devotion - "
           "that we here highly resolve that these dead shall not have died in vain - that this nation, under God, "
           "shall have a new birth of freedom - and that government of the people, by the people, for the people, "
           "shall not perish from the earth.")

print(summarizer(speech)[0]["summary_text"])
