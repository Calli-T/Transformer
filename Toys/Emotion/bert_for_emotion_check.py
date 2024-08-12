import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch import optim
from transformers import BertForSequenceClassification
from torch import nn
from tqdm import tqdm

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper params
epochs = 20
batch_size = 32
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path="bert-base-multilingual-cased",
    do_lower_case=False
)

'''# ----- get Dataset -----

# Raw 2 df
df1 = pd.read_excel("./한국어_연속적_대화_데이터셋.xlsx", skiprows=1)
df1 = df1.iloc[:, 1:3]
df1.iloc[:, 1] = df1.iloc[:, 1].apply(lambda x: 0 if x == '중립' else 1)

df2 = pd.read_excel("./한국어_단발성_대화_데이터셋.xlsx")
df2 = df2.iloc[:, 0:2]
df2.iloc[:, 1] = df2.iloc[:, 1].apply(lambda x: 0 if x == '중립' else 1)

df = pd.concat([df1, df2.rename(columns={"Sentence": "발화", "Emotion": "감정"})], ignore_index=True)
df = df.rename(columns={"발화": "text", "감정": "label"})
df['label'] = df['label'].astype(int)  # 감정 column을 str 2 int
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # 순서 무작위로 섞기
df.dropna(inplace=True)  # 결측치 제거
# print(df[:20].text.tolist())
# print(type(df[:20].label.values))

# train:val:test = 8:1:1
train, valid, test = np.split(df.sample(frac=1, random_state=42), [int(0.8 * len(df)), int(0.9 * len(df))])


# print(len(train), len(valid), len(test))

# df 2 dataset
def make_dataset(data, tokenizer, device):
    tokenized = tokenizer(
        text=data.text.tolist(),
        padding="longest",
        truncation=True,
        return_tensors='pt'
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    labels = torch.tensor(data.label.values, dtype=torch.long).to(device)

    return TensorDataset(input_ids, attention_mask, labels)


# dataset 2 dataloader
def get_dataloader(dataset, sampler, batch_size):
    data_sampler = sampler(dataset)
    dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=batch_size)

    return dataloader


train_dataset = make_dataset(train, tokenizer, device)
train_dataloader = get_dataloader(train_dataset, RandomSampler, batch_size)

valid_dataset = make_dataset(valid, tokenizer, device)
valid_dataloader = get_dataloader(valid_dataset, RandomSampler, batch_size)

test_dataset = make_dataset(test, tokenizer, device)
test_dataloader = get_dataloader(test_dataset, RandomSampler, batch_size)

# print(train_dataset[0])
'''

# ----- set model -----

model = BertForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path="bert-base-multilingual-cased",
    num_labels=2
).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)


# acc 계산
def calc_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# ----- train -----

'''# 학습시키는 함수, 매개변수로 모델/최적화/데이터로더(train)을 줘야함
def train(model, optimizer, dataloader):
    model.train()
    train_loss = 0.0

    for input_ids, attention_mask, labels in tqdm(dataloader):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(dataloader)
    return train_loss


# val, 중간 점검용
def evaluation(model, dataloader):
    with torch.no_grad():
        model.eval()
        criterion = nn.CrossEntropyLoss()
        val_loss, val_accuracy = 0.0, 0.0

        for input_ids, attention_mask, labels in tqdm(dataloader):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            logits = outputs.logits

            loss = criterion(logits, labels)
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to("cpu").numpy()
            accuracy = calc_accuracy(logits, label_ids)

            val_loss += loss
            val_accuracy += accuracy

    val_loss = val_loss / len(dataloader)
    val_accuracy = val_accuracy / len(dataloader)
    return val_loss, val_accuracy


# 학습 시작
best_loss = 10000
# model.load_state_dict(torch.load("./models/BERT_SENTENCE_CLASSIFICATION.pt"))
for epoch in range(epochs):
    print(f'Epoch: {epoch + 1}')
    train_loss = train(model, optimizer, train_dataloader)
    val_loss, val_accuracy = evaluation(model, valid_dataloader)
    print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Val Accuracy {val_accuracy:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), f'./models/BERT_SENTENCE_CLASSIFICATION_{epoch}.pt')
        print("Saved the model weights")

# 최종 정확도 테스트, val 함수랑 다를게 없으니 그대로 갑니다
model.config.pad_token_id = model.config.eos_token_id
# model.load_state_dict(torch.load("./models/BERT_SENTENCE_CLASSIFICATION.pt"))
test_loss, test_accuracy = evaluation(model, test_dataloader)

print("Testing")
print(f"Test Loss : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")'''

model.load_state_dict(torch.load("./models/BERT_SENTENCE_CLASSIFICATION_1.pt"))
# torch.save(model, './models/BERT_SENTENCE_CLASSIFICATION_model.pt')
# model = torch.load('./models/BERT_SENTENCE_CLASSIFICATION_model.pt')

def inference(sentence, _model, _tokenizer, _device):
    tokenized = _tokenizer(
        text=sentence,
        padding="longest",
        truncation=True,
        return_tensors='pt'
    )

    input_ids = tokenized["input_ids"].to(_device)
    attention_mask = tokenized["attention_mask"].to(_device)
    # labels = torch.tensor(data.label.values, dtype=torch.long).to(device)

    with torch.no_grad():
        _model.eval()

        outputs = _model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        prob = torch.nn.Softmax()(logits)
        # print(logits)
        # print(prob)
        return prob.argmax(dim=1).cpu().numpy()


# 여기 sentence에다 문장 list로 여러개 넣으면 여러 개 분석 다 해줌
# 0은 중립 1은 감정
print(inference(
    ["역시 정상화는 신창섭", "지금 몇 시지?", "코딩은 즐거워", "맙소사 여태까지 했던게 모두 WWE에 불과하다고", "점심 추천 좀", "내가 어제 뭐 했지?", "내가 어제 쓴 일기 봤어?",
     "오늘 존나 슬프다"], model, tokenizer, device))
