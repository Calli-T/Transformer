from get_pretrained import *
from sentence_classifier_model_with_cnn import SentenceClassifier

from Transformer_basic.chap6.SentenceClassifier.dataloader import getDataLoader

from torch import nn, optim
import torch
import numpy as np

# ---------------하이퍼 파라미터와 모델, 그리고 기타등등--------------------------------------
from torch_directml import device
#device = "cpu"
device = device()

token_to_id, id_to_token = getVocabTable()
train_loader, test_loader = getDataLoader()

classifier = SentenceClassifier(pretrained_embedding=getWord2VecInitEmbedding(), filter_sizes=[3, 3, 4, 4, 5, 5],
                                max_length=32).to(device)
# 임베딩 테이블, 필터크기, 문장당 최대 단어수(= sequence)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# ------학습-----------------------------------------------------------------------------
def train(_model, _datasets, _criterion, _optimizer, _device, _interval):
    _model.train()
    losses = []

    for step, (input_ids, labels) in enumerate(_datasets):
        input_ids = input_ids.to(_device)
        labels = labels.to(_device).unsqueeze(1)

        logits = _model(input_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % _interval == 0:
            print(f"Train Loss {step} : {np.mean(losses)}")


def test(_model, _datasets, _criterion, _device):
    _model.eval()
    losses = []
    corrects = []

    for step, (input_ids, labels) in enumerate(_datasets):
        input_ids = input_ids.to(_device)
        labels = labels.to(_device).unsqueeze(1)

        logits = _model(input_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())

        yhat = torch.sigmoid(logits) > .5
        corrects.extend(torch.eq(yhat, labels).cpu().tolist())

    print(f"Val Loss : {np.mean(losses)}, Val Accuracy : {np.mean(corrects)}")


epochs = 5
interval = 500

for epoch in range(epochs):
    train(classifier, train_loader, criterion, optimizer, device, interval)
    test(classifier, test_loader, criterion, device)

torch.save(classifier.state_dict(), f"./models/cnn_model_word2vec_init_state_dict_{device}.pt")