from torch import nn, optim
from get_vocab_dictionary import getVocabTable
from dataloader import getDataLoader
from torch_directml import device
import numpy as np
import torch
from sentence_classifier_model import SentenceClassifier

token_to_id, id_to_token = getVocabTable()
train_loader, test_loader = getDataLoader()
dml = device()

# 단어사전 길이, 특성 차원, 임베딩 크기, 레이어 개수/분류 모델의 매개변수
n_vocab = len(token_to_id)
hidden_dim = 64
embedding_dim = 128
n_layers = 2

classifier = SentenceClassifier(_n_vocab=n_vocab, _hidden_dim=hidden_dim, _embedding_dim=embedding_dim,
                                _n_layers=n_layers).to(dml)
criterion = nn.BCEWithLogitsLoss().to(
    dml)  # https://ok-lab.tistory.com/241     이진교차엔트로피인데, 시그모이드 없으니 끝에 달아줘서 확률(Logits)로 자동으로 변환처리해서 해줌
optimizer = optim.RMSprop(classifier.parameters(), lr=0.001)


# ------학습-----------------------------------------------------------------------------

def train(_model, _datasets, _criterion, _optimizer, _device, _interval):
    _model.train()  # https://wikidocs.net/195118        모델을 학습모드로 변환
    losses = []

    for step, (input_ids, labels) in enumerate(_datasets):
        input_ids = input_ids.to(_device)
        labels = labels.to(_device).unsqueeze(1)  # https://iambeginnerdeveloper.tistory.com/210      차원을 늘려주는 함수

        # 모델에 넣고, 손실함수에 넣고, 손실값을 손실 리스트에 추가     https://discuss.pytorch.org/t/what-is-loss-item/61218
        logits = _model(input_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())  # item은 loss의 값, float

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
        losses.append(loss.item())  # item은 loss의 값, float

        # 예측과 라벨값이 일치하는지 검증(validate), torch.eq
        yhat = torch.sigmoid(logits) > .5
        corrects.extend(torch.eq(yhat, labels).cpu().tolist())

    print(f"Val Loss : {np.mean(losses)}, Val Accuracy : {np.mean(corrects)}")


epochs = 5
interval = 500

for epoch in range(epochs):
    # 에포크마다 테스트 데이터셋으로 모델의 검증손실과 검증정확도를 확인
    train(classifier, train_loader, criterion, optimizer, dml, interval)
    test(classifier, test_loader, criterion, dml)

torch.save(classifier.state_dict(), './models/model_random_init_state_dict.pt')