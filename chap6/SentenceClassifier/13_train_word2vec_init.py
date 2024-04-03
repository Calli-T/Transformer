from gensim.models import Word2Vec

from get_vocab_dictionary import getVocabTable
from dataloader import getDataLoader

from sentence_classifier_model import SentenceClassifier

from torch import nn, optim
import torch
from torch_directml import device
import numpy as np


def getWord2VecInitEmbedding():
    token_to_id, id_to_token = getVocabTable()

    word2vec = Word2Vec.load("../models/word2vec.model")
    init_embeddings = np.zeros((5002, 128))  # n_vocab, embedding_dim

    '''
    토큰과 인덱스를 가져와서, 토큰에 맞는 임베딩을
    (사전학습된) word2vec에서 가져온다음
    인덱스에 맞게 집어넣어준다.
    패딩과 OOV는 초기화에서 제외한다.
    '''
    for index, token in id_to_token.items():  # id_to_tokend은 dict
        if token not in ["<pad>", "<unk>"]:
            init_embeddings[index] = word2vec.wv[token]

    # torch의 Embedding클래스 중 from_pretrained로 텐서를 집어넣어주면된다
    # 해당 텐서는 아까 선언해둔 넘파일 배열을 적절히 dtype을 지정해서 변환한 텐서

    '''
    embedding_layer = nn.Embedding.from_pretrained(torch.tensor(init_embeddings, dtype=torch.float32))

    return embedding_layer
    '''
    return init_embeddings


# ---------------하이퍼 파라미터와 모델, 그리고 기타등등--------------------------------------
token_to_id, id_to_token = getVocabTable()
train_loader, test_loader = getDataLoader()
dml = device()

n_vocab = len(token_to_id)
hidden_dim = 64
embedding_dim = 128
n_layers = 2

classifier = SentenceClassifier(_n_vocab=n_vocab, _hidden_dim=hidden_dim, _embedding_dim=embedding_dim,
                                _n_layers=n_layers, pretrained_embedding=getWord2VecInitEmbedding()).to(dml)
criterion = nn.BCEWithLogitsLoss().to(dml)
optimizer = optim.RMSprop(classifier.parameters(), lr=0.001)


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
    train(classifier, train_loader, criterion, optimizer, dml, interval)
    test(classifier, test_loader, criterion, dml)

torch.save(classifier.state_dict(), './models/model_word2vec_init_state_dict.pt')
