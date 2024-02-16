from sentence_classifier_model import SentenceClassifier
from get_vocab_dictionary import getVocabTable, getVocab
from torch_directml import device
from torch import nn, optim
import torch
from sentence_classifier_model import SentenceClassifier

# -----학습된거 가져오기----------------------------------------------------------------

token_to_id, id_to_token = getVocabTable()
dml = device()

# 단어사전 길이, 특성 차원, 임베딩 크기, 레이어 개수/분류 모델의 매개변수
n_vocab = len(token_to_id)
hidden_dim = 64
embedding_dim = 128
n_layers = 2

classifier = SentenceClassifier(_n_vocab=n_vocab, _hidden_dim=hidden_dim, _embedding_dim=embedding_dim,
                                _n_layers=n_layers).to(dml)

# 이 부분에서 모델 params를 가져온다
model_state_dict = torch.load('./models/model_random_init_state_dict.pt', map_location=dml)
classifier.load_state_dict(model_state_dict)

# -------------------------------------------------------------------------------
vocab = getVocab()
token_to_embedding = {}
embedding_matrix = classifier.embedding.weight.detach().cpu().numpy()

for word, emb in zip(vocab, embedding_matrix):
    token_to_embedding[word] = emb

'''
token = vocab[1000]
print(token, token_to_embedding)
'''

'''
for i in range(50):
    print(f"{vocab[i]}: 시작 5개: {token_to_embedding[vocab[i]][:5]}, 끝 5개: {token_to_embedding[vocab[i]][-5:]}")
'''