from gensim.models import Word2Vec
import numpy as np
from get_vocab_dictionary import getVocabTable
from torch import nn
import torch

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

    embedding_layer = nn.Embedding.from_pretrained(torch.tensor(init_embeddings, dtype=torch.float32))

    return embedding_layer
