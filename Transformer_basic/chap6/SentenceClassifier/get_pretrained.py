from gensim.models import Word2Vec
import numpy as np

from get_vocab_dictionary import getVocabTable


def getWord2VecInitEmbedding():
    token_to_id, id_to_token = getVocabTable()

    word2vec = Word2Vec.load("../models/word2vec.model")
    init_embeddings = np.zeros((5002, 128))  # n_vocab, embedding_dim

    for index, token in id_to_token.items():  # id_to_tokend은 dict
        if token not in ["<pad>", "<unk>"]:
            init_embeddings[index] = word2vec.wv[token]

    return init_embeddings
