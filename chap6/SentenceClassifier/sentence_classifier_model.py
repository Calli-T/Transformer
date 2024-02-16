from torch import nn


class SentenceClassifier(nn.Module):
    def __init__(self, _n_vocab, _hidden_dim, _embedding_dim, _n_layers, dropout=0.5,
                 bidirectional=True, model_type="lstm", pretrained_embedding=None):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=_n_vocab, embedding_dim=_embedding_dim, padding_idx=0)

        if model_type == "rnn":
            self.model = nn.RNN(input_size=_embedding_dim,
                                hidden_size=_hidden_dim,
                                num_layers=_n_layers,
                                bidirectional=bidirectional,
                                dropout=dropout,
                                batch_first=True)
        elif model_type == "lstm":
            self.model = nn.LSTM(input_size=_embedding_dim,
                                 hidden_size=_hidden_dim,
                                 num_layers=_n_layers,
                                 bidirectional=bidirectional,
                                 dropout=dropout,
                                 batch_first=True)

        if bidirectional:
            self.classifier = nn.Linear(_hidden_dim * 2, 1)
        else:
            self.classifier = nn.Linear(_hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        output, _ = self.model(embeddings)
        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)
        logits = self.classifier(last_output)
        return logits
