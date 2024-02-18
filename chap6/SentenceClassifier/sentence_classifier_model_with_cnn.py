import torch
from torch import nn


class SentenceClassifier(nn.Module):
    def __init__(self, pretrained_embedding, filter_sizes, max_length, dropout=0.5):
        super().__init__()

        # 사전 학습된 단어의 임베딩 가져온다
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(pretrained_embedding, dtype=torch.float32))

        # 임베딩 테이블을 열개수를 보고 임베딩 벡터의 길이를 확인한다
        embedding_dim = self.embedding.weight.shape[1]

        conv = []
        # 입력된 사이즈들의 필터를 하나씩 만들어준다
        for size in filter_sizes:
            temp_filter = nn.Sequential(
                nn.Conv1d(in_channels=embedding_dim, out_channels=1, kernel_size=size),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=max_length - size - 1)  # 식이 좀 복잡하긴한데, 결국 풀링의 결과는 숫자 단 하나
            )
            conv.append(temp_filter)
        '''
        https://bo-10000.tistory.com/entry/nnModuleList
        모듈리스트는 토치의 뉴럴 네트워크의 조상클래스인 module의 list를 가지고, 관리한다.
        '''
        self.conv_filters = nn.ModuleList(conv)

        '''
        풀링의 결과가 하나이므로, 이어붙인건 결국 필터 개수만큼의 차원을가진 벡터이다
        분류 이전에 완전 연결층을 하나 더 추가한 느낌, 입력이나 출력이나 필터 개수 n개만큼이다
        드롭아웃 규제 때려주고
        분류기는 n개 -> 1개 결과를 뱉는다.
        '''
        output_size = len(filter_sizes)
        self.pre_classifier = nn.Linear(output_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_size, 1)

    def forward(self, inputs):
        '''
        임베딩 벡터를 얻고
        그걸 permute로 차원(의 순서를) 변경(맞교환)한다
        '''
        embeddings = self.embedding(inputs)
        embeddings = embeddings.permute(0, 2, 1)

        '''
        가진 모든 합성곱 필터에 임베딩을 넣고 결과를 뽑아낸다
        결과 각각의 차원을 쥐어짜(squeeze)고, 그걸 이어붙(concatenate)여서 하나의 벡터로 만들어낸다 
        '''
        conv_outputs = [conv(embeddings) for conv in self.conv_filters]
        concat_outputs = torch.cat([conv.squeeze(-1) for conv in conv_outputs], dim=1)

        logits = self.pre_classifier(concat_outputs)
        logits = self.dropout(logits)
        logits = self.classifier(logits)

        return logits
