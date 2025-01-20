from train_model import *


def greedy_decode(model, source_tensor, source_mask, max_len, start_symbol):
    # memory를 뽑기위한 소스를 텐서로
    # 모든 토큰이 어텐션 될 수있도록 마스크 값은 0
    source_tensor = source_tensor.to(DEVICE)
    source_mask = source_mask.to(DEVICE)

    # memory는 마지막 인코더 트랜스포머 블록의 벡터 (context vector)
    # ys는 타깃 데이터의 입력텐서, 언더바는 바꿔치기 연산을 의미한다
    # https://tutorials.pytorch.kr/beginner/former_torchies/tensor_tutorial_old.html
    memory = model.encode(source_tensor, source_mask)
    # 1, 1 크기의 텐서, BOS로 시작한다
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        # 타깃 마스크 생성
        target_mask = generate_square_subsequent_mask(ys.size(0))
        target_mask = target_mask.type(torch.bool).to(DEVICE)

        # 입력텐서, context vector, mask를 사용한 decode
        out = model.decode(ys, memory, target_mask)
        out = out.transpose(0, 1)  # 전치
        prob = model.generator(out[:, -1])  # 결과물을 선형층으로
        _, next_word = torch.max(prob, dim=1)  # 다음 단어, 언더바는 뭘 받아주는가???
        next_word = next_word.item()  # ???

        ys = torch.cat(  # 아마도, 입력 텐서를 만들기 위해 병합하는 과정
            [ys, torch.ones(1, 1).type_as(source_tensor.data).fill_(next_word)], dim=0
        )
        if next_word == EOS_IDX:
            break
    return ys


def translate(model, source_sentence):
    model.eval()
    source_tensor = text_transform[SRC_LANGUAGE](source_sentence).view(-1, 1)  # view 함수로(?, 1)의 크기로 변경, 즉 한 줄로 쭉
    num_tokens = source_tensor.shape[0]  # 토큰수
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)  # 소스 마스크는 0으로
    tgt_tokens = greedy_decode(  # 결과 시퀀스를 가져옴
        model, source_tensor, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX
    ).flatten()
    output = vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))[1:-1]  # 단어로 변경
    return " ".join(output)  # 띄워쓰기를 사용하여 join 즉 문장으로 변경


# model = torch.load('./models/de2en_transformer_privateuseone')

output_oov = translate(model, "Eine Gruppe von Menschen steht vor einem Iglu .")
output = translate(model, "Eine Gruppe von Menschen steht vor einem Gebäude .")
print(output_oov)
print(output)

# 6
