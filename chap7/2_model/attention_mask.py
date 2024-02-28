from batch_dataloader import *

'''
어텐션 마스크를 만드는 트릭
배열 크기를 받아 1로 가득찬 배열 만들고
이를 상삼각행렬로 만들고
이를 전치하여 하삼각행렬로 만듬(대각성분 1)
값이 0인 곳은 -inf로 어텐션 마스크가 되고, 값이 1인 곳은 0.0 값이된다
'''


def generate_square_subsequent_mask(s):
    mask = (torch.triu(torch.ones((s, s), device=DEVICE))).transpose(0, 1).transpose(0, 1)
    mask = (mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0)))

    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    # 타겟과 소스의 마스크가 다름, 인코더는 마스크드 멀티헤드 어텐션 안씀!!!, 0 행렬 사용
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    # 패딩 마스크 제작, 패딩일 경우 해당 위치가 1?
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

'''
패딩 마스크 생성 이전, 입력 값과 출력 값은 토큰 순서를 한칸 시프트 하여
이전 토큰들이 주어지면 다음 토큰을 예측하도록 함
torch.Size([34, 128])로 나옴
'''
target_input = target_tensor[:-1, :]
target_out = target_tensor[1:, :]

print("- Attention Mask -")
source_mask, target_mask, source_padding_mask, target_padding_mask = create_mask(source_tensor, target_input)


print("source_mask: ", source_mask.shape)
print(source_mask)
print("target_mask: ", target_mask.shape)
print(target_mask)
print("source_padding_mask: ", source_padding_mask.shape)
print(source_padding_mask)
print("target_padding_mask: ", target_padding_mask.shape)
print(target_padding_mask)

# 5