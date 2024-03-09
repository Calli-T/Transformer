from mask import *


def greedy_decode(_model, source_tensor, source_mask, max_len, start_symbol):
    source_tensor = source_tensor.to(DEVICE)
    source_mask = source_mask.to(DEVICE)

    memory = _model.encode(source_tensor, source_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    print(memory)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        for i in memory:
            for ii in i:
                print(ii[:10])

        target_mask = generate_square_subsequent_mask(ys.size(0))
        target_mask = target_mask.type(torch.bool).to(DEVICE)

        out = _model.decode(ys, memory, target_mask)
        out = out.transpose(0, 1)

        prob = _model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()  # ???

        ys = torch.cat(
            [ys, torch.ones(1, 1).type_as(source_tensor.data).fill_(next_word)], dim=0
        )
        if next_word == EOS_IDX:
            break

    return ys


def translate(_model, source_sentence):
    _model.eval()
    source_tensor = text_transform[SRC_EDITION](source_sentence).view(-1, 1)
    # print(source_tensor)
    num_tokens = source_tensor.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    # print(src_mask)
    tgt_tokens = greedy_decode(
        _model, source_tensor, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX
    ).flatten()
    # print(tgt_tokens)
    output = vocab_transform[TGT_EDITION].lookup_tokens(list(tgt_tokens.cpu().numpy()))[1:-1]  # 단어로 변경
    # print(output)
    return " ".join(output)


model_state_dict = torch.load('./models/5sentence2bible.pt', map_location=DEVICE)
model.load_state_dict(model_state_dict)

print('---------------------------------------')
output = translate(model, "떠나는 길에 네가 내게 말했지 너는 바라는 게 너무나 많아")
print(output)
print(len(output))
print('---------------------------------------')

# 6
