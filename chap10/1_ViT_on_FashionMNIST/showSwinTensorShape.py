import inspect

from setSwinModel import *
from FashionMNIST_Dataloader import *

model = get_model(classes, class_to_idx)


def showArchitecture():
    for main_name, main_module in model.named_children():
        print(main_name)
        for sub_name, sub_module in main_module.named_children():
            print("└", sub_name)
            for ssub_name, ssub_module in sub_module.named_children():
                print("│ └", ssub_name)
                for sssub_name, sssub_module in sub_module.named_children():
                    print("│ │ └", sssub_name)


# swin transformer block구조
def show_swin_block():
    for main_name, main_module in model.swin.encoder.layers[0].named_children():
        print(main_name)
        for sub_name, sub_module in main_module.named_children():
            print("└", sub_name)
            for ssub_name, ssub_module in sub_module.named_children():
                print("│ └", ssub_name)


# showArchitecture()
# show_swin_block()
# print(model.swin.encoder.layers[0].blocks[0])

batch = next(iter(train_dataloader))
print("Dimension :", batch["pixel_values"].shape)

patch_emb_output, shape = model.swin.embeddings.patch_embeddings(batch["pixel_values"])

print("패치 임베딩 차원 :", patch_emb_output.shape)

W_MSA = model.swin.encoder.layers[0].blocks[0]
SW_MSA = model.swin.encoder.layers[0].blocks[1]

W_MSA_output = W_MSA(patch_emb_output, W_MSA.input_resolution)[0]
SW_MSA_output = SW_MSA(W_MSA_output, SW_MSA.input_resolution)[0]

print("W-MSA 결과 차원: ", W_MSA_output.shape)
print("SW-MSA 결과 차원: ", SW_MSA_output.shape)

POOLER = model.swin.pooler
print(POOLER) # 정황상 채널 96차원을 output크기인 1으로 만들어주는듯
# https://underflow101.tistory.com/41
# adaptive pooling은 출력 크기를 고정한다
NORM = model.swin.layernorm
print(NORM)
