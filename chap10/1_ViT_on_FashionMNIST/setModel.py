from transformers import ViTForImageClassification, TrainingArguments
# --- set model ---
def get_model(classes, class_to_idx):
    model = ViTForImageClassification.from_pretrained(
        pretrained_model_name_or_path="google/vit-base-patch16-224-in21k",
        num_labels=len(classes),
        id2label={idx: label for label, idx, in class_to_idx.items()},  # 번호<->클래스 대응표, 아래는 반대로
        label2id=class_to_idx,
        ignore_mismatched_sizes=True
    )
    return model


# print(model.classifier)
# 모델 임베딩

'''# 패치임베딩이 224*224에 16*16크기의 커널을 768개를 한 필터로 하여 16stride로 적용하여 conv2d적용, 14x14개의 패치가 제작됨
# 좌->우/상->하 순서로 늘어서서 196개가 된다
# CLS 토큰을 포함하여 197개가 됨/(CLS 토큰은 [4, 1, 768] 차원이라고 한다)
print(model.vit.embeddings)
batch = next(iter(train_dataloader))
print("image shape: ", batch["pixel_values"].shape)
print("patch embeddings shape: ", model.vit.embeddings.patch_embeddings(batch["pixel_values"]).shape)
print("[CLS] + patch embeddings shape :", model.vit.embeddings(batch["pixel_values"]).shape)'''


# --- set hyper parameters  ---
args = TrainingArguments(
    output_dir="./models/ViT-FashionMNIST",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1",  # 매크로 평균 F1원 점수
    logging_dir="logs",
    logging_steps=125,
    remove_unused_columns=False,
    seed=42,
)