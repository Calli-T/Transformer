from transformers import CvtForImageClassification, TrainingArguments


def get_model(classes, class_to_idx):
    model = CvtForImageClassification.from_pretrained(
        pretrained_model_name_or_path='microsoft/cvt-21',
        num_labels=len(classes),
        id2label={idx: label for label, idx, in class_to_idx.items()},  # 번호<->클래스 대응표, 아래는 반대로
        label2id=class_to_idx,
        ignore_mismatched_sizes=True
    )

    return model


# --- set hyper parameters  ---
args = TrainingArguments(
    output_dir="./models/CvT-FashionMNIST",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.001,
    load_best_model_at_end=True,
    metric_for_best_model="f1",  # 매크로 평균 F1원 점수
    logging_dir="logs",
    logging_steps=125,
    remove_unused_columns=False,
    seed=42,
)
