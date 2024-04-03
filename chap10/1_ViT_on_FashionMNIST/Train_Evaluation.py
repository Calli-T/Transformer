from setModel import *
from FashionMNIST_Dataloader import *
from setMetric import *

from transformers import TrainingArguments, Trainer

# -
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- set hyper parameters & set Trainer ---
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
    seed=42
)
trainer = Trainer(
    model_init=lambda x: get_model(classes, class_to_idx),
    args=args,
    train_dataset=subset_train_dataset,
    eval_dataset=subset_test_dataset,
    data_collator=lambda x: collator(x, transform),
    compute_metrics=compute_metrics,
    tokenizer=image_processor,
)
trainer.train()

# --- 혼동행렬을 활용한 모델의 성능 평가 ---

outputs = trainer.predict(subset_test_dataset)
print(outputs)

y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

labels = list(classes)
matrix = confusion_matrix(y_true, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
_, ax = plt.subplots(figsize=(10, 10))
display.plot(xticks_rotation=45, ax=ax)
plt.show()
