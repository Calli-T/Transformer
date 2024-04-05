import os

# 모델 스위치하려면 FashionMNIST_Dataloader.py의 image_processor도 스왑할것
# from setModel import *
from setSwinModel import *
from FashionMNIST_Dataloader import *
from setMetric import *

from transformers import Trainer

# -
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# set device
from device_converter import device

# -

trainer = Trainer(
    model_init=lambda x: get_model(classes, class_to_idx).to(device),
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
