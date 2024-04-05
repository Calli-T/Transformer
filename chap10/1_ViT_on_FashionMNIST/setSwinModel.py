from transformers import SwinForImageClassification, TrainingArguments


def get_model(classes, class_to_idx):
    model = SwinForImageClassification.from_pretrained(
        pretrained_model_name_or_path="microsoft/swin-tiny-patch4-window7-224",
        num_labels=len(classes),
        id2label={idx: label for label, idx in class_to_idx.items()},
        label2id=class_to_idx,
        ignore_mismatched_sizes=True,
    )
    return model


args = TrainingArguments(
    output_dir="./models/Swin-FashionMNIST",
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

'''def showArchitecture():
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
# print(model.swin.encoder.layers[0].blocks[0])'''
