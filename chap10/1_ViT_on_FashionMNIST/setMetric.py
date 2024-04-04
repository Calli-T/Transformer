import evaluate
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    metric = evaluate.load("f1")
    macro_f1 = metric.compute(
        predictions=predictions, references=labels, average="macro"
    )

    return macro_f1

# F1-score란?
# https://velog.io/@jadon/F1-score%EB%9E%80
# https://data-minggeul.tistory.com/11