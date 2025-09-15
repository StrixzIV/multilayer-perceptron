from utils.nn.variable import Variable

def variables_to_float(variables: list[list[Variable]]) -> list[list[float]]:
    return [[float(var) for var in var_list] for var_list in variables]


def accuracy_score(y_true: list[list[float]], y_predict: list[list[Variable]]) -> float:

    y_predict = variables_to_float(y_predict)

    y_true_classes = [t.index(max(t)) for t in y_true]
    y_pred_classes = [p.index(max(p)) for p in y_predict]

    correct = sum(p == t for p, t in zip(y_pred_classes, y_true_classes))

    return correct / len(y_true_classes)


def f1_score(y_true: list[list[float]], y_predict: list[list[Variable]]) -> float:

    y_predict = variables_to_float(y_predict)

    y_true_classes = [true.index(max(true)) for true in y_true]
    y_pred_classes = [pred.index(max(pred)) for pred in y_predict]
    
    f1_scores = []
    num_classes = len(y_true[0])

    for c in range(num_classes):

        tp = sum(p == c and t == c for p, t in zip(y_pred_classes, y_true_classes))
        fp = sum(p == c and t != c for p, t in zip(y_pred_classes, y_true_classes))
        fn = sum(p != c and t == c for p, t in zip(y_pred_classes, y_true_classes))

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        f1_scores.append(f1)
    
    return sum(f1_scores) / num_classes


def r2_score(y_true: list[list[float]], y_predict: list[list[Variable]]) -> float:

    true_values = [true[0] for true in y_true]
    pred_values = [pred[0] for pred in y_predict]
    
    mean_true = sum(true_values) / len(true_values)
    ss_total = sum((y - mean_true) ** 2 for y in true_values)
    ss_residual = sum((t - p) ** 2 for t, p in zip(true_values, pred_values))
    
    return 1 - ss_residual / ss_total if ss_total > 0 else (1 if ss_residual == 0 else 0)
