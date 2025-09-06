from utils.nn.variable import Variable, EPSILON

def categorical_crossentropy_loss(y_true: list[list[int]], y_predict: list[list[Variable]]) -> Variable:

    loss = 0.0
    real_val_size = len(y_true)

    for i in range(real_val_size):
        for j in range(len(y_true[i])):
            loss -= y_true[i][j] * (y_predict[i][j] + EPSILON).log()

    return loss / real_val_size


def mean_squared_error_loss(y_true: list[list[int]], y_predict: list[list[Variable]]) -> Variable:

    if len(y_true) != len(y_predict):
        raise ValueError('y_true and y_predict must be the same size')
    
    real_val_size = len(y_true)
    loss = sum([((y_t[0] - y_p[0]) ** 2) for y_t, y_p in zip(y_true, y_predict)]) / real_val_size

    return loss
