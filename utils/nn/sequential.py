import json

from utils.nn.layer import Layer

def is_in_type_union(value: any, union_type: any):
    
    if hasattr(union_type, '__args__'):
        return issubclass(value, union_type.__args__)
    
    return False


class Sequential:

    def __init__(self, layers: list[Layer]):

        self.layers = layers
        self._validate_layers(layers)

        self.optmizer = None
        self.loss_function = None
        self.metric = None

        self.batch_size = None
        self.epoch = None

        self.x_train = None
        self.y_train = None
        self.x_validate = None
        self.y_validate = None


    def _validate_layers(layers: list[Layer]) -> None:

        if not isinstance(layers, list) or len(layers) == 0:
            raise ValueError('Sequential model layer list should not be empty')

        for (idx, layer) in enumerate(layers):
            if not is_in_type_union(layer, Layer):
                raise TypeError(f'Invalid type of sequential model layer at idx #{idx} (got {type(layer)})')

        # Layer continuity check
        for idx in range(len(layers) - 1):

            output_size = layers[idx].input_size
            next_input_size = layers[idx + 1].input_size

            if output_size != next_input_size:
                raise ValueError(f'Layer shape mismatched between layer #{idx} and #{idx + 1} (output size not match with input size of next layer)')
            
    
    def as_json(self, path: str = './model.json') -> None:

        states = [layer.as_dict() for layer in self.layers]

        with open(path, 'w') as f:
            json.dump({ 'layers': states }, f, indent=4)