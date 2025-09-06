import random
import math
from typing import List

# Assuming your modules are in utils.nn package
from utils.nn.sequential import Sequential
from utils.nn.layer import Dense
from utils.nn.neuron import Activation
from utils.nn.initializer import Initializer, InitializationType
from utils.nn.optimizer import Adam, SGD
from utils.nn.losses import categorical_crossentropy_loss, mean_squared_error_loss


def generate_classification_data(n_samples: int = 1000, n_features: int = 4, n_classes: int = 3) -> tuple:
    """Generate synthetic classification data"""
    X = []
    y = []
    
    for _ in range(n_samples):
        # Generate random features
        features = [random.uniform(-2, 2) for _ in range(n_features)]
        
        # Simple rule to assign classes based on feature combinations
        if features[0] + features[1] > 0.5:
            class_idx = 0
        elif features[2] - features[3] > 0:
            class_idx = 1
        else:
            class_idx = 2
        
        # One-hot encode the class
        one_hot = [0.0] * n_classes
        one_hot[class_idx] = 1.0
        
        X.append(features)
        y.append(one_hot)
    
    return X, y


def generate_regression_data(n_samples: int = 800) -> tuple:
    """Generate synthetic regression data"""
    X = []
    y = []
    
    for _ in range(n_samples):
        # Generate 2 input features
        x1 = random.uniform(-3, 3)
        x2 = random.uniform(-3, 3)
        
        # Non-linear target function with some noise
        target = 0.5 * x1**2 + 0.3 * x2 + 0.2 * x1 * x2 + random.normalvariate(0, 0.1)
        
        X.append([x1, x2])
        y.append([target])
    
    return X, y


def train_test_split(X: List, y: List, test_size: float = 0.2) -> tuple:
    """Simple train-test split"""
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Shuffle indices
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test


def classification_example():
    """Example of MLP for classification"""
    print("=" * 60)
    print("CLASSIFICATION EXAMPLE")
    print("=" * 60)
    
    # Generate synthetic classification data
    X, y = generate_classification_data(n_samples=1000, n_features=4, n_classes=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input features: {len(X_train[0])}")
    print(f"Output classes: {len(y_train[0])}")
    
    # Create MLP architecture
    # Input: 4 features -> Hidden: 8 neurons -> Hidden: 6 neurons -> Output: 3 classes
    model = Sequential([
        Dense(
            shape=(4, 8), 
            activation=Activation.RELU,
            initializer=Initializer(
                fill_type=InitializationType.HE_NORMAL,
                fan_in=4,
                fan_out=8
            )
        ),
        Dense(
            shape=(8, 6), 
            activation=Activation.RELU,
            initializer=Initializer(
                fill_type=InitializationType.HE_NORMAL,
                fan_in=8,
                fan_out=6
            )
        ),
        Dense(
            shape=(6, 3), 
            activation=Activation.SOFTMAX,
            initializer=Initializer(
                fill_type=InitializationType.GLOROT_NORMAL,
                fan_in=6,
                fan_out=3
            )
        )
    ])
    
    # Get all parameters for the optimizer
    all_params = []
    for layer in model.layers:
        all_params.extend(layer.get_params())
    
    # Create optimizer
    optimizer = Adam(params=all_params, learning_rate=0.001)
    
    print(f"\nTotal parameters: {len(all_params)}")
    print("Model architecture:")
    for i, layer in enumerate(model.layers):
        print(f"  Layer {i+1}: {layer.input_size} -> {layer.output_size} ({layer.activation.value})")
    
    # Train the model
    print("\nTraining...")
    model.fit(
        x_train=X_train,
        y_train=y_train,
        optimizer=optimizer,
        loss_func=categorical_crossentropy_loss,
        epochs=50,
        batch_size=32,
        metric="accuracy",
        x_validate=X_test,
        y_validate=y_test,
        display_interval=10
    )
    
    # Make predictions on test set
    print("\nFinal evaluation on test set:")
    test_predictions = model.forward_batch(X_test[:5])  # Just first 5 samples
    
    print("Sample predictions vs actual:")
    for i in range(5):
        pred_values = [float(v) for v in test_predictions[i]]
        pred_class = pred_values.index(max(pred_values))
        actual_class = y_test[i].index(max(y_test[i]))
        
        print(f"  Sample {i+1}: Predicted class {pred_class} (confidence: {max(pred_values):.3f}), "
              f"Actual class {actual_class}")


def regression_example():
    """Example of MLP for regression"""
    print("\n" + "=" * 60)
    print("REGRESSION EXAMPLE")
    print("=" * 60)
    
    # Generate synthetic regression data
    X, y = generate_regression_data(n_samples=800)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input features: {len(X_train[0])}")
    
    # Create MLP architecture for regression
    # Input: 2 features -> Hidden: 10 neurons -> Hidden: 5 neurons -> Output: 1 value
    model = Sequential([
        Dense(
            shape=(2, 10), 
            activation=Activation.TANH,
            initializer=Initializer(
                fill_type=InitializationType.GLOROT_UNIFORM,
                fan_in=2,
                fan_out=10
            )
        ),
        Dense(
            shape=(10, 5), 
            activation=Activation.TANH,
            initializer=Initializer(
                fill_type=InitializationType.GLOROT_UNIFORM,
                fan_in=10,
                fan_out=5
            )
        ),
        Dense(
            shape=(5, 1), 
            activation=Activation.LINEAR,  # Linear activation for regression output
            initializer=Initializer(
                fill_type=InitializationType.GLOROT_UNIFORM,
                fan_in=5,
                fan_out=1
            )
        )
    ])
    
    # Get all parameters for the optimizer
    all_params = []
    for layer in model.layers:
        all_params.extend(layer.get_params())
    
    # Create optimizer
    optimizer = SGD(params=all_params, learning_rate=0.01)
    
    print(f"\nTotal parameters: {len(all_params)}")
    print("Model architecture:")
    for i, layer in enumerate(model.layers):
        print(f"  Layer {i+1}: {layer.input_size} -> {layer.output_size} ({layer.activation.value})")
    
    # Train the model
    print("\nTraining...")
    model.fit(
        x_train=X_train,
        y_train=y_train,
        optimizer=optimizer,
        loss_func=mean_squared_error_loss,
        epochs=100,
        batch_size=16,
        metric="r2_score",
        x_validate=X_test,
        y_validate=y_test,
        display_interval=20
    )
    
    # Make predictions on test set
    print("\nSample predictions vs actual:")
    test_predictions = model.forward_batch(X_test[:5])
    
    for i in range(5):
        pred_value = float(test_predictions[i][0])
        actual_value = y_test[i][0]
        error = abs(pred_value - actual_value)
        
        print(f"  Sample {i+1}: Predicted {pred_value:.3f}, Actual {actual_value:.3f}, "
              f"Error: {error:.3f}")


def model_persistence_example():
    """Example of saving and loading models"""
    print("\n" + "=" * 60)
    print("MODEL PERSISTENCE EXAMPLE")
    print("=" * 60)
    
    # Create a simple model
    model = Sequential([
        Dense(shape=(3, 5), activation=Activation.RELU),
        Dense(shape=(5, 2), activation=Activation.SIGMOID)
    ])
    
    # Make a prediction before saving
    test_input = [[1.0, 0.5, -0.3]]
    prediction_before = model.forward_batch(test_input)
    print(f"Prediction before saving: {[float(v) for v in prediction_before[0]]}")
    
    # Save model
    model.as_json("./example_model.json")
    print("Model saved to 'example_model.json'")
    
    # Create a new model and load the saved state
    new_model = Sequential([
        Dense(shape=(3, 5), activation=Activation.RELU),
        Dense(shape=(5, 2), activation=Activation.SIGMOID)
    ])
    
    new_model.from_json("./example_model.json")
    print("Model loaded from 'example_model.json'")
    
    # Make the same prediction with loaded model
    prediction_after = new_model.forward_batch(test_input)
    print(f"Prediction after loading: {[float(v) for v in prediction_after[0]]}")
    
    # Check if predictions match
    matches = all(abs(float(a) - float(b)) < 1e-10 
                  for a, b in zip(prediction_before[0], prediction_after[0]))
    print(f"Predictions match: {matches}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Run examples
    classification_example()
    regression_example()
    model_persistence_example()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)