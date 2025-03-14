# instmodel

**instmodel** is a Python package designed to simplify the creation of *instruction-based* neural network models using a Keras backend. With `instmodel`, you can quickly build, train, and export your models into a compact “instruction” format suitable for lightweight inference, serialization, or deployment.

---

## Features

- **High-Level Model Building**: Easily construct neural networks using familiar Keras-like layers such as `Dense`, `Attention`, and more.
- **Instruction Model Export**: Convert your trained models into a JSON-based “instruction model” that precisely reflects the network architecture, weights, and activations.
- **Validation & Debugging**: Verify that the instruction model produces the same outputs as the original Keras model with built-in validation functions.
- **Easy Deployment**: Use the exported instruction model for lightweight or custom inference scenarios (e.g., embedded systems).

---

## Installation

Install the latest release from PyPI:

```bash
pip install instmodel
```

---

## Quick Example

Below is a simplified example (adapted from the test suite) showing how to:

1. Define a feed-forward network using Keras-like syntax.
2. Train it on dummy data.
3. Export it as an instruction model.
4. Validate it against the original Keras model outputs.

```python
import numpy as np
from instmodel.model import (
    Dense,
    InputBuffer,
    ModelGraph,
    ff_model,
    validate_keras_model
)
from instmodel.instruction_model import validate_instruction_model

# 1. Define a simple feed-forward model (three Dense layers).
input_buffer = InputBuffer(4, name="simple_input")
hidden = Dense(8, activation="relu", name="hidden_relu_1")(input_buffer)
hidden = Dense(6, activation="relu", name="hidden_relu_2")(hidden)
output = Dense(1, activation="sigmoid", name="output_sigmoid")(hidden)

model_graph = ModelGraph(input_buffer, output)
model_graph.compile(optimizer="adam", loss="binary_crossentropy")

# 2. Train the model on dummy data.
x_data = np.random.random((10, 4))
y_data = np.random.randint(0, 2, size=(10, 1))
model_graph.fit(x_data, y_data, epochs=1, verbose=0)

# 3. Export the trained Keras model to an instruction model.
instruction_model = model_graph.create_instruction_model()

# 4. Validate the exported instruction model against the original Keras outputs.
keras_pred = model_graph.predict(x_data, verbose=0)
instruction_model["validation_data"] = {
    "inputs": x_data.tolist(),
    "expected_outputs": keras_pred.tolist(),
}

validate_instruction_model(instruction_model)  # Check instruction-model output
validate_keras_model(model_graph.get_keras(), instruction_model["validation_data"])  # Compare with Keras model
```
---

## License

This project is licensed under the [MIT License](LICENSE).