# instmodel

**instmodel** is a Python package for building *instruction-based* neural network models with either a **PyTorch** or **TensorFlow/Keras** backend. Build, train, and export models into a compact JSON "instruction" format for lightweight, backend-agnostic inference.

---

## Features

- **Dual Backend**: Build models with PyTorch (`instmodel.torch`) or TensorFlow/Keras (`instmodel.tf`) — both are optional dependencies.
- **Instruction Model Export**: Convert trained models into a JSON-based instruction format that captures architecture, weights, and activations.
- **Backend-Agnostic Inference**: Run exported instruction models with pure NumPy via `instmodel.instruction_model` — no framework required at inference time.
- **Validation**: Verify that the instruction model produces the same outputs as the original trained model.

---

## Installation

Install the core package (NumPy inference only):

```bash
pip install instmodel
```

Install with a training backend:

```bash
pip install instmodel[pytorch]    # PyTorch backend
pip install instmodel[tensorflow] # TensorFlow/Keras backend
```

---

## Quick Example — PyTorch

```python
import numpy as np
from instmodel.torch import (
    Dense,
    InputBuffer,
    ModelGraph,
    ff_model,
    validate_torch_model,
)
from instmodel.instruction_model import validate_instruction_model

# 1. Define a simple feed-forward model.
input_buffer = InputBuffer(4, name="simple_input")
hidden = Dense(8, activation="relu", name="hidden_relu_1")(input_buffer)
hidden = Dense(6, activation="relu", name="hidden_relu_2")(hidden)
output = Dense(1, activation="sigmoid", name="output_sigmoid")(hidden)

model_graph = ModelGraph(input_buffer, output)
model_graph.compile(optimizer="adam", loss="binary_crossentropy")

# 2. Train on dummy data.
x_data = np.random.random((10, 4))
y_data = np.random.randint(0, 2, size=(10, 1))
model_graph.fit(x_data, y_data, epochs=1, verbose=0)

# 3. Export to instruction model.
instruction_model = model_graph.create_instruction_model()

# 4. Validate.
torch_pred = model_graph.predict(x_data)
instruction_model["validation_data"] = {
    "inputs": x_data.tolist(),
    "expected_outputs": torch_pred.tolist(),
}
validate_instruction_model(instruction_model)
validate_torch_model(model_graph.get_torch(), instruction_model["validation_data"])
```

## Quick Example — TensorFlow/Keras

```python
import numpy as np
from instmodel.tf import (
    Dense,
    InputBuffer,
    ModelGraph,
    ff_model,
    validate_keras_model,
)
from instmodel.instruction_model import validate_instruction_model

# 1. Define a simple feed-forward model.
input_buffer = InputBuffer(4, name="simple_input")
hidden = Dense(8, activation="relu", name="hidden_relu_1")(input_buffer)
hidden = Dense(6, activation="relu", name="hidden_relu_2")(hidden)
output = Dense(1, activation="sigmoid", name="output_sigmoid")(hidden)

model_graph = ModelGraph(input_buffer, output)
model_graph.compile(optimizer="adam", loss="binary_crossentropy")

# 2. Train on dummy data.
x_data = np.random.random((10, 4))
y_data = np.random.randint(0, 2, size=(10, 1))
model_graph.fit(x_data, y_data, epochs=1, verbose=0)

# 3. Export to instruction model.
instruction_model = model_graph.create_instruction_model()

# 4. Validate.
keras_pred = model_graph.predict(x_data, verbose=0)
instruction_model["validation_data"] = {
    "inputs": x_data.tolist(),
    "expected_outputs": keras_pred.tolist(),
}
validate_instruction_model(instruction_model)
validate_keras_model(model_graph.get_keras(), instruction_model["validation_data"])
```

---

## API Overview

Both backends expose the same model-building API:

| Layer / Op | Description |
|---|---|
| `InputBuffer` | Model input |
| `Dense` | Fully connected layer |
| `Attention` | Attention mechanism |
| `Concatenate` | Concatenate buffers |
| `ReduceSum` | Sum reduction |
| `Add` | Element-wise addition |
| `Multiply` | Element-wise multiplication |
| `MultiplyHeads` | Head-wise broadcast multiply |
| `AddHeads` | Head-wise broadcast add |
| `ScaleVectorized` | Learnable per-element scale |
| `ShiftVectorized` | Learnable per-element shift |
| `SingleIdEmbeddings` | Single-ID embedding lookup |
| `MultiIdEmbeddings` | Multi-ID embedding lookup |
| `ModelGraph` | Compiles the computation graph for training and export |
| `ff_model` | Helper to build a feed-forward stack |
| `validate_model` | Backend-specific validator (alias) |

Backend-specific validators:
- `instmodel.tf.validate_keras_model`
- `instmodel.torch.validate_torch_model`

Backend-agnostic inference:
- `instmodel.instruction_model.instruction_model_inference`
- `instmodel.instruction_model.validate_instruction_model`

---

## GPU Testing

For running PyTorch tests on RTX 50-series GPUs (CUDA 13.1), a custom Dockerfile is provided:

```
custom_cuda_builds/Dockerfile.torch.cuda13
```

---

## License

This project is licensed under the [MIT License](LICENSE).
