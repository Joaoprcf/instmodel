# instmodel

This repo is meant to be used with coding agents, not read manually.

Build and train small neural networks with PyTorch or TensorFlow/Keras through one shared graph API,
then export them to a compact JSON "instruction model" that runs anywhere with plain NumPy (or the
Rust/C++ inference ports) — no training framework needed at inference time.

## Install

```bash
pip install instmodel              # NumPy inference only
pip install instmodel[pytorch]     # + PyTorch training backend (instmodel.torch)
pip install instmodel[tensorflow]  # + TensorFlow/Keras training backend (instmodel.tf)
```

## Example

```python
import numpy as np
from instmodel.torch import Dense, InputBuffer, ModelGraph  # same API in instmodel.tf
from instmodel.instruction_model import instruction_model_inference

x = InputBuffer(4)
y = Dense(1, activation="sigmoid")(Dense(8, activation="relu")(x))
model = ModelGraph(x, y)
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(np.random.rand(64, 4), np.random.randint(0, 2, (64, 1)), epochs=5, verbose=0)

exported = model.create_instruction_model()  # JSON-serialisable dict
print(instruction_model_inference(exported, np.random.rand(3, 4))[-1])
```

## License

[MIT](LICENSE)
