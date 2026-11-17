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

## Sequence models

`instmodel.sequence` trains small causal transformers and exports them as an instruction model
that processes one token per run. The attention caches are explicit state owned by the caller,
so any runtime can decode token by token (NumPy here, or `SequenceModel` in the Rust port).

```python
import numpy as np
from instmodel import sequence
from instmodel.sequence import torch as backend  # same API in instmodel.sequence.tf

config = sequence.TransformerConfig(vocab_size=16, width=16, heads=4, kv_heads=2)
model = backend.Transformer(config, members=2)   # two copies trained in one batch

start = np.random.randint(0, 16, (4096, 1))
tokens = (start + np.arange(13)) % 16            # rows [N, time + 1]; learn "next = token + 1"
rows = dict(tokens=backend.torch.as_tensor(tokens), groups=backend.torch.zeros(4096, 12, dtype=int))
trainer = backend.Trainer(model, rows, batch=64, lr=3e-3, ema=0.99)
for step in range(400):
    trainer.step(backend.schedule(step / 400, 3e-3))

exported = sequence.export_step_model(config, trainer.averaged().member_parameters(0))
session = sequence.SequenceSession(exported)     # keeps the key/value cache between steps
print(session.step([3]).argmax(-1))              # -> [4]
```

`TransformerConfig` also covers grouped-query attention, looped (weight-shared) passes and
`SegmentFeatures`; `sequence_loss` takes per-position groups, allowed-token masks and soft targets.
`add_validation_sequences` stores reference logits in the file for other runtimes to replay.

## License

[MIT](LICENSE)
