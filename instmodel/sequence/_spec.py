"""Causal transformer definition shared by the training backends and the step-program export.

A model is a dict of named float32 arrays (see ``parameter_shapes``). Training
backends hold the same arrays with a leading members axis and run them over
whole sequences. ``export_step_model`` turns one member into a format-v2
instruction model: the instruction list processes one token, and the state
declared under ``"sequence"`` carries the attention caches and feature scanner
from one token to the next.
"""

from dataclasses import asdict, dataclass

import numpy as np

FORMAT_VERSION = 2


@dataclass(frozen=True)
class SegmentFeatures:
    """Per-token ``(ordinal, index)`` of runs of tokens, each with a learned embedding table.

    A token belongs to a run when it is in ``members``, or in ``openers`` and
    the previous token is not a member (a sign before a digit). ``ordinal``
    counts the runs started so far; ``index`` is the one-based place inside
    the current run and 0 outside a run. Both clamp to ``rows - 1``.
    """

    members: tuple
    openers: tuple = ()
    rows: int = 16

    def __post_init__(self):
        object.__setattr__(self, "members", tuple(int(t) for t in self.members))
        object.__setattr__(self, "openers", tuple(int(t) for t in self.openers))
        if self.rows < 2 or not self.members:
            raise ValueError("segment features need members and at least two rows")
        if set(self.members) & set(self.openers):
            raise ValueError("a token cannot be both a member and an opener")


@dataclass(frozen=True)
class TransformerConfig:
    """Pre-norm causal transformer.

    ``heads`` query heads share ``kv_heads`` key/value heads (grouped-query
    attention; query head ``h`` uses key/value head ``h // (heads // kv_heads)``).
    With ``loops`` above one the stack of ``layers`` blocks runs several times
    with shared weights; ``inject`` re-adds the input embedding before each
    later pass, ``pass_embed`` adds a learned vector per pass, ``pass_norms``
    gives every pass its own layer-norm parameters, and ``lora`` is the rank
    of a per-pass correction to each block's query/key/value projection.
    """

    vocab_size: int
    width: int
    heads: int
    layers: int = 2
    ff_width: int = 0
    kv_heads: int = 0
    loops: int = 1
    inject: bool = False
    pass_embed: bool = False
    pass_norms: bool = False
    lora: int = 0
    tied: bool = True
    features: SegmentFeatures | None = None
    max_positions: int = 192
    epsilon: float = 1e-5

    def __post_init__(self):
        if isinstance(self.features, dict):
            object.__setattr__(self, "features", SegmentFeatures(**self.features))
        object.__setattr__(self, "ff_width", self.ff_width or self.width)
        object.__setattr__(self, "kv_heads", self.kv_heads or self.heads)
        if self.vocab_size < 2 or self.width < 1 or self.layers < 1 or self.loops < 1:
            raise ValueError("vocab_size, width, layers and loops must be positive")
        if self.width % self.heads or self.heads % self.kv_heads:
            raise ValueError("width must divide into heads, and heads into kv_heads")
        if self.lora < 0 or self.max_positions < 1 or self.ff_width < 1:
            raise ValueError("lora, max_positions and ff_width must be valid sizes")
        if self.features is not None and max(self.features.members + self.features.openers) >= self.vocab_size:
            raise ValueError("segment feature tokens must be vocabulary ids")

    @property
    def head_dim(self):
        return self.width // self.heads

    @property
    def qkv_size(self):
        return self.width + 2 * self.kv_heads * self.head_dim

    @property
    def norm_passes(self):
        return self.loops if self.pass_norms else 1

    def to_dict(self):
        return asdict(self)


def parameter_shapes(config):
    """Name -> shape of every learned array of one member; linear maps are ``[input, output]``."""
    c, d = config, config.width
    shapes = {"embedding": (c.vocab_size, d)}
    if c.features is not None:
        shapes["ordinal"] = (c.features.rows, d)
        shapes["index"] = (c.features.rows, d)
    if c.pass_embed:
        shapes["passes"] = (c.loops, d)
    for layer in range(c.layers):
        prefix = f"blocks.{layer}."
        shapes[prefix + "norm1"] = (c.norm_passes, 2, d)  # Row 0 is the gain, row 1 the bias.
        shapes[prefix + "qkv"] = (d, c.qkv_size)
        shapes[prefix + "qkv_bias"] = (c.qkv_size,)
        if c.lora:
            shapes[prefix + "lora_a"] = (c.loops, d, c.lora)
            shapes[prefix + "lora_b"] = (c.loops, c.lora, c.qkv_size)
        shapes[prefix + "out"] = (d, d)
        shapes[prefix + "out_bias"] = (d,)
        shapes[prefix + "norm2"] = (c.norm_passes, 2, d)
        shapes[prefix + "up"] = (d, c.ff_width)
        shapes[prefix + "up_bias"] = (c.ff_width,)
        shapes[prefix + "down"] = (c.ff_width, d)
        shapes[prefix + "down_bias"] = (d,)
    shapes["norm"] = (2, d)
    if not c.tied:
        shapes["readout"] = (d, c.vocab_size)
    shapes["readout_bias"] = (c.vocab_size,)
    return shapes


def parameter_count(config):
    return sum(int(np.prod(shape)) for shape in parameter_shapes(config).values())


def init_parameters(config, seed=0, members=None):
    """Initial float32 parameters; with ``members`` every array gains a leading members axis.

    Tables and LoRA inputs are normal with scale ``width ** -0.5``; linear maps
    are uniform in ``±1/sqrt(input)``; norms start as identity; biases, pass
    vectors and LoRA outputs start at zero, so every pass starts shared.
    """
    rng = np.random.default_rng(seed)
    lead = () if members is None else (members,)
    d, out = config.width, {}
    for name, shape in parameter_shapes(config).items():
        leaf = name.rsplit(".", 1)[-1]
        full = lead + shape
        if leaf in ("embedding", "ordinal", "index", "lora_a"):
            value = rng.standard_normal(full) * d ** -0.5
        elif leaf in ("norm1", "norm2", "norm"):
            value = np.zeros(full)
            value[..., 0, :] = 1
        elif leaf in ("qkv", "out", "up", "down", "readout"):
            bound = shape[0] ** -0.5
            value = rng.uniform(-bound, bound, full)
        else:
            value = np.zeros(full)
        out[name] = value.astype(np.float32)
    return out


def check_parameters(config, parameters):
    shapes = parameter_shapes(config)
    if set(parameters) != set(shapes):
        raise ValueError(f"parameter names differ: {sorted(set(parameters) ^ set(shapes))}")
    for name, shape in shapes.items():
        if tuple(np.shape(parameters[name])) != shape:
            raise ValueError(f"{name} has shape {np.shape(parameters[name])}, expected {shape}")


def segment_features(tokens, features):
    """NumPy ``(ordinal, index)`` for token arrays shaped ``[..., time]``."""
    tokens = np.asarray(tokens)

    def shifted(mask):
        return np.concatenate((np.zeros_like(mask[..., :1]), mask[..., :-1]), -1)

    member = np.isin(tokens, features.members)
    run = member | (np.isin(tokens, features.openers) & ~shifted(member))
    start = run & ~shifted(run)
    place = np.broadcast_to(np.arange(tokens.shape[-1]), tokens.shape)
    last = np.maximum.accumulate(np.where(start, place, -1), -1)
    index = np.where(run, place - last + 1, 0)
    ordinal = np.cumsum(start, -1)
    limit = features.rows - 1
    return np.minimum(ordinal, limit), np.minimum(index, limit)


class _Program:
    """Accumulates buffers, instructions and deduplicated weight and parameter entries."""

    def __init__(self):
        self.model = dict(buffer_sizes=[], instructions=[], maps=[], weights=[], bias=[], parameters=[])
        self._entries = {}

    def buffer(self, size):
        self.model["buffer_sizes"].append(int(size))
        return len(self.model["buffer_sizes"]) - 1

    def emit(self, **instruction):
        self.model["instructions"].append(instruction)
        return instruction.get("output")

    def weights(self, key, matrix, bias=None):
        """Index of an ``[output][input]`` matrix, stored once per ``key``."""
        if key not in self._entries:
            matrix = np.asarray(matrix, dtype=np.float32)
            bias = np.zeros(len(matrix), dtype=np.float32) if bias is None else np.asarray(bias, dtype=np.float32)
            self.model["weights"].append(matrix.tolist())
            self.model["bias"].append(bias.tolist())
            self._entries[key] = len(self.model["weights"]) - 1
        return self._entries[key]

    def parameters(self, key, vector):
        key = ("parameters", key)
        if key not in self._entries:
            self.model["parameters"].append(np.asarray(vector, dtype=np.float32).tolist())
            self._entries[key] = len(self.model["parameters"]) - 1
        return self._entries[key]


def export_step_model(config, parameters, vocabulary=None, metadata=None):
    """Format-v2 instruction model that processes one token of one member.

    Buffer 0 holds the token id and the last buffer the next-token logits.
    Looped passes are unrolled; their instructions reference the same weight
    entries, and every attention instruction owns one key/value cache slot.
    """
    check_parameters(config, parameters)
    c, p, d = config, parameters, config.width
    program = _Program()
    token = program.buffer(1)

    def linear(name, source, size, bias=None, activation=None, key=None):
        index = program.weights(key or name, np.asarray(p[name]).T, None if bias is None else p[bias])
        instruction = dict(type="DOT", input=source, output=program.buffer(size), weights=index)
        if activation:
            instruction["activation"] = activation
        return program.emit(**instruction)

    def norm(name, values, source):
        gain = program.parameters(name + ".gain", values[0])
        bias = program.parameters(name + ".bias", values[1])
        return program.emit(type="LAYER_NORM", input=source, output=program.buffer(d), gain=gain, bias=bias,
                            epsilon=c.epsilon)

    def add(*sources):
        return program.emit(type="ADD_ELEMENTWISE_BUFFERS", input=list(sources), output=program.buffer(d))

    # The token table doubles as the tied readout matrix, whose bias is the readout bias.
    table = program.weights("embedding", p["embedding"], p["readout_bias"] if c.tied else None)
    embedded = [program.emit(type="EMBEDDING", input=token, output=program.buffer(d), weights=table)]
    if c.features is not None:
        scanned = program.emit(type="SEGMENT_FEATURES", input=token, output=program.buffer(2),
                               members=list(c.features.members), openers=list(c.features.openers),
                               rows=c.features.rows, scanner=0)
        for column, name in enumerate(("ordinal", "index")):
            embedded.append(program.emit(type="EMBEDDING", input=scanned, internal_input_index=column,
                                         output=program.buffer(d), weights=program.weights(name, p[name])))
    source = add(*embedded) if len(embedded) > 1 else embedded[0]

    x, slots = source, []
    for step in range(c.loops):
        if step and c.inject:
            x = add(x, source)
        if c.pass_embed:
            if x == source:  # The embedding stays intact for later injection.
                x = program.emit(type="COPY", input=source, output=program.buffer(d), internal_index=0)
            program.emit(type="ADD_ELEMENTWISE", input=x,
                         parameters=program.parameters(f"passes.{step}", p["passes"][step]))
        row = step if c.pass_norms else 0
        for layer in range(c.layers):
            prefix = f"blocks.{layer}."
            hidden = norm(f"{prefix}norm1.{row}", p[prefix + "norm1"][row], x)
            qkv = linear(prefix + "qkv", hidden, c.qkv_size, prefix + "qkv_bias")
            if c.lora:
                low = program.emit(type="DOT", input=hidden, output=program.buffer(c.lora),
                                   weights=program.weights(f"{prefix}lora_a.{step}", p[prefix + "lora_a"][step].T))
                delta = program.emit(type="DOT", input=low, output=program.buffer(c.qkv_size),
                                     weights=program.weights(f"{prefix}lora_b.{step}", p[prefix + "lora_b"][step].T))
                qkv = program.emit(type="ADD_ELEMENTWISE_BUFFERS", input=[qkv, delta],
                                   output=program.buffer(c.qkv_size))
            attended = program.emit(type="CAUSAL_ATTENTION", input=qkv, output=program.buffer(d),
                                    heads=c.heads, kv_heads=c.kv_heads, slot=len(slots))
            slots.append(dict(heads=c.heads, kv_heads=c.kv_heads, head_dim=c.head_dim))
            x = add(x, linear(prefix + "out", attended, d, prefix + "out_bias"))
            hidden = norm(f"{prefix}norm2.{row}", p[prefix + "norm2"][row], x)
            raised = linear(prefix + "up", hidden, c.ff_width, prefix + "up_bias", activation="GELU")
            x = add(x, linear(prefix + "down", raised, d, prefix + "down_bias"))
    final = norm("norm", p["norm"], x)
    if c.tied:
        program.emit(type="DOT", input=final, output=program.buffer(c.vocab_size), weights=table)
    else:
        linear("readout", final, c.vocab_size, "readout_bias")

    model = program.model
    model["format_version"] = FORMAT_VERSION
    model["features"] = ["token"]
    model["sequence"] = dict(
        vocabulary=list(vocabulary) if vocabulary is not None else [str(i) for i in range(c.vocab_size)],
        max_positions=c.max_positions, kv_slots=slots, scanners=int(c.features is not None),
        config=c.to_dict(), metadata=dict(metadata or {}))
    if len(model["sequence"]["vocabulary"]) != c.vocab_size:
        raise ValueError("vocabulary length differs from vocab_size")
    return model
