"""TensorFlow training backend: a population of independent transformers trained as one XLA step.

Mirrors ``instmodel.sequence.torch``. Every parameter is a variable of
``parameter_shapes`` with a leading members axis, so ``members`` models train
side by side in the same kernels. One ``tf.function`` performs the whole step
(sampling, loss, clipping, AdamW, EMA) without a host round trip.
"""

import math

import numpy as np
import tensorflow as tf

from ._spec import check_parameters, init_parameters

SOFT = ("soft_positions", "soft_tokens", "soft_shares")


def _shifted(mask):
    return tf.concat((tf.zeros_like(mask[..., :1]), mask[..., :-1]), -1)


def _cummax(x):
    """Running maximum over the last axis of values at least -1, as a log-step scan."""
    pad, shift = [[0, 0]] * (x.shape.rank - 1), 1
    while shift < x.shape[-1]:
        x = tf.maximum(x, tf.pad(x[..., :-shift], pad + [[shift, 0]], constant_values=-1))
        shift *= 2
    return x


def lower_bound(ascending, values):
    """``np.searchsorted(ascending, values)`` (side left) as a fixed-depth binary search.

    Under XLA, ``tf.searchsorted`` took time linear in ``len(ascending)`` per value.
    """
    n = ascending.shape[0]
    low, high = tf.zeros_like(values, tf.int64), tf.fill(tf.shape(values), tf.constant(n, tf.int64))
    for _ in range(n.bit_length()):
        middle = (low + high) // 2
        right = (low < high) & (tf.gather(ascending, tf.minimum(middle, n - 1)) < values)
        low, high = tf.where(right, middle + 1, low), tf.where(right, high, middle)
    return low


class Transformer(tf.Module):
    """``members`` independent causal transformers of one ``TransformerConfig``.

    ``named_parameters`` yields the ``parameter_shapes`` names, each float32
    variable with a leading members axis. With ``half`` and a visible GPU the
    matrix products and attention run in float16 while the residual stream,
    norms and softmax stay float32; logits are always float32.
    """

    def __init__(self, config, members=1, seed=0, half=True):
        super().__init__()
        self.config, self.members, self.half = config, members, half
        self.compute = tf.float16 if half and tf.config.list_physical_devices("GPU") else tf.float32
        self._parameters = {name: tf.Variable(value, name=name)
                            for name, value in init_parameters(config, seed, members).items()}
        features = config.features
        self._lookups = []
        for tokens in (features.members, features.openers) if features else ():
            lookup = np.zeros(config.vocab_size, dtype=bool)
            lookup[list(tokens)] = True
            self._lookups.append(lookup)

    def named_parameters(self):
        """``(name, variable)`` pairs in ``parameter_shapes`` order."""
        return list(self._parameters.items())

    @property
    def parameter_count(self):
        """Learned values of one member."""
        return sum(int(np.prod(v.shape)) for v in self._parameters.values()) // self.members

    def segment_features(self, tokens):
        """``(ordinal, index)`` of ``tokens[..., time]``, as ``_spec.segment_features``."""
        tokens = tf.cast(tokens, tf.int64)
        member, opener = (tf.gather(lookup, tokens) for lookup in self._lookups)
        run = member | (opener & ~_shifted(member))
        start = run & ~_shifted(run)
        place = tf.broadcast_to(tf.range(tokens.shape[-1], dtype=tf.int64), tokens.shape)
        index = tf.where(run, place - _cummax(tf.where(start, place, -1)) + 1, 0)
        limit = self.config.features.rows - 1
        return tf.minimum(tf.cumsum(tf.cast(start, tf.int64), -1), limit), tf.minimum(index, limit)

    def _norm(self, x, weight):
        x = x - tf.reduce_mean(x, -1, keepdims=True)
        x = x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), -1, keepdims=True) + self.config.epsilon)
        return x * weight[:, None, 0] + weight[:, None, 1]

    def _dense(self, x, weight, bias=None):
        y = tf.matmul(tf.cast(x, self.compute), tf.cast(weight, self.compute))
        return y if bias is None else y + tf.cast(bias, self.compute)[:, None]

    def __call__(self, tokens):
        """Logits ``[members, batch, time, vocab]`` (float32) for integer ``tokens[members, batch, time]``."""
        c, p = self.config, self._parameters
        tokens = tf.cast(tokens, tf.int64)
        k, b, t = tokens.shape
        if t > c.max_positions:
            raise ValueError("sequence exceeds max_positions")
        x = tf.gather(p["embedding"], tokens, batch_dims=1)
        if c.features is not None:
            ordinal, index = self.segment_features(tokens)
            x = x + tf.gather(p["ordinal"], ordinal, batch_dims=1) + tf.gather(p["index"], index, batch_dims=1)
        return self._layers(tf.reshape(x, (k, b * t, c.width)), k, b, t)

    def _layers(self, x, k, b, t):
        c, p = self.config, self._parameters
        mask = tf.constant(np.triu(np.full((t, t), -np.inf, dtype=np.float32), 1))
        source = x
        for step in range(c.loops):
            if step and c.inject:
                x = x + source
            if c.pass_embed:
                x = x + p["passes"][:, step, None]
            for layer in range(c.layers):
                x = self._block(x, layer, step, mask, k, b, t)
        readout = tf.transpose(p["embedding"], (0, 2, 1)) if c.tied else p["readout"]
        logits = self._dense(self._norm(x, p["norm"]), readout, p["readout_bias"])
        return tf.reshape(tf.cast(logits, tf.float32), (k, b, t, -1))

    def _block(self, x, layer, step, mask, k, b, t):
        c, half = self.config, self.compute
        d, head, kv, row = c.width, c.head_dim, c.kv_heads, step if c.pass_norms else 0
        w = {name.rsplit(".", 1)[1]: v for name, v in self._parameters.items() if name.startswith(f"blocks.{layer}.")}
        hidden = self._norm(x, w["norm1"][:, row])
        qkv = self._dense(hidden, w["qkv"], w["qkv_bias"])
        if c.lora:
            qkv = qkv + self._dense(self._dense(hidden, w["lora_a"][:, step]), w["lora_b"][:, step])
        q, keys, values = tf.split(qkv, (d, kv * head, kv * head), -1)
        # Query head h = kv_head * group + g reads key/value head h // group.
        q = tf.transpose(tf.reshape(q, (k * b, t, kv, c.heads // kv, head)), (0, 2, 3, 1, 4))
        keys, values = (tf.transpose(tf.reshape(z, (k * b, t, kv, head)), (0, 2, 1, 3)) for z in (keys, values))
        scores = tf.einsum("nkgqd,nksd->nkgqs", q, keys) * tf.cast(head ** -0.5, half)
        weights = tf.cast(tf.nn.softmax(tf.cast(scores, tf.float32) + mask), half)
        y = tf.einsum("nkgqs,nksd->nkgqd", weights, values)
        y = tf.reshape(tf.transpose(y, (0, 3, 1, 2, 4)), (k, b * t, d))
        x = x + tf.cast(self._dense(y, w["out"], w["out_bias"]), tf.float32)
        hidden = tf.nn.gelu(self._dense(self._norm(x, w["norm2"][:, row]), w["up"], w["up_bias"]), approximate=False)
        return x + tf.cast(self._dense(hidden, w["down"], w["down_bias"]), tf.float32)

    def member_parameters(self, member=0):
        """One member as float32 NumPy arrays named and shaped as ``parameter_shapes``."""
        return {name: v[member].numpy().copy() for name, v in self._parameters.items()}

    def load_parameters(self, parameters, member=None):
        """Copies one member's arrays into ``member``, or into every member when it is None."""
        check_parameters(self.config, parameters)
        for name, v in self._parameters.items():
            value = np.asarray(parameters[name], dtype=np.float32)
            v.assign(np.broadcast_to(value, v.shape)) if member is None else v[member].assign(value)
        return self

    def select(self, member):
        """One-member ``Transformer`` holding ``member``."""
        single = Transformer(self.config, 1, half=self.half)
        for (_, p), (_, q) in zip(single.named_parameters(), self.named_parameters()):
            p.assign(q[member:member + 1])
        return single


def batch_targets(tokens, groups):
    """``(inputs, targets, groups)`` of rows ``tokens[..., L + 1]`` and ``groups[..., L]``.

    Inputs are ``tokens[..., :-1]``; targets are ``tokens[..., 1:]`` where the
    group is nonnegative and -100 elsewhere.
    """
    tokens, groups = tf.cast(tokens, tf.int64), tf.cast(groups, tf.int64)
    return tokens[..., :-1], tf.where(groups >= 0, tokens[..., 1:], tf.constant(-100, tf.int64)), groups


def sequence_loss(logits, targets, groups, allowed=None, soft_positions=None, soft_tokens=None, soft_shares=None,
                  groups_count=None):
    """Per-row mean of the group-mean token cross entropies, plus exactness flags.

    ``groups[..., time]`` assigns each position a group in ``[0, G)`` or -1
    (ignored). ``allowed[G, vocab]`` masks each position's logits to its
    group's tokens; G is ``allowed.shape[0]``, else ``groups_count``, else
    inferred from ``groups`` (eager only, so pass it in compiled code).
    Absent groups contribute 0 to the mean over G.

    A soft position ``soft_positions[..., F]`` (-1 = unused slot) replaces its
    hard target with ``-sum(share * log p)`` over ``soft_tokens[..., F, B]``
    with ``soft_shares[..., F, B]`` and counts as one token of its own group,
    which must be nonnegative. Returns ``(loss, wrong, follows)``: ``wrong``
    flags a mistaken argmax outside the soft positions; ``follows`` says every
    soft-position argmax equals the row's own target there.
    """
    targets, groups = tf.cast(targets, tf.int64), tf.cast(groups, tf.int64)
    lead, time = targets.shape.rank - 1, targets.shape[-1]
    if allowed is not None:
        allowed = tf.cast(allowed, tf.bool)
        logits = tf.where(tf.gather(allowed, tf.maximum(groups, 0)), logits, -math.inf)
        groups_count = allowed.shape[0]
    elif groups_count is None:
        groups_count = int(tf.reduce_max(groups)) + 1
    follows = tf.ones(tf.shape(targets)[:-1], dtype=tf.bool)
    soft = None
    if soft_positions is not None:
        soft_positions = tf.cast(soft_positions, tf.int64)
        valid = soft_positions >= 0
        position = tf.maximum(soft_positions, 0)
        hit = tf.range(time, dtype=tf.int64) == soft_positions[..., None]  # [..., F, T]
        own = tf.gather(targets, position, batch_dims=lead)
        targets = tf.where(tf.reduce_any(hit, -2), tf.constant(-100, tf.int64), targets)
        picked = tf.gather(logits, position, batch_dims=lead)
        soft_shares = tf.cast(soft_shares, logits.dtype)
        child = (soft_shares > 0) & valid[..., None]
        logp = tf.gather(tf.nn.log_softmax(picked), tf.cast(soft_tokens, tf.int64), batch_dims=lead + 1)
        logp = tf.where(child, logp, 0.)
        soft = tf.reduce_sum(tf.cast(hit, logits.dtype) * -tf.reduce_sum(soft_shares * logp, -1)[..., None], -2)
        follows = tf.reduce_all((tf.argmax(picked, -1) == own) | ~valid, -1)
    ignored = targets == -100
    picked = tf.gather(tf.nn.log_softmax(logits), tf.where(ignored, tf.constant(0, tf.int64), targets),
                       batch_dims=lead + 1)
    flat = tf.where(ignored, 0., -picked)
    if soft is not None:
        flat = flat + soft
    member = tf.cast(groups[..., None] == tf.range(groups_count, dtype=tf.int64), flat.dtype)  # [..., T, G]
    means = tf.reduce_sum(flat[..., None] * member, -2) / tf.maximum(tf.reduce_sum(member, -2), 1)
    wrong = tf.reduce_any((tf.argmax(logits, -1) != targets) & ~ignored, -1)
    return tf.reduce_mean(means, -1), wrong, follows


class Trainer:
    """Whole training step (sampling, loss, per-member clipping, AdamW, EMA) as one compiled function.

    ``rows`` holds arrays or tensors ``tokens[N, L + 1]``, ``groups[N, L]``,
    optionally ``weight[N]`` (sampling weights, uniform by default) and the
    ``sequence_loss`` soft targets ``soft_positions``, ``soft_tokens`` and
    ``soft_shares``. Each step draws ``batch`` rows per member. ``loss`` holds
    the last per-member losses. With ``compile`` the step is XLA-compiled.
    AdamW matches ``torch.optim.AdamW`` (decoupled decay, bias-corrected
    moments, ``eps = 1e-8``).
    """

    def __init__(self, model, rows, batch, lr, ema=0.999, allowed=None, compile=True, betas=(0.9, 0.98),
                 weight_decay=0.01, seed=0):
        self.model, self.batch, self.ema, self.betas, self.weight_decay = model, batch, ema, betas, weight_decay
        self.tokens = tf.convert_to_tensor(np.asarray(rows["tokens"]), tf.int64)
        self.groups = tf.convert_to_tensor(np.asarray(rows["groups"]), tf.int64)
        self.soft = tuple(tf.convert_to_tensor(np.asarray(rows[name])) for name in SOFT) if SOFT[0] in rows else ()
        weight = rows.get("weight")
        cdf = np.cumsum(np.ones(len(rows["tokens"])) if weight is None else np.asarray(weight, dtype=np.float64))
        self.cdf = tf.constant(cdf / cdf[-1], tf.float64)
        self.allowed = None if allowed is None else tf.cast(allowed, tf.bool)
        self.count = allowed.shape[0] if allowed is not None else int(tf.reduce_max(self.groups)) + 1
        self.parameters = [v for _, v in model.named_parameters()]
        self.lr = tf.Variable(lr, dtype=tf.float32, trainable=False)
        self.steps = tf.Variable(0., trainable=False)
        self.moments = [[tf.Variable(tf.zeros_like(p), trainable=False) for p in self.parameters] for _ in range(2)]
        self.average = [tf.Variable(p, trainable=False) for p in self.parameters]
        self.loss = tf.Variable(tf.zeros(model.members), trainable=False)
        self.generator = tf.random.Generator.from_seed(seed)
        self._sampled = tf.function(lambda: self._update(self._draw()), jit_compile=compile)
        self._step_rows = tf.function(self._update, jit_compile=compile)  # Fixed rows [members, batch], for tests.

    def _draw(self):
        draw = self.generator.uniform((self.model.members, self.batch), dtype=tf.float64)
        return tf.minimum(lower_bound(self.cdf, draw), self.tokens.shape[0] - 1)

    def _update(self, index):
        x, y, g = batch_targets(tf.gather(self.tokens, index), tf.gather(self.groups, index))
        soft = [tf.gather(t, index) for t in self.soft]
        with tf.GradientTape() as tape:
            losses = sequence_loss(self.model(x), y, g, self.allowed, *soft, groups_count=self.count)[0]
            loss = tf.reduce_mean(losses, -1)
            total = tf.reduce_sum(loss)
        grads = [tf.zeros_like(p) if q is None else tf.convert_to_tensor(q)
                 for p, q in zip(self.parameters, tape.gradient(total, self.parameters))]
        members = self.model.members
        norms = tf.sqrt(tf.add_n([tf.reduce_sum(tf.reshape(tf.square(q), (members, -1)), -1) for q in grads]))
        scale = tf.minimum(1 / (norms + 1e-6), 1)
        lr, (b1, b2) = self.lr, self.betas
        self.steps.assign_add(1)
        step_size = lr / (1 - b1 ** self.steps)
        root = tf.sqrt(1 - b2 ** self.steps)
        for p, q, m, v, a in zip(self.parameters, grads, *self.moments, self.average):
            q = q * tf.reshape(scale, (-1,) + (1,) * (q.shape.rank - 1))
            p.assign(p * (1 - lr * self.weight_decay))
            m.assign(m + (1 - b1) * (q - m))
            v.assign(v * b2 + (1 - b2) * q * q)
            p.assign_sub(step_size * m / (tf.sqrt(v) / root + 1e-8))
            a.assign_add((1 - self.ema) * (p - a))
        self.loss.assign(loss)

    def step(self, lr):
        """One training step at learning rate ``lr``."""
        self.lr.assign(lr)
        self._sampled()

    def averaged(self):
        """``Transformer`` holding the parameter EMA of every member."""
        model = Transformer(self.model.config, self.model.members, half=self.model.half)
        for (_, p), a in zip(model.named_parameters(), self.average):
            p.assign(a)
        return model


def schedule(fraction, peak, warmup=0.03, floor=0.02):
    """Learning rate at ``fraction`` of training: linear warmup, then cosine decay to ``floor * peak``."""
    if fraction < warmup:
        return peak * fraction / warmup
    progress = (fraction - warmup) / (1 - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.))))
