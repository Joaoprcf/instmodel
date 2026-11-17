"""PyTorch training backend: a population of independent transformers trained as one CUDA graph.

Every parameter is an array of ``parameter_shapes`` with a leading members
axis, so ``members`` models train side by side in the same kernels. On CUDA
the whole step (sampling, loss, clipping, AdamW, EMA) is captured once and
replayed; on CPU it runs eagerly.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ._spec import check_parameters, init_parameters, parameter_shapes

SOFT = ("soft_positions", "soft_tokens", "soft_shares")


class Transformer(nn.Module):
    """``members`` independent causal transformers of one ``TransformerConfig``.

    ``named_parameters`` yields the ``parameter_shapes`` names, each array
    with a leading members axis. With ``bf16`` the layers run under bfloat16
    autocast on CUDA; logits are always float32.
    """

    def __init__(self, config, members=1, seed=0, bf16=True):
        super().__init__()
        self.config, self.members, self.bf16 = config, members, bf16
        self.blocks = nn.ModuleList(nn.ParameterDict() for _ in range(config.layers))
        for name, value in init_parameters(config, seed, members).items():
            parameter = nn.Parameter(torch.from_numpy(value))
            if name.startswith("blocks."):
                _, layer, leaf = name.split(".")
                self.blocks[int(layer)][leaf] = parameter
            else:
                self.register_parameter(name, parameter)
        features = config.features
        for name, tokens in (("member_ids", features.members if features else ()),
                             ("opener_ids", features.openers if features else ())):
            lookup = torch.zeros(config.vocab_size, dtype=torch.bool)
            lookup[list(tokens)] = True
            self.register_buffer(name, lookup, persistent=False)

    @property
    def parameter_count(self):
        """Learned values of one member."""
        return sum(p.numel() for p in self.parameters()) // self.members

    def segment_features(self, tokens):
        """``(ordinal, index)`` of ``tokens[..., time]``, as ``_spec.segment_features``."""
        def shifted(mask):
            return torch.cat((torch.zeros_like(mask[..., :1]), mask[..., :-1]), -1)

        member = self.member_ids[tokens]
        run = member | (self.opener_ids[tokens] & ~shifted(member))
        start = run & ~shifted(run)
        place = torch.arange(tokens.shape[-1], device=tokens.device).expand_as(tokens)
        last = torch.where(start, place, -1).cummax(-1).values
        index = torch.where(run, place - last + 1, 0)
        limit = self.config.features.rows - 1
        return start.long().cumsum(-1).clamp_max(limit), index.clamp_max(limit)

    def _norm(self, x, weight):
        return F.layer_norm(x, x.shape[-1:], eps=self.config.epsilon) * weight[:, None, 0] + weight[:, None, 1]

    def forward(self, tokens):
        """Logits ``[members, batch, time, vocab]`` (float32) for long ``tokens[members, batch, time]``."""
        c = self.config
        k, b, t = tokens.shape
        if t > c.max_positions:
            raise ValueError("sequence exceeds max_positions")
        member = torch.arange(k, device=tokens.device)[:, None, None]
        if c.features is None:
            x = F.embedding(tokens + member * c.vocab_size, self.embedding.flatten(0, 1))
        else:
            ordinal, index = self.segment_features(tokens)
            tables = torch.cat((self.embedding, self.ordinal, self.index), 1)
            ids = torch.stack((tokens, ordinal + c.vocab_size, index + c.vocab_size + c.features.rows), -1)
            x = F.embedding(ids + member[..., None] * tables.shape[1], tables.flatten(0, 1)).sum(-2)
        with torch.autocast("cuda", torch.bfloat16, enabled=tokens.is_cuda and self.bf16):
            return self._layers(x.reshape(k, b * t, -1), k, b, t)

    def _layers(self, x, k, b, t):
        c = self.config
        source = x
        for step in range(c.loops):
            if step and c.inject:
                x = x + source
            if c.pass_embed:
                x = x + self.passes[:, step, None]
            for block in self.blocks:
                x = self._block(x, block, step, k, b, t)
        readout = self.embedding.transpose(1, 2) if c.tied else self.readout
        logits = torch.baddbmm(self.readout_bias[:, None], self._norm(x, self.norm), readout)
        return logits.float().reshape(k, b, t, -1)

    def _block(self, x, block, step, k, b, t):
        c = self.config
        d, head, row = c.width, c.head_dim, step if c.pass_norms else 0
        hidden = self._norm(x, block["norm1"][:, row])
        qkv = torch.baddbmm(block["qkv_bias"][:, None], hidden, block["qkv"])
        if c.lora:
            qkv = qkv + torch.bmm(torch.bmm(hidden, block["lora_a"][:, step]), block["lora_b"][:, step])
        q, kk, v = qkv.split((d, c.kv_heads * head, c.kv_heads * head), -1)
        q = q.reshape(k * b, t, c.heads, head).transpose(1, 2)
        kk, v = (z.reshape(k * b, t, c.kv_heads, head).transpose(1, 2) for z in (kk, v))
        y = F.scaled_dot_product_attention(q, kk, v, is_causal=True, enable_gqa=True)
        x = x + torch.baddbmm(block["out_bias"][:, None], y.transpose(1, 2).reshape(k, b * t, d), block["out"])
        hidden = F.gelu(torch.baddbmm(block["up_bias"][:, None], self._norm(x, block["norm2"][:, row]), block["up"]))
        return x + torch.baddbmm(block["down_bias"][:, None], hidden, block["down"])

    def member_parameters(self, member=0):
        """One member as float32 NumPy arrays named and shaped as ``parameter_shapes``."""
        return {name: p[member].detach().float().cpu().numpy().copy() for name, p in self.named_parameters()}

    @torch.no_grad()
    def load_parameters(self, parameters, member=None):
        """Copies one member's arrays into ``member``, or into every member when it is None."""
        check_parameters(self.config, parameters)
        for name, p in self.named_parameters():
            value = torch.as_tensor(np.asarray(parameters[name]), dtype=p.dtype, device=p.device)
            (p if member is None else p[member]).copy_(value)
        return self

    @torch.no_grad()
    def select(self, member):
        """One-member ``Transformer`` on the same device holding ``member``."""
        device = self.embedding.device
        single = Transformer(self.config, 1, bf16=self.bf16).to(device)
        for p, q in zip(single.parameters(), self.parameters()):
            p.copy_(q[member:member + 1])
        return single


def batch_targets(tokens, groups):
    """``(inputs, targets, groups)`` of rows ``tokens[..., L + 1]`` and ``groups[..., L]``.

    Inputs are ``tokens[..., :-1]``; targets are ``tokens[..., 1:]`` where the
    group is nonnegative and -100 elsewhere.
    """
    tokens, groups = tokens.long(), groups.long()
    return tokens[..., :-1], torch.where(groups >= 0, tokens[..., 1:], -100), groups


def sequence_loss(logits, targets, groups, allowed=None, soft_positions=None, soft_tokens=None, soft_shares=None,
                  groups_count=None):
    """Per-row mean of the group-mean token cross entropies, plus exactness flags.

    ``groups[..., time]`` assigns each position a group in ``[0, G)`` or -1
    (ignored). ``allowed[G, vocab]`` masks each position's logits to its
    group's tokens; G is ``allowed.shape[0]``, else ``groups_count``, else
    inferred from ``groups`` (a host sync, so pass it in captured code).
    Absent groups contribute 0 to the mean over G.

    A soft position ``soft_positions[..., F]`` (-1 = unused slot) replaces its
    hard target with ``-sum(share * log p)`` over ``soft_tokens[..., F, B]``
    with ``soft_shares[..., F, B]`` and counts as one token of its own group,
    which must be nonnegative. Returns ``(loss, wrong, follows)``: ``wrong``
    flags a mistaken argmax outside the soft positions; ``follows`` says every
    soft-position argmax equals the row's own target there.
    """
    if allowed is not None:
        logits = logits.masked_fill(~allowed[groups.clamp_min(0)], -torch.inf)
        groups_count = allowed.shape[0]
    elif groups_count is None:
        groups_count = int(groups.max()) + 1
    follows = torch.ones(targets.shape[:-1], dtype=torch.bool, device=targets.device)
    soft = None
    if soft_positions is not None:
        valid = soft_positions >= 0
        position = soft_positions.long().clamp_min(0)
        hit = torch.arange(targets.shape[-1], device=targets.device) == soft_positions[..., None]  # [..., F, T]
        own = targets.gather(-1, position)
        targets = targets.masked_fill(hit.any(-2), -100)
        picked = logits.gather(-2, position[..., None].expand(*position.shape, logits.shape[-1]))
        child = (soft_shares > 0) & valid[..., None]
        logp = picked.log_softmax(-1).gather(-1, soft_tokens.long()).masked_fill(~child, 0)
        soft = (hit * -(soft_shares * logp).sum(-1)[..., None]).sum(-2)
        follows = ((picked.argmax(-1) == own) | ~valid).all(-1)
    flat = F.cross_entropy(logits.flatten(0, -2), targets.flatten(), reduction="none").reshape_as(targets)
    if soft is not None:
        flat = flat + soft
    member = groups[..., None] == torch.arange(groups_count, device=groups.device)  # [..., T, G]
    means = (flat[..., None] * member).sum(-2) / member.sum(-2).clamp_min(1)
    wrong = ((logits.argmax(-1) != targets) & (targets != -100)).any(-1)
    return means.mean(-1), wrong, follows


class Trainer:
    """Whole training step (sampling, loss, per-member clipping, AdamW, EMA) replayed as one CUDA graph.

    ``rows`` holds device tensors ``tokens[N, L + 1]``, ``groups[N, L]``,
    optionally ``weight[N]`` (sampling weights, uniform by default) and the
    ``sequence_loss`` soft targets ``soft_positions``, ``soft_tokens`` and
    ``soft_shares``. Each step draws ``batch`` rows per member. ``loss`` holds
    the last per-member losses. On CPU the step runs eagerly.
    """

    def __init__(self, model, rows, batch, lr, ema=0.999, allowed=None, compile=False, betas=(0.9, 0.98),
                 weight_decay=0.01):
        tokens = rows["tokens"]
        device = tokens.device
        self.model, self.tokens, self.groups, self.batch, self.ema = model, tokens, rows["groups"], batch, ema
        self.soft = tuple(rows[name] for name in SOFT) if SOFT[0] in rows else ()
        weight = rows.get("weight")
        self.cdf = (torch.ones(len(tokens), device=device) if weight is None else weight).double().cumsum(0)
        self.cdf /= self.cdf[-1].clone()
        self.allowed = None if allowed is None else allowed.to(device, torch.bool)
        count = allowed.shape[0] if allowed is not None else int(self.groups.max()) + 1
        self.parameters = list(model.parameters())
        self.lr = torch.tensor(lr, device=device)
        self.optimizer = torch.optim.AdamW(self.parameters, lr=self.lr, betas=betas, weight_decay=weight_decay,
                                           fused=True, capturable=True)
        self.average = [p.detach().clone() for p in self.parameters]

        def objective(x, y, g, *soft):
            return sequence_loss(model(x), y, g, self.allowed, *soft, groups_count=count)[0].mean(-1)

        # Default Inductor mode: the surrounding manual CUDA graph replaces cudagraph trees.
        self.objective = torch.compile(objective, fullgraph=True, dynamic=False) if compile else objective
        self.loss = torch.zeros(model.members, device=device)
        self.graph = None
        if device.type != "cuda":
            return
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        state = [p.detach().clone() for p in self.parameters]
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.graph):
            self._step()
        with torch.no_grad():  # Warmup updates are discarded; Adam moments restart.
            for p, s, a in zip(self.parameters, state, self.average):
                p.copy_(s), a.copy_(s)
            for value in self.optimizer.state.values():
                for tensor in value.values():
                    tensor.zero_()

    def _step(self):
        draw = torch.rand((self.model.members, self.batch), device=self.tokens.device, dtype=torch.float64)
        index = torch.searchsorted(self.cdf, draw).clamp_max(len(self.tokens) - 1)
        x, y, g = batch_targets(self.tokens[index], self.groups[index])
        loss = self.objective(x, y, g, *(t[index] for t in self.soft))
        self.optimizer.zero_grad(set_to_none=False) if self.parameters[0].grad is not None else None
        loss.sum().backward()
        with torch.no_grad():
            norms = sum(p.grad.flatten(1).pow(2).sum(-1) for p in self.parameters).sqrt()
            scale = (1 / (norms + 1e-6)).clamp_max(1)
            for p in self.parameters:
                p.grad.mul_(scale.reshape(-1, *[1] * (p.dim() - 1)))
        self.optimizer.step()
        with torch.no_grad():
            torch._foreach_lerp_(self.average, [p.detach() for p in self.parameters], 1 - self.ema)
            self.loss.copy_(loss.detach())

    def step(self, lr, steps=1):
        """``steps`` training steps at learning rate ``lr``."""
        self.lr.fill_(lr)
        for _ in range(steps):
            self.graph.replay() if self.graph is not None else self._step()

    @torch.no_grad()
    def averaged(self):
        """``Transformer`` holding the parameter EMA of every member."""
        model = Transformer(self.model.config, self.model.members, bf16=self.model.bf16).to(self.tokens.device)
        for p, a in zip(model.parameters(), self.average):
            p.copy_(a)
        return model


def schedule(fraction, peak, warmup=0.03, floor=0.02):
    """Learning rate at ``fraction`` of training: linear warmup, then cosine decay to ``floor * peak``."""
    if fraction < warmup:
        return peak * fraction / warmup
    progress = (fraction - warmup) / (1 - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.))))
