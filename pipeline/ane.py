"""Convert the selected taggers to Core ML for the Apple Neural Engine: `uv run convert-ane`.

Each task is one package with the fixed length of the token limit. Several units share a sequence:
the inputs are input_ids, position_ids (restart at 0 for each unit) and segment_ids (0 for
padding, 1 for the first unit, ...). The model builds the block-diagonal mask from the segments
and keeps the local window inside a unit. The layout follows apple/ml-ane-transformers: (B, C, 1,
S) tensors, 1x1 Conv2d for Linear, layer norm over the channel axis, one einsum attention per
head. The activations are fp16; the weights are int8, with the embedding table in int4. The
logits are (1, tags, 1, 512): 7 tags for the question task, 3 for the option task. The output
directory is self-contained: the tokenizer files and settings.json hold what a reader needs,
with the digests of the checkpoint, the packages and the source code.
"""

import argparse
import hashlib
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from jsonl import DATA, file_digest, sha256, write_json
from model import CHECKPOINT, TASKS, load, load_tokenizer
from tagging import CLOSING, MAX_TOKENS, OPENING, OPTION_TRAILING, TAG_NAMES
from units import DECISION

# external: coremltools warns on import that the installed torch is newer than the tested one
logging.getLogger("coremltools").setLevel(logging.ERROR)
# external: torch.jit.trace makes torch.distributed log that redirects are unsupported on macOS
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import coremltools as ct  # noqa: E402

COREML_ANE = DATA / "coreml-ane-pack"
MASKED = -10000.0
INPUTS = ("input_ids", "position_ids", "segment_ids")


def compress(model):
    """The weights as int8 per channel, and the embedding table as int4 per block of 32. The
    packages need macOS 15, the deployment target."""
    linear = ct.optimize.coreml.OpLinearQuantizerConfig
    int8 = linear(mode="linear_symmetric", dtype="int8", granularity="per_channel")
    int4 = linear(mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=32)
    config = ct.optimize.coreml.OptimizationConfig(
        global_config=int8, op_type_configs={"gather": int4}
    )
    return ct.optimize.coreml.linear_quantize_weights(model, config)


def directory_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()[:16]


def source_digest() -> str:
    sources = sorted(Path(__file__).parent.glob("*.py"))
    return sha256("".join(f"{p.name}:{sha256(p.read_text())}\n" for p in sources))


class Norm(nn.Module):
    """Layer norm over the channel axis, without bias. The deviation is scaled by 1/16 (exact in
    fp16) before it is squared, so the square cannot overflow."""

    def __init__(self, norm: nn.LayerNorm) -> None:
        super().__init__()
        self.weight = nn.Parameter(norm.weight.detach().clone().view(1, -1, 1, 1))
        self.eps = norm.eps

    def forward(self, x):
        x = x - x.mean(1, keepdim=True)
        small = x * 0.0625
        variance = (small * small).mean(1, keepdim=True) * 256.0
        return x * torch.rsqrt(variance + self.eps) * self.weight


def conv(weight: torch.Tensor, bias: torch.Tensor | None = None) -> nn.Conv2d:
    out = nn.Conv2d(weight.shape[1], weight.shape[0], 1, bias=bias is not None)
    out.weight.data = weight.detach()[:, :, None, None].clone()
    if bias is not None:
        out.bias.data = bias.detach().clone()
    return out


class Layer(nn.Module):
    def __init__(self, layer, heads: int, head_dim: int) -> None:
        super().__init__()
        size = heads * head_dim
        self.heads, self.half = heads, head_dim // 2
        self.local = layer.attention_type == "sliding_attention"
        self.attn_norm = None if layer.layer_idx == 0 else Norm(layer.attn_norm)
        query, key, value = layer.attn.Wqkv.weight.detach().split(size)
        self.query = conv(query * head_dim**-0.5)
        self.key = conv(key)
        self.value = conv(value)
        self.out = conv(layer.attn.Wo.weight)
        self.mlp_norm = Norm(layer.mlp_norm)
        self.up = conv(layer.mlp.Wi.weight)
        self.down = conv(layer.mlp.Wo.weight)

    def forward(self, x, bias, cos, sin):
        h = x if self.attn_norm is None else self.attn_norm(x)

        def rotate(t):
            return t * cos + torch.cat((-t[:, self.half :], t[:, : self.half]), dim=1) * sin

        queries = [rotate(t) for t in self.query(h).chunk(self.heads, dim=1)]
        keys = [rotate(t) for t in self.key(h).chunk(self.heads, dim=1)]
        values = self.value(h).chunk(self.heads, dim=1)
        heads = []
        for q, k, v in zip(queries, keys, values, strict=True):
            scores = torch.einsum("bchq,bkhc->bkhq", q, k.transpose(1, 3)) + bias
            heads.append(torch.einsum("bkhq,bchk->bchq", torch.softmax(scores, dim=1), v))
        x = x + self.out(torch.cat(heads, dim=1))
        gated, gate = self.up(self.mlp_norm(x)).chunk(2, dim=1)
        return x + self.down(nn.functional.gelu(gated) * gate)


class Packed(nn.Module):
    """The tagger as channels-first tensors of the fixed length, for several units in one
    sequence. The rotary angles come from a 1x1 convolution on the one-hot of the position, so the
    positions are an input and no gather is needed."""

    def __init__(self, tagger) -> None:
        super().__init__()
        config, encoder = tagger.config, tagger.model
        heads = config.num_attention_heads
        self.window_size = config.sliding_window
        self.types = config.layer_types
        self.kinds = sorted(set(self.types))
        self.embedding = encoder.embeddings.tok_embeddings
        self.embedding_norm = Norm(encoder.embeddings.norm)
        self.layers = nn.ModuleList(
            Layer(layer, heads, config.hidden_size // heads) for layer in encoder.layers
        )
        positions = torch.arange(MAX_TOKENS).unsqueeze(0)
        with torch.no_grad():
            angles = [
                part[0].T
                for kind in self.kinds
                for part in encoder.rotary_emb(torch.zeros(1, MAX_TOKENS, 1), positions, kind)
            ]
        self.rope = conv(torch.cat(angles))
        self.register_buffer(
            "slots", torch.arange(MAX_TOKENS, dtype=torch.float32).view(1, -1, 1, 1)
        )
        self.final_norm = Norm(encoder.final_norm)
        self.dense = conv(tagger.head.dense.weight)
        self.head_norm = Norm(tagger.head.norm)
        self.classifier = conv(tagger.classifier.weight)

    def forward(self, input_ids, position_ids, segment_ids):
        x = self.embedding(input_ids.long()).transpose(1, 2).unsqueeze(2)
        position = position_ids.to(x.dtype).view(1, 1, 1, -1)
        segment = segment_ids.to(x.dtype)
        apart = (segment.view(1, -1, 1, 1) - segment.view(1, 1, 1, -1)).abs().clamp(0, 1)
        far = ((position.view(1, -1, 1, 1) - position).abs() - self.window_size).clamp(0, 1)
        one_hot = 1.0 - (position - self.slots.to(x.dtype)).abs().clamp(0, 1)
        parts = self.rope(one_hot).chunk(2 * len(self.kinds), dim=1)
        angles = {k: (parts[2 * n], parts[2 * n + 1]) for n, k in enumerate(self.kinds)}
        global_bias, local_bias = apart * MASKED, (apart + far).clamp(0, 1) * MASKED
        x = self.embedding_norm(x)
        for layer, kind in zip(self.layers, self.types, strict=True):
            cos, sin = angles[kind]
            x = layer(x, local_bias if layer.local else global_bias, cos, sin)
        x = self.head_norm(nn.functional.gelu(self.dense(self.final_norm(x))))
        return self.classifier(x)


def convert_task(tagger, task: str, output: Path, quantize: bool) -> None:
    """One package for the task, with the input length MAX_TOKENS, and its compiled model. Without
    `quantize` the package is fp32 and its weights stay as they are."""
    index = torch.arange(MAX_TOKENS)[None]
    example = (torch.randint(0, 1000, (1, MAX_TOKENS)), index % 100, index // 100 + 1)
    with warnings.catch_warnings():
        # external: the converter passes of coremltools 9 warn about values that they handle
        warnings.simplefilter("ignore")
        traced = torch.jit.trace(Packed(tagger).eval(), example, strict=False)
        model = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name=name, shape=[1, MAX_TOKENS], dtype=np.int32) for name in INPUTS
            ],
            outputs=[ct.TensorType(name="logits")],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16 if quantize else ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.macOS15,
        )
    package = output / f"{task}.mlpackage"
    (compress(model) if quantize else model).save(str(package))
    ct.utils.compile_model(str(package), str(output / f"{task}.mlmodelc"))


def build(model: Path, output: Path, quantize: bool) -> None:
    """The packed taggers of the checkpoint in `output`, with the tokenizer and settings.json."""
    output.mkdir(parents=True)
    for task, tagger in load(model).items():
        convert_task(tagger, task, output, quantize)
    tokenizer = load_tokenizer(model)
    tokenizer.save_pretrained(output)
    write_json(
        output / "settings.json",
        {
            "pad_token_id": tokenizer.pad_token_id,
            "tags": TAG_NAMES,
            "max_tokens": MAX_TOKENS,
            "trim": {"closing": CLOSING, "opening": OPENING, "option_trailing": OPTION_TRAILING},
            "decision": {"pattern": DECISION.pattern, "flags": "i"},
            "precision": "float16" if quantize else "float32",
            "coremltools": ct.__version__,
            "torch": torch.__version__,
            "digests": {
                "source": source_digest(),
                "checkpoint": {
                    task: file_digest(model / task / "model.safetensors") for task in TASKS
                },
                "packages": {
                    task: directory_digest(output / f"{task}.mlpackage") for task in TASKS
                },
            },
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=COREML_ANE)
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"convert-ane: {args.output} exists", file=sys.stderr)
        return 1
    build(args.model, args.output, quantize=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
