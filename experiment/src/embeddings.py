"""
embeddings.py - tokenizer, embedding model and embedding cache.

The model, its remote code and its tokenizer are loaded at pinned revisions. Every text is
checked against the model's input limit before embedding (prefix and special tokens included),
so nothing is silently truncated: an over-long input raises.

Cache key = SHA-256 over: model id, model revision, remote-code revision, tokenizer revision,
prefix, normalisation flag, max_seq_length, library versions and every input text, in order.
Any change to chunk boundaries changes the texts and therefore the key.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class Tokenizer:
    def __init__(self, name: str, revision: str):
        from transformers import AutoTokenizer
        self.name, self.revision = name, revision
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True,
                       truncation=False, verbose=False)
        offs = [tuple(o) for o in enc["offset_mapping"]]
        for (a0, a1), (b0, b1) in zip(offs, offs[1:]):
            if not (a0 < a1 <= b0 < b1):
                raise ValueError(f"non-monotonic token offsets {(a0, a1)} {(b0, b1)}")
        return offs

    def n_input_tokens(self, texts: list[str]) -> list[int]:
        """Tokens the model will actually see, special tokens included."""
        enc = self.tok(texts, add_special_tokens=True, truncation=False, verbose=False)
        return [len(x) for x in enc["input_ids"]]


class Embedder:
    def __init__(self, cfg: dict):
        import sentence_transformers
        import torch
        import transformers
        from sentence_transformers import SentenceTransformer
        self.cfg = cfg
        torch.manual_seed(0)
        self.model = SentenceTransformer(
            cfg["id"], revision=cfg["revision"], trust_remote_code=True,
            model_kwargs={"code_revision": cfg["code_revision"]},
            config_kwargs={"code_revision": cfg["code_revision"]}, device=cfg.get("device", "cpu"))
        self.max_len = int(self.model.max_seq_length)
        if self.max_len != cfg["max_seq_length"]:
            raise RuntimeError(f"model max_seq_length {self.max_len} != config {cfg['max_seq_length']}")
        self.tokenizer = Tokenizer(cfg["id"], cfg["revision"])
        self.versions = {"sentence_transformers": sentence_transformers.__version__,
                         "transformers": transformers.__version__, "torch": torch.__version__}

    def _check_lengths(self, texts):
        n = self.tokenizer.n_input_tokens(texts)
        over = [(i, x) for i, x in enumerate(n) if x > self.max_len]
        if over:
            raise ValueError(f"{len(over)} inputs exceed {self.max_len} tokens, e.g. {over[:3]}")
        return n

    def encode(self, texts: list[str], prefix: str) -> tuple[np.ndarray, list[int]]:
        full = [prefix + t for t in texts]
        n = self._check_lengths(full)
        E = self.model.encode(full, batch_size=self.cfg.get("batch_size", 16),
                              normalize_embeddings=True, show_progress_bar=False,
                              convert_to_numpy=True)
        return E.astype(np.float32), n

    def key_parts(self, prefix: str) -> list[str]:
        c = self.cfg
        return [c["id"], c["revision"], c["code_revision"], c["revision"], prefix, "normalize=True",
                str(self.max_len), json.dumps(self.versions, sort_keys=True)]


def cached_encode(emb, texts: list[str], prefix: str, cache_dir: Path | None, tag: str):
    """Returns (embeddings, input_token_counts, cache_hit)."""
    if cache_dir is None:
        E, n = emb.encode(texts, prefix)
        return E, n, False
    h = hashlib.sha256()
    for part in emb.key_parts(prefix) + texts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    key = h.hexdigest()[:20]
    npy, meta = cache_dir / f"{tag}_{key}.npy", cache_dir / f"{tag}_{key}.json"
    if npy.exists() and meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        E = np.load(npy)
        if m.get("key") == key and E.shape[0] == len(texts) == m["n"]:
            return E, m["input_tokens"], True
    E, n = emb.encode(texts, prefix)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy, E)
    meta.write_text(json.dumps({"key": key, "n": len(texts), "key_parts": emb.key_parts(prefix),
                                "input_tokens": n}, indent=1), encoding="utf-8")
    return E, n, False


class HashingEmbedder:
    """Deterministic bag-of-words embedder for the synthetic unit/integration fixture ONLY.
    It is never used for real results; run_experiment.py refuses it outside --fixture mode."""
    max_len = 10 ** 9

    def __init__(self, tokenizer, dim: int = 256):
        self.tokenizer, self.dim = tokenizer, dim
        self.versions = {"hashing": "1"}

    def encode(self, texts, prefix):
        import re
        E = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", (prefix + t).lower()):
                E[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        return E, [len(t.split()) for t in texts]

    def key_parts(self, prefix):
        return ["hashing", str(self.dim), prefix]
