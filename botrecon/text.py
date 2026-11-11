"""Text encoders.

Variant A [M 3.1]: frozen bert-base-uncased; the 20 most recent tweets joined
into one document, WordPiece, max 128 tokens; t_v = last-layer [CLS] state (768-d).

Variant B [M Alg.1, 4.1]: a Sentence-Transformers model on the profile
description, attention-masked mean pooling. The manuscript names no
checkpoint; all-MiniLM-L6-v2 (384-d) is a reconstruction choice [C].

Empty input -> zero vector [C]; the empty rate per class is recorded because
empty text can act as a label shortcut (e.g. Cresci-2017 traditional spambots
#2-#4 have no tweets in the release).

HashingTestEncoder exists ONLY for the synthetic smoke test. Runs that use it
are labelled SMOKE-TEST-NOT-A-RESULT.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np


def text_inputs(data, variant: str, max_tweets: int = 20) -> list:
    if variant == "A":
        if data.tweets is None:
            raise ValueError("Variant A needs tweets; load the dataset with load_tweets=True")
        return [" ".join(tw[:max_tweets]) for tw in data.tweets]
    if variant == "B":
        return list(data.description)
    raise ValueError(variant)


def empty_rates(texts: list, y: np.ndarray) -> dict:
    empty = np.array([len(t.strip()) == 0 for t in texts])
    return {"empty_text_rate_bot": float(empty[y == 1].mean()) if (y == 1).any() else float("nan"),
            "empty_text_rate_human": float(empty[y == 0].mean()) if (y == 0).any() else float("nan")}



LOCAL_MODEL_ROOTS = ("/kaggle/input",)


def local_model_path(model_name: str, roots=LOCAL_MODEL_ROOTS, marker: str = "config.json"):
    """A local copy of a Hugging Face model attached as a Kaggle input (for notebooks without
    internet): a folder named like the model's last path part that holds `marker`. None if absent."""
    import glob
    want = model_name.rstrip("/").split("/")[-1].lower()
    hits = []
    for r in roots:
        if not os.path.isdir(r):
            continue
        for m in glob.glob(os.path.join(r, "**", marker), recursive=True):
            d = os.path.dirname(m)
            if os.path.basename(d).lower() == want:
                hits.append(d)
    return sorted(hits, key=len)[0] if hits else None


def model_revision(path: str) -> str | None:
    """Git commit of a cloned model folder (provenance of an offline copy), if recorded."""
    try:
        head = open(os.path.join(path, ".git", "HEAD")).read().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1]
            f = os.path.join(path, ".git", ref)
            if os.path.exists(f):
                return open(f).read().strip()
            packed = os.path.join(path, ".git", "packed-refs")
            for line in open(packed):
                if line.strip().endswith(ref):
                    return line.split()[0]
        return head
    except OSError:
        return None


def _resolve(model_name: str, marker: str):
    """(path to load, provenance string). A local copy wins, so no internet is needed."""
    if os.path.isdir(model_name):
        return model_name, f"local:{model_name}"
    local = local_model_path(model_name, LOCAL_MODEL_ROOTS, marker=marker)
    if local:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return local, f"local:{local} (git {model_revision(local) or 'unknown'})"
    return model_name, "huggingface.co"


class BertCLSEncoder:
    test_only = False

    def __init__(self, model_name: str = "bert-base-uncased", max_tokens: int = 128,
                 device: str | None = None, batch_size: int = 64):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        path, self.source = _resolve(model_name, "config.json")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(self.device).eval()
        self.max_tokens, self.batch_size = max_tokens, batch_size
        self.dim = self.model.config.hidden_size
        self.name = f"{model_name}|cls|L{max_tokens}"

    def encode(self, texts: list) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        nonempty = [i for i, t in enumerate(texts) if t.strip()]
        with self.torch.no_grad():
            for b in range(0, len(nonempty), self.batch_size):
                idx = nonempty[b:b + self.batch_size]
                enc = self.tok([texts[i] for i in idx], truncation=True, max_length=self.max_tokens,
                               padding=True, return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state[:, 0]          # [CLS]
                out[idx] = h.float().cpu().numpy()
        return out


class SBERTMeanEncoder:
    test_only = False

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str | None = None, batch_size: int = 128):
        from sentence_transformers import SentenceTransformer
        path, self.source = _resolve(model_name, "modules.json")
        self.model = SentenceTransformer(path, device=device)
        get_dim = (getattr(self.model, "get_sentence_embedding_dimension", None)
                   or getattr(self.model, "get_embedding_dimension"))
        self.dim = get_dim()
        self.batch_size = batch_size
        self.name = f"{model_name}|mean"

    def encode(self, texts: list) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        nonempty = [i for i, t in enumerate(texts) if t.strip()]
        if nonempty:
            emb = self.model.encode([texts[i] for i in nonempty], batch_size=self.batch_size,
                                    convert_to_numpy=True, normalize_embeddings=False,
                                    show_progress_bar=False)
            out[nonempty] = emb.astype(np.float32)
        return out


class HashingTestEncoder:
    """Deterministic bag-of-words hashing + fixed random projection. TEST ONLY."""
    test_only = True

    def __init__(self, dim: int = 32):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.vec = HashingVectorizer(n_features=2 ** 12, alternate_sign=False, norm="l2")
        rng = np.random.default_rng(0)
        self.proj = rng.normal(size=(2 ** 12, dim)).astype(np.float32) / np.sqrt(dim)
        self.dim = dim
        self.name = f"hashing-test-only|{dim}"

    def encode(self, texts: list) -> np.ndarray:
        out = np.asarray(self.vec.transform(texts) @ self.proj, dtype=np.float32)
        out[[not t.strip() for t in texts]] = 0.0
        return out


def make_encoder(cfg, device=None, test_encoder: bool = False):
    if test_encoder:
        return HashingTestEncoder()
    if cfg.text == "A":
        return BertCLSEncoder(cfg.text_model_a, cfg.max_tokens, device)
    return SBERTMeanEncoder(cfg.text_model_b, device)


def cached_encode(encoder, texts: list, cache_dir: str | None, key: str) -> np.ndarray:
    """Embeddings depend only on text (never on labels or splits), so they are computed
    once for all nodes and cached."""
    if cache_dir is None:
        return encoder.encode(texts)
    h = hashlib.sha1(("\x1f".join(texts) + encoder.name).encode("utf-8", "ignore")).hexdigest()[:12]
    path = os.path.join(cache_dir, f"{key}_{h}.npy")
    if os.path.exists(path):
        return np.load(path)
    os.makedirs(cache_dir, exist_ok=True)
    emb = encoder.encode(texts)
    np.save(path, emb)
    return emb
