"""
- load tokenizer + PEFT-wrapped model
- return parsed scores and flagged token sequences
"""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict

import boto3
import botocore
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import CanineTokenizer, CanineForTokenClassification
from peft import PeftConfig, PeftModel


# ---- Configuration via environment or constants ----
LOG = logging.getLogger("model_service")
logging.basicConfig(level=logging.INFO)

HF_BASE = os.getenv("HF_BASE_MODEL", "google/canine-c")
S3_BUCKET = os.getenv("S3_BUCKET", "data-science")
S3_BASE_KEY = os.getenv("S3_QUANT_KEY", "projects/blocked_words_project/models/v1/quantized_model")
S3_QUANT_NAME = os.getenv("S3_QUANT_NAME", "quantized_model_gpu_2.pt")
S3_PEFT_KEY = os.getenv("S3_PEFT_KEY", "projects/blocked_words_project/models/v1/blocked-word-model-canine-c-lora")
PEFT_FILES = ["adapter_config.json", "adapter_model.bin"]

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "./main/artifacts"))
LOCAL_QUANT_PATH = ARTIFACT_DIR / S3_QUANT_NAME

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
MIN_FLAGGED_TOKEN_LEN = int(os.getenv("MIN_FLAGGED_TOKEN_LEN", "3"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- Utility: S3 download with simple retry ----
def download_from_s3(bucket: str, key: str, dest: Path, max_attempts: int = 3, backoff: float = 1.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    attempt = 0
    while attempt < max_attempts:
        try:
            LOG.info("Downloading s3://%s/%s -> %s (attempt %d)", bucket, key, dest, attempt + 1)
            s3.download_file(bucket, key, str(dest))
            return
        except botocore.exceptions.BotoCoreError as exc:
            attempt += 1
            LOG.warning("S3 download failed (attempt %d): %s", attempt, exc)
            time.sleep(backoff * attempt)
    raise RuntimeError(f"Failed to download s3://{bucket}/{key} after {max_attempts} attempts")


# ---- Model wrapper class ----
class FlagModel():
    """
    Purely rewritten model wrapper.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model: Optional[PeftModel] = None
        self._tokenizer: Optional[CanineTokenizer] = None
        self._device: torch.device = DEVICE
        self._loaded = False

    # ----- build time: fetch artifacts to local disk -----
    def build(self) -> None:
        """
        Called at build time to fetch artifacts from S3 into ARTIFACT_DIR.
        """
        LOG.info("Build: ensure artifacts directory exists at %s", ARTIFACT_DIR)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

        # quantized checkpoint
        quant_key = f"{S3_BASE_KEY}/{S3_QUANT_NAME}"
        download_from_s3(S3_BUCKET, quant_key, LOCAL_QUANT_PATH)

        # peft/hf-contained files (e.g. adapter_config.json & adapter_model.bin)
        for fname in PEFT_FILES:
            remote = f"{S3_PEFT_KEY}/{fname}"
            local = ARTIFACT_DIR / fname
            download_from_s3(S3_BUCKET, remote, local)

        LOG.info("Build: artifact downloads complete")

    # ----- serving-time model initialization -----
    def initialize_model(self) -> None:
        """
        Called once per serving instance. Loads model weights and tokenizer.
        """
        LOG.info("Initializing model on device %s", self._device)
        if not LOCAL_QUANT_PATH.exists():
            raise FileNotFoundError(f"Quantized checkpoint missing at {LOCAL_QUANT_PATH}")

        # load quantized state (assumed to be a torch.save object)
        quant_obj = torch.load(str(LOCAL_QUANT_PATH), map_location="cpu")

        # load base + peft adapter
        base_net = CanineForTokenClassification.from_pretrained(HF_BASE)
        peft_conf = PeftConfig.from_pretrained(str(ARTIFACT_DIR))
        peft_wrapped = PeftModel.from_pretrained(base_net, str(ARTIFACT_DIR), state_dict=getattr(quant_obj, "state_dict", lambda: quant_obj)())
        self._model = peft_wrapped.to(self._device)
        self._model.eval()

        # tokenizer
        self._tokenizer = CanineTokenizer.from_pretrained(HF_BASE)

        LOG.info("Model initialization complete. CUDA available=%s", torch.cuda.is_available())
        self._loaded = True

    # ----- dataset / dataloader helpers -----
    def _make_dataset(self, texts: List[str]) -> Dataset:
        payload = {"text": texts}
        return Dataset.from_dict(payload)

    def _tokenize_single(self, text: str, **kwargs):
        assert self._tokenizer is not None
        return self._tokenizer(text, truncation=True, padding="max_length", max_length=MAX_TOKENS, **kwargs)

    def _tokenize_dataset(self, ds: Dataset) -> Dataset:
        ds_tok = ds.map(lambda e: self._tokenize_single(e["text"]), batched=True)
        ds_tok = ds_tok.remove_columns(["text"])
        ds_tok.set_format(type="torch")
        return ds_tok

    def _make_dataloader(self, ds_tokenized: Dataset, batch_size: int = 8) -> DataLoader:
        # rely on datasets to produce tensors and then wrap into DataLoader
        return DataLoader(ds_tokenized, batch_size=batch_size)

    # ----- low-level inference -----    
    def _infer_single(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._model is not None and self._tokenizer is not None
        enc = self._tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_TOKENS)
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(enc["input_ids"], enc.get("attention_mask"))
        logits = out.logits
        probs = logits.softmax(-1)
        return probs.cpu(), enc["input_ids"].cpu()

    def _infer_batch(self, texts: List[str], batch_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._model is not None
        ds = self._make_dataset(texts)
        ds_tok = self._tokenize_dataset(ds)
        loader = self._make_dataloader(ds_tok, batch_size=batch_size)

        all_probs = []
        all_ids = []
        for batch in loader:
            # move tensors to device
            batch = {k: v.to(self._device) for k, v in batch.items()}
            with torch.no_grad():
                out = self._model(batch["input_ids"], batch.get("attention_mask"))
            logits = out.logits
            probs = logits.softmax(-1)
            all_probs.append(probs.cpu())
            all_ids.append(batch["input_ids"].cpu())

        return torch.cat(all_probs, dim=0), torch.cat(all_ids, dim=0)

    # ----- parsing outputs into human-friendly result -----
    def _extract_flags(self, probas: torch.Tensor, input_ids: torch.Tensor, threshold: float = 0.5) -> Dict[str, Union[float, List[str]]]:
        # probas shape: (seq_len, num_classes) or (batch, seq_len, num_classes)
        # We expect to feed a 1D prob array corresponding to the "positive" class probability.
        if probas.ndim == 2:  # (seq_len, num_classes)
            pos_probs = probas[:, 1]
        elif probas.ndim == 3:  # (batch, seq_len, num_classes)
            pos_probs = probas[:, :, 1]
        else:
            raise ValueError("Unexpected probability tensor dims")

        # We'll handle the single-example case here
        if pos_probs.ndim == 1:
            mask = pos_probs > threshold
            ids = input_ids[mask].tolist()
            tokens = self._tokenizer.convert_ids_to_tokens(ids)
            words = _reconstruct_token_groups(tokens)
            max_score = float(pos_probs.max().item()) if pos_probs.numel() > 0 else 0.0
            filtered = [w for w in words if len(w.strip()) >= MIN_FLAGGED_TOKEN_LEN]
            return {"score": max_score, "details": filtered}
        else:
            # batch case: return list of dicts
            results = []
            for single_pos, single_ids in zip(pos_probs, input_ids):
                mask = single_pos > threshold
                ids = single_ids[mask].tolist()
                tokens = self._tokenizer.convert_ids_to_tokens(ids)
                words = _reconstruct_token_groups(tokens)
                filtered = [w for w in words if len(w.strip()) >= MIN_FLAGGED_TOKEN_LEN]
                max_score = float(single_pos.max().item()) if single_pos.numel() > 0 else 0.0
                results.append({"score": max_score, "details": filtered})
            return {"batch_results": results}

    # ----- public API used by caller -----
    def predict_texts(self, inputs: Union[str, List[str], pd.Series], batch_size: int = 8, threshold: float = 0.5) -> List[Dict[str, Union[float, List[str]]]]:
        """
        Accepts a single string, a list of strings, or a pandas Series. Returns a list of result dictionaries
        identical in length to the number of inputs.
        """
        if not self._loaded:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")

        # Normalize input to list[str]
        if isinstance(inputs, pd.Series):
            texts = inputs.fillna("").astype(str).tolist()
        elif isinstance(inputs, str):
            texts = [inputs]
        elif isinstance(inputs, list):
            texts = [str(t) if t is not None else "" for t in inputs]
        else:
            raise TypeError("Unsupported input type for prediction")

        # Preprocessing hook: user-provided preprocess can be applied if exists
        preproc = getattr(self, "preprocess_fn", None)
        if callable(preproc):
            texts = [preproc(t) for t in texts]

        if len(texts) == 0:
            return [{"score": 0.0, "details": []}]

        if len(texts) == 1:
            probs, ids = self._infer_single(texts[0])
            parsed = self._extract_flags(probs.squeeze(0), ids.squeeze(0), threshold=threshold)
            # ensure list output for compatibility
            if "batch_results" in parsed:
                return parsed["batch_results"]
            return [parsed]
        else:
            probs_batch, ids_batch = self._infer_batch(texts, batch_size=batch_size)
            parsed = self._extract_flags(probs_batch, ids_batch, threshold=threshold)
            return parsed.get("batch_results", [])

    # ----- endpoint -----
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame with 'score' and 'details' columns.
        """
        try:
            results = self.predict_texts(df["text"].fillna("").astype(str).tolist())
            out = pd.DataFrame(results)
            # ensure columns exist and normalize
            out["details"] = out["details"].apply(lambda l: [w for w in l if len(w.strip()) >= MIN_FLAGGED_TOKEN_LEN])
            out.loc[out["details"].str.len() == 0, "score"] = 0.0
            return out
        except Exception as exc:
            LOG.exception("Prediction error")
            return pd.DataFrame({"score": [0.0] * len(df), "details": [[] for _ in range(len(df))]})


# ---- small helper: reconstruct contiguous token groups into strings ----
def _reconstruct_token_groups(tokens: List[str]) -> List[str]:
    """
    Receive a flat list of tokens (subword or char-level tokens)
    and group contiguous tokens into joined strings.
    This is intentionally simple: join sequential tokens (no position info).
    """
    if not tokens:
        return []
    groups = []
    current = tokens[0]
    for t in tokens[1:]:
        # Heuristic: if token starts with '##' or is a continuation marker, glue it; otherwise break group.
        # Note: Canine uses character-based tokens; adjust heuristics if using a different tokenizer.
        if isinstance(t, str) and (t.startswith("##") or t.startswith("Ġ") or t.startswith("▁")):
            # remove markers and append
            current += t.lstrip("#Ġ▁")
        else:
            groups.append(current)
            current = t
    groups.append(current)
    return groups


# If someone imports module and wants a ready-to-use instance:
_service_instance: Optional[FlagModel] = None


def get_service_instance() -> FlagModel:
    global _service_instance
    if _service_instance is None:
        _service_instance = FlagModel()
    return _service_instance
