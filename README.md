# content-filtration

FlagModel Service – Blocked Word Detection (Canine + LoRA + Quantized Inference)

This repository provides a model service that detects flagged or blocked words in text using a quantized Canine model fine-tuned with PEFT/LoRA adapters.
It is designed for scalable serving (GPU/CPU compatible), S3 artifact management, and batched inference over text inputs (single, list, or DataFrame).

Overview

The service wraps a HuggingFace Canine model (google/canine-c) with PEFT fine-tuned adapters for token classification.
It supports:

Loading quantized model weights from S3

Loading LoRA adapter configuration and parameters

Performing token-level classification to detect potentially blocked/flagged words

Producing structured outputs containing probability scores and flagged token groups

Architecture
 ┌────────────────────────────┐
 │        S3 Storage          │
 │  ├── quantized_model_gpu_2.pt
 │  └── adapter_model.bin
 └──────────────┬─────────────┘
                │ (download)
                ▼
 ┌────────────────────────────┐
 │       FlagModel Class      │
 │  ├── build()               │ Fetches artifacts
 │  ├── initialize_model()    │ Loads model + tokenizer
 │  ├── predict_texts()       │ Inference (string/list/DataFrame)
 │  ├── _extract_flags()      │ Parse probabilities → tokens
 │  └── predict()             │ High-level DF endpoint
 └────────────────────────────┘
                │
                ▼
         Human-friendly Results

Model Components
Component	Description
Base Model	google/canine-c (character-level token classification)
Adapter	LoRA fine-tuned adapter for blocked-word classification
Quantized Model	Torch quantized checkpoint (quantized_model_gpu_2.pt)
Tokenizer	CanineTokenizer from HuggingFace
Device	Auto-detects GPU if available (torch.device("cuda" if torch.cuda.is_available() else "cpu"))

Environment Configuration
Variable	Default	Description
HF_BASE_MODEL	google/canine-c	HuggingFace base model name
S3_BUCKET	data-science	S3 bucket containing model artifacts
S3_QUANT_KEY	projects/blocked_words_project/models/v1/quantized_model	Folder path in S3 for quantized model
S3_QUANT_NAME	quantized_model_gpu_2.pt	Filename for quantized model checkpoint
S3_PEFT_KEY	projects/blocked_words_project/models/v1/blocked-word-model-canine-c-lora	Folder path for adapter config
ARTIFACT_DIR	./main/artifacts	Local directory to store downloaded artifacts
MAX_TOKENS	2048	Tokenization max length
MIN_FLAGGED_TOKEN_LEN	3	Minimum token length to count as a flagged word

Installation
pip install torch transformers peft datasets boto3 botocore pandas numpy

🔍 Inference Usage
1️ Single String
svc.predict_texts("This text contains suspicious words")
→ [{'score': 0.92, 'details': ['suspicious', 'words']}]

2️ Batch of Texts
texts = ["normal text", "blocked phrase found here"]
svc.predict_texts(texts)
→ [{'score': 0.0, 'details': []}, {'score': 0.88, 'details': ['blocked', 'phrase']}]

3️ Pandas DataFrame Endpoint
import pandas as pd
df = pd.DataFrame({"text": ["hello world", "forbidden word inside"]})
result = svc.predict(df)
print(result)
→ DataFrame with columns: ['score', 'details']

Output Format

Each prediction returns a list of dictionaries:

Key	Type	Description
score	float	Maximum probability of flagged token
details	List[str]	List of reconstructed flagged token groups

Internal Details
S3 Downloader (download_from_s3)

Retries failed downloads up to 3 times

Uses exponential backoff (1s × attempt)

Logs progress via logging

Flag Extraction (_extract_flags)

Extracts “positive” class probability per token ([:, 1])

Reconstructs contiguous flagged token sequences

Filters out tokens below minimum length (MIN_FLAGGED_TOKEN_LEN)

Token Reconstruction (_reconstruct_token_groups)

Joins continuation tokens like ##sub or ▁word into readable strings

Designed for character-level models (Canine)
