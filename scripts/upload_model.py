#!/usr/bin/env python3
"""
Upload NVFP4 quantized checkpoint to Hugging Face Hub.

This script uploads the quantized model checkpoint (not the TensorRT engine)
to Hugging Face, allowing others with Blackwell GPUs to build their own engines.

Usage:
    python scripts/upload_model.py --repo-id <username>/Qwen3-8B-NVFP4

    # Or with environment variable
    HF_REPO_ID=<username>/Qwen3-8B-NVFP4 python scripts/upload_model.py
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Error: huggingface_hub not installed.")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)


# Default paths (can be overridden)
DEFAULT_CHECKPOINT_DIR = Path("/models/checkpoint_nvfp4")
DEFAULT_MODELS_VOLUME = "mailbrain-models"

MODEL_CARD_TEMPLATE = """---
license: apache-2.0
base_model: Qwen/Qwen3-8B
tags:
  - tensorrt
  - tensorrt-llm
  - nvfp4
  - blackwell
  - quantized
library_name: tensorrt-llm
pipeline_tag: text-generation
---

# Qwen3-8B-NVFP4

NVFP4 quantized version of [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) optimized for NVIDIA Blackwell GPUs.

## Quantization Details

| Parameter | Value |
|-----------|-------|
| Format | NVFP4 (4-bit floating point) |
| KV Cache | FP8 |
| Calibration samples | 512 |
| Calibration dataset | cnn_dailymail |
| Calibration max seq length | 2048 |
| Base precision | BFloat16 |

### Tools Used
- TensorRT-LLM: 1.2.0
- ModelOpt: 0.37.0
- CUDA: 13.0

## Hardware Requirements

- **Required**: NVIDIA Blackwell GPU (SM120)
  - RTX 5090
  - RTX PRO 4000
  - RTX 5080
  - B100/B200 (datacenter)

This checkpoint will **not work** on older GPUs (Ada, Hopper, Ampere, etc.) as NVFP4 is a Blackwell-exclusive format.

## Usage

### Build TensorRT Engine

```bash
# Install TensorRT-LLM 1.2+
pip install tensorrt-llm

# Download checkpoint
huggingface-cli download {repo_id} --local-dir ./checkpoint

# Build engine for your GPU
trtllm-build \\
    --checkpoint_dir ./checkpoint \\
    --output_dir ./engine \\
    --max_input_len 4096 \\
    --max_seq_len 4608 \\
    --max_batch_size 32 \\
    --gemm_plugin nvfp4 \\
    --gpt_attention_plugin bfloat16 \\
    --kv_cache_type paged \\
    --tokens_per_block 64 \\
    --remove_input_padding enable \\
    --use_fused_mlp enable \\
    --multiple_profiles enable
```

### Run Inference

```python
from tensorrt_llm import LLM, SamplingParams

llm = LLM(model="./engine")
output = llm.generate("Hello, world!", SamplingParams(max_tokens=100))
print(output)
```

## Performance

Compared to FP16 baseline on RTX PRO 4000:

| Metric | FP16 | NVFP4 |
|--------|------|-------|
| Memory | ~16 GB | ~4 GB |
| Throughput | 1x | ~2.5x |

## Limitations

- NVFP4 format is Blackwell-exclusive
- Some accuracy loss compared to FP16 (typically <1% on benchmarks)
- `lm_head` is quantized (may affect output quality slightly)

## License

This model inherits the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0) from Qwen3-8B.

## Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen) for the base model
- [NVIDIA](https://github.com/NVIDIA/TensorRT-LLM) for TensorRT-LLM and ModelOpt
"""


def get_checkpoint_path_from_docker(volume_name: str) -> Path | None:
    """Try to find checkpoint path via Docker volume inspection."""
    import subprocess

    try:
        result = subprocess.run(
            ["docker", "volume", "inspect", volume_name, "--format", "{{.Mountpoint}}"],
            capture_output=True,
            text=True,
            check=True,
        )
        mount_point = result.stdout.strip()
        checkpoint_path = Path(mount_point) / "checkpoint_nvfp4"
        if checkpoint_path.exists():
            return checkpoint_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


def find_checkpoint_dir() -> Path | None:
    """Find the NVFP4 checkpoint directory."""
    # Try common locations
    candidates = [
        DEFAULT_CHECKPOINT_DIR,
        Path("./models/checkpoint_nvfp4"),
        Path("../models/checkpoint_nvfp4"),
    ]

    for path in candidates:
        if path.exists() and (path / "config.json").exists():
            return path

    # Try Docker volume
    docker_path = get_checkpoint_path_from_docker(DEFAULT_MODELS_VOLUME)
    if docker_path:
        return docker_path

    return None


def upload_checkpoint(
    checkpoint_dir: Path,
    repo_id: str,
    private: bool = False,
) -> str:
    """Upload checkpoint to Hugging Face Hub."""
    api = HfApi()

    # Create repo if it doesn't exist
    print(f"Creating/accessing repository: {repo_id}")
    create_repo(repo_id, exist_ok=True, private=private)

    # Generate model card
    model_card = MODEL_CARD_TEMPLATE.format(repo_id=repo_id)
    readme_path = checkpoint_dir / "README.md"
    readme_path.write_text(model_card)
    print(f"Generated model card: {readme_path}")

    # Upload all files
    print(f"Uploading checkpoint from: {checkpoint_dir}")
    print("This may take a while for large files...")

    api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
        commit_message="Upload NVFP4 quantized Qwen3-8B checkpoint",
    )

    # Clean up temporary README
    if readme_path.exists():
        readme_path.unlink()

    return f"https://huggingface.co/{repo_id}"


def main():
    parser = argparse.ArgumentParser(
        description="Upload NVFP4 checkpoint to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=os.environ.get("HF_REPO_ID"),
        help="Hugging Face repository ID (e.g., username/Qwen3-8B-NVFP4)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Path to checkpoint directory (auto-detected if not specified)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )

    args = parser.parse_args()

    # Validate repo ID
    if not args.repo_id:
        print("Error: Repository ID required.")
        print("Use --repo-id <username>/Qwen3-8B-NVFP4")
        print("Or set HF_REPO_ID environment variable")
        sys.exit(1)

    if "/" not in args.repo_id:
        print(f"Error: Invalid repo ID format: {args.repo_id}")
        print("Expected format: <username>/<model-name>")
        sys.exit(1)

    # Find checkpoint directory
    checkpoint_dir = args.checkpoint_dir or find_checkpoint_dir()

    if not checkpoint_dir or not checkpoint_dir.exists():
        print("Error: Could not find checkpoint directory.")
        print("Specify with --checkpoint-dir or ensure the model was built.")
        print(f"Looked in: {DEFAULT_CHECKPOINT_DIR}, ./models/checkpoint_nvfp4")
        sys.exit(1)

    # Verify checkpoint contents
    required_files = ["config.json"]
    for f in required_files:
        if not (checkpoint_dir / f).exists():
            print(f"Error: Missing required file: {f}")
            print(f"Directory contents: {list(checkpoint_dir.iterdir())}")
            sys.exit(1)

    print("=" * 60)
    print("NVFP4 Checkpoint Upload")
    print("=" * 60)
    print(f"  Repository: {args.repo_id}")
    print(f"  Checkpoint: {checkpoint_dir}")
    print(f"  Private: {args.private}")
    print()

    # Check if logged in
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"  Logged in as: {user_info['name']}")
    except Exception:
        print("Error: Not logged in to Hugging Face.")
        print("Run: huggingface-cli login")
        sys.exit(1)

    print()

    # Upload
    url = upload_checkpoint(checkpoint_dir, args.repo_id, args.private)

    print()
    print("=" * 60)
    print("Upload complete!")
    print("=" * 60)
    print(f"  URL: {url}")
    print()
    print("Others can now use your checkpoint:")
    print(f"  huggingface-cli download {args.repo_id} --local-dir ./checkpoint")


if __name__ == "__main__":
    main()
