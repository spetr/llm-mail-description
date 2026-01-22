#!/usr/bin/env python3
"""
Model download, quantization (NVFP4), and TensorRT-LLM engine build script.

Workflow for Blackwell GPUs (RTX PRO 4000):
1. Download Qwen3-8B from Hugging Face
2. Convert to TensorRT-LLM checkpoint format
3. Quantize to NVFP4 (Blackwell optimized, ~4x compression)
4. Build TensorRT engine with full optimizations

NVFP4 Benefits:
- 4-bit floating point optimized for Blackwell tensor cores
- ~4x memory reduction vs FP16
- Minimal accuracy loss compared to INT4
- Native hardware support on Blackwell GPUs
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_command(cmd: list[str], description: str) -> None:
    """Run a command with logging."""
    logger.info(f"{description}")
    logger.info(f"Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"{description} failed: {e}")
        raise


def download_model(repo_id: str, target_dir: Path, hf_token: str | None = None) -> Path:
    """Download model from Hugging Face Hub."""
    logger.info(f"Downloading model: {repo_id}")

    local_dir = target_dir / "hf_model"
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        # token: HuggingFace token for gated models (e.g., Llama requires approval).
        # Qwen3 is open, but token still useful for rate limits.
        token=hf_token,
        # ignore_patterns: Skip files we don't need.
        # *.gguf, *.ggml are quantized formats for llama.cpp - not needed for TensorRT-LLM.
        # We want the original safetensors/pytorch weights for our own quantization.
        ignore_patterns=["*.gguf", "*.ggml"],
    )

    logger.info(f"Model downloaded to: {local_dir}")
    return local_dir


def find_script(search_paths: list[Path]) -> Path | None:
    """Find a script in multiple possible locations."""
    for path in search_paths:
        if path.exists():
            return path
    return None


def convert_checkpoint(
    hf_model_dir: Path,
    output_dir: Path,
    dtype: str = "bfloat16",
) -> Path:
    """
    Convert HuggingFace model to TensorRT-LLM checkpoint format.

    This prepares the model for quantization.
    """
    logger.info("Step 2/5: Converting HF model to TensorRT-LLM checkpoint")

    checkpoint_dir = output_dir / "trtllm_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Find convert_checkpoint.py for Qwen
    search_paths = [
        Path("/usr/local/lib/python3.12/dist-packages/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py"),
        Path("/usr/local/lib/python3.12/dist-packages/tensorrt_llm/examples/qwen/convert_checkpoint.py"),
        Path("/workspace/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py"),
        Path("/app/tensorrt_llm/examples/qwen/convert_checkpoint.py"),
    ]

    convert_script = find_script(search_paths)

    if convert_script is None:
        logger.warning("convert_checkpoint.py not found, using HF model directly")
        return hf_model_dir

    cmd = [
        "python", str(convert_script),
        # --model_dir: Path to HuggingFace model (with config.json, *.safetensors, etc.)
        "--model_dir", str(hf_model_dir),
        # --output_dir: Where to save TensorRT-LLM checkpoint format.
        # This format is optimized for TensorRT-LLM and includes weight layout changes.
        "--output_dir", str(checkpoint_dir),
        # --dtype: Data type for weights in checkpoint.
        # bfloat16 recommended - same range as FP32, good starting point for quantization.
        # Will be quantized to NVFP4 in next step.
        "--dtype", dtype,
    ]

    run_command(cmd, "Checkpoint conversion")
    return checkpoint_dir


def quantize_nvfp4(
    checkpoint_dir: Path,
    output_dir: Path,
    calib_size: int = 512,
    calib_max_seq_length: int = 2048,
    calib_dataset: str | None = None,
) -> Path:
    """
    Quantize model to NVFP4 format for Blackwell GPUs.

    NVFP4 (4-bit floating point):
    - Block-wise quantization with block size 16
    - Calibrated activation scales
    - Native Blackwell tensor core support
    - ~4x compression with better accuracy than INT4

    Configuration matches NVIDIA's official nvidia/Qwen3-8B-NVFP4 model:
    - quant_algo: NVFP4
    - kv_cache_quant_algo: FP8
    - group_size: 16
    - exclude_modules: lm_head (keeps output layer in full precision)
    """
    logger.info("Step 3/5: Quantizing to NVFP4 (Blackwell optimized)")

    quantized_dir = output_dir / "checkpoint_nvfp4"
    quantized_dir.mkdir(parents=True, exist_ok=True)

    # Find quantize.py
    search_paths = [
        Path("/usr/local/lib/python3.12/dist-packages/tensorrt_llm/examples/quantization/quantize.py"),
        Path("/workspace/tensorrt_llm/examples/quantization/quantize.py"),
        Path("/app/tensorrt_llm/examples/quantization/quantize.py"),
    ]

    quantize_script = find_script(search_paths)

    if quantize_script is None:
        logger.error("quantize.py not found!")
        logger.info("Available paths checked:")
        for p in search_paths:
            logger.info(f"  - {p}: {'EXISTS' if p.exists() else 'NOT FOUND'}")
        raise FileNotFoundError("quantize.py script not found")

    cmd = [
        "python", str(quantize_script),
        "--model_dir", str(checkpoint_dir),
        "--output_dir", str(quantized_dir),
        # --dtype bfloat16: Intermediate precision for computations during quantization.
        # BF16 has same dynamic range as FP32 but less precision - good balance for LLMs.
        "--dtype", "bfloat16",
        # --qformat nvfp4: NVIDIA FP4 quantization format exclusive to Blackwell GPUs.
        # Uses 4-bit floating point (not integer!) with block-wise scaling (block=16).
        # Better accuracy than INT4 because it preserves floating-point semantics.
        # ~4x model compression: 8B model goes from ~16GB to ~4GB.
        "--qformat", "nvfp4",
        # --kv_cache_dtype fp8: FP8 quantization for KV cache (matches NVIDIA's config).
        # FP8 provides good compression while maintaining accuracy for attention.
        # This is what NVIDIA uses in their official Qwen3-8B-NVFP4 model.
        "--kv_cache_dtype", "fp8",
        # --calib_size 512: Number of calibration samples for activation scaling.
        # Quantization needs to know typical activation ranges - more samples = better scales.
        # 512 is good balance between accuracy and calibration time.
        "--calib_size", str(calib_size),
        # --batch_size 4: Batch size during calibration forward passes.
        # Larger = faster calibration but more VRAM. 4 is safe for most GPUs.
        "--batch_size", "4",
        # --calib_max_seq_length: Maximum sequence length during calibration.
        # Longer sequences = better scale estimation for long inputs.
        # Should match your expected input length distribution.
        # Default 512 is often too short; 2048 better captures email content.
        "--calib_max_seq_length", str(calib_max_seq_length),
    ]

    # --calib_dataset: Custom calibration dataset for domain-specific quantization.
    # Default is cnn_dailymail which works well for general text.
    # For email analysis, you could use a custom email dataset for better accuracy.
    # Format: HuggingFace dataset name (e.g., "openwebtext", "wikitext")
    if calib_dataset:
        cmd.extend(["--calib_dataset", calib_dataset])

    # NOTE: lm_head exclusion
    # In TensorRT-LLM 1.2.x, lm_head is NOT quantized by default.
    # Only add --quantize_lm_head if you explicitly want to quantize it.
    # This matches NVIDIA's official config where lm_head stays in full precision.
    # The lm_head converts hidden states to vocabulary logits - quantizing it
    # can cause significant accuracy degradation.

    run_command(cmd, "NVFP4 quantization (NVIDIA config: FP8 KV cache)")
    return quantized_dir


def build_engine(
    checkpoint_dir: Path,
    output_dir: Path,
    max_input_len: int = 4096,
    max_output_len: int = 512,
    max_batch_size: int = 32,
    max_num_tokens: int = 8192,
) -> Path:
    """
    Build TensorRT engine with Blackwell optimizations.

    SM120 (RTX PRO 4000) compatible configuration:
    - gemm_plugin nvfp4: Use Blackwell FP4 tensor cores
    - paged_kv_cache: Memory-efficient KV cache
    - remove_input_padding: Skip padding computation
    - use_fused_mlp: Fused feed-forward layers

    Note: Some optimizations (context_fmha, fuse_fp4_quant) are disabled
    because FMHA kernels for NVFP4+FP8 KV cache are not yet available
    on SM120 (consumer/workstation Blackwell) in TensorRT-LLM 1.2.x.
    """
    logger.info("Step 4/5: Building TensorRT engine for SM120 Blackwell")

    engine_dir = output_dir / "trt_engine"
    engine_dir.mkdir(parents=True, exist_ok=True)

    max_seq_len = max_input_len + max_output_len

    cmd = [
        "trtllm-build",
        "--checkpoint_dir", str(checkpoint_dir),
        "--output_dir", str(engine_dir),
        #
        # ============================================================
        # SEQUENCE CONFIGURATION
        # ============================================================
        #
        # --max_input_len: Maximum prompt/input length in tokens.
        # Affects memory allocation. Set to your longest expected input.
        "--max_input_len", str(max_input_len),
        # --max_seq_len: Total sequence length = input + output.
        # Engine reserves memory for this. Too high = wasted VRAM, too low = truncation.
        "--max_seq_len", str(max_seq_len),
        # --max_batch_size: Maximum concurrent requests in one batch.
        # Higher = better throughput but more VRAM. 32 is good for 8B model on 24GB GPU.
        "--max_batch_size", str(max_batch_size),
        # --max_num_tokens: Maximum total tokens across all sequences in a batch.
        # Limits memory for variable-length batches. 8192 = reasonable for batch of 32.
        "--max_num_tokens", str(max_num_tokens),
        #
        # ============================================================
        # NVFP4 BLACKWELL OPTIMIZATIONS
        # ============================================================
        #
        # --gemm_plugin nvfp4: Use Blackwell's native FP4 tensor cores for matrix ops.
        # GEMM (General Matrix Multiply) is the core operation in transformers.
        # FP4 tensor cores do 2x more ops/cycle than FP8, 4x more than FP16.
        "--gemm_plugin", "nvfp4",
        #
        # ============================================================
        # ATTENTION CONFIGURATION
        # ============================================================
        #
        # --gpt_attention_plugin: Custom attention kernels optimized for inference.
        # Handles KV cache updates, causal masking, rotary embeddings efficiently.
        # bfloat16 precision for attention scores (before softmax needs higher precision).
        "--gpt_attention_plugin", "bfloat16",
        #
        # SM120 COMPATIBILITY NOTE:
        # The following FMHA optimizations are DISABLED because TensorRT-LLM 1.2.x
        # does not have FMHA kernels for NVFP4 weights + FP8 KV cache on SM120.
        # These work on SM100 (B200 datacenter) but crash on SM120 (RTX PRO/5090).
        #
        # When TensorRT-LLM adds SM120 FMHA support, uncomment these for ~30% speedup:
        # "--context_fmha", "enable",
        # "--use_fp8_context_fmha", "enable",
        # "--fuse_fp4_quant", "enable",
        #
        # ============================================================
        # MEMORY OPTIMIZATIONS
        # ============================================================
        #
        # --kv_cache_type paged: Use PagedAttention for KV cache management.
        # Instead of pre-allocating max_seq_len for each request, allocates on-demand.
        # Enables much higher batch sizes without OOM. Essential for production.
        # Note: Replaces deprecated --paged_kv_cache flag.
        "--kv_cache_type", "paged",
        # --tokens_per_block: Block size for paged KV cache.
        # Larger blocks = less overhead, but more internal fragmentation.
        # 64 is optimal for Blackwell's memory hierarchy.
        "--tokens_per_block", "64",
        # --remove_input_padding: Pack tokens from different sequences together.
        # Without this: [seq1_tokens, PAD, PAD, seq2_tokens, PAD, PAD, PAD]
        # With this:    [seq1_tokens, seq2_tokens] - no wasted computation on padding.
        # Significant speedup for batches with variable-length inputs.
        "--remove_input_padding", "enable",
        #
        # ============================================================
        # COMPUTE OPTIMIZATIONS
        # ============================================================
        #
        # --use_fused_mlp: Fuse Feed-Forward Network (MLP) operations.
        # Transformer FFN: x -> Linear1 -> GELU/SiLU -> Linear2
        # Fused version does gate projection + activation in one kernel.
        # Reduces memory reads/writes between operations.
        "--use_fused_mlp", "enable",
        #
        # --norm_quant_fusion: Fuse LayerNorm and quantization into single kernel.
        # Normally: LayerNorm -> write to memory -> read -> quantize
        # Fused:    LayerNorm + quantize in one pass
        # Reduces memory bandwidth, improves latency.
        "--norm_quant_fusion", "enable",
        #
        # ============================================================
        # SINGLE-TURN INFERENCE OPTIMIZATIONS
        # ============================================================
        #
        # --max_beam_width 1: Disable beam search, use greedy decoding.
        # Beam search generates N candidate sequences in parallel and picks best.
        # For constrained JSON output (via schema), beam search is unnecessary -
        # the output structure is already determined by the JSON schema.
        # Greedy (beam_width=1) saves VRAM and is faster.
        "--max_beam_width", "1",
        #
        # ============================================================
        # BUILD SETTINGS
        # ============================================================
        #
        # --workers: Parallel compilation workers.
        # More workers = faster build but more RAM. 1 is safe, increase if you have RAM.
        "--workers", "1",
        #
        # --multiple_profiles: Enable multiple TensorRT optimization profiles.
        # TensorRT selects best kernels for different input sizes.
        # Trade-off: Longer build time (~2x), but better runtime kernel selection.
        # Especially beneficial when GEMM plugin is disabled or for variable batch sizes.
        "--multiple_profiles", "enable",
    ]

    run_command(cmd, "TensorRT engine build (SM120 compatible)")
    logger.info(f"Engine saved to: {engine_dir}")
    return engine_dir


def setup_triton_model_repository(
    engine_dir: Path,
    models_dir: Path,
    model_name: str = "qwen3",
) -> None:
    """
    Setup Triton Inference Server model repository.

    Structure:
    models/model_repository/
    └── qwen3/
        ├── 1/
        │   └── (engine files)
        └── config.pbtxt
    """
    logger.info("Step 5/5: Setting up Triton model repository")

    repo_dir = models_dir / "model_repository" / model_name
    version_dir = repo_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    # Copy engine files
    for engine_file in engine_dir.glob("*"):
        if engine_file.is_file():
            dest = version_dir / engine_file.name
            logger.info(f"  Copying: {engine_file.name}")
            shutil.copy2(engine_file, dest)

    # Create Triton config
    config_content = f'''name: "{model_name}"
backend: "tensorrtllm"
max_batch_size: 32

input [
  {{
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }},
  {{
    name: "input_lengths"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }},
  {{
    name: "request_output_len"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }}
]

output [
  {{
    name: "output_ids"
    data_type: TYPE_INT32
    dims: [ -1, -1 ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
'''

    config_path = repo_dir / "config.pbtxt"
    config_path.write_text(config_content)
    logger.info(f"  Triton config: {config_path}")


def main() -> None:
    """Main entry point."""
    config_path = Path(os.getenv("CONFIG_PATH", "/config")) / "app.yaml"
    models_dir = Path(os.getenv("MODELS_PATH", "/models"))
    hf_token = os.getenv("HF_TOKEN")

    logger.info("=" * 60)
    logger.info("TensorRT-LLM Model Builder (NVFP4 Blackwell)")
    logger.info("=" * 60)

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    model_config = config.get("model", {})
    inference_config = config.get("inference", {})

    repo_id = model_config.get("hf_repo", "Qwen/Qwen3-8B")
    model_name = inference_config.get("model_name", "qwen3").lower()
    max_input_len = model_config.get("max_input_tokens", 4096)
    max_output_len = model_config.get("max_output_tokens", 512)

    logger.info(f"Model: {repo_id}")
    logger.info(f"Quantization: NVFP4 (Blackwell)")
    logger.info(f"Max input: {max_input_len}, Max output: {max_output_len}")
    logger.info("")

    # Check if engine already exists
    engine_dir = models_dir / "trt_engine"
    if engine_dir.exists() and any(engine_dir.glob("*.engine")):
        logger.info("TensorRT engine already exists, skipping build")
        setup_triton_model_repository(engine_dir, models_dir, model_name)
        logger.info("Done!")
        return

    # Step 1: Download model
    logger.info("Step 1/5: Downloading model from Hugging Face")
    hf_model_dir = download_model(repo_id, models_dir, hf_token)

    # Step 2: Convert to TensorRT-LLM checkpoint
    checkpoint_dir = convert_checkpoint(hf_model_dir, models_dir)

    # Step 3: Quantize to NVFP4
    quantized_dir = quantize_nvfp4(checkpoint_dir, models_dir)

    # Step 4: Build TensorRT engine
    engine_dir = build_engine(
        checkpoint_dir=quantized_dir,
        output_dir=models_dir,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
    )

    # Step 5: Setup Triton model repository
    setup_triton_model_repository(engine_dir, models_dir, model_name)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Model initialization complete!")
    logger.info("=" * 60)
    logger.info(f"  Model: {repo_id}")
    logger.info(f"  Quantization: NVFP4 (weights) + FP8 (KV cache)")
    logger.info(f"  Excluded: lm_head (full precision)")
    logger.info(f"  Target: SM120 (RTX PRO 4000 / Blackwell workstation)")
    logger.info(f"  Engine: {engine_dir}")
    logger.info(f"  Triton repo: {models_dir / 'model_repository'}")


if __name__ == "__main__":
    main()
