# Model download and conversion container
# Using TensorRT-LLM release container which includes:
# - tensorrt_llm (pre-installed)
# - transformers (pre-installed)
# - huggingface_hub (pre-installed)
# This eliminates redundant pip installs and ensures version compatibility
# https://github.com/NVIDIA/TensorRT-LLM/releases
FROM nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc8

WORKDIR /app

# Set LD_LIBRARY_PATH for TensorRT libraries
# libnvinfer.so.10 is in /usr/local/tensorrt/targets/x86_64-linux-gnu/lib/
# libnvinfer_plugin_tensorrt_llm.so is in /app/tensorrt_llm/lib/
ENV LD_LIBRARY_PATH="/usr/local/tensorrt/targets/x86_64-linux-gnu/lib:\
/app/tensorrt_llm/lib:\
/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs:\
/usr/local/cuda/lib64:\
${LD_LIBRARY_PATH}"

# Only install pyyaml (not included in base image)
RUN pip install --no-cache-dir pyyaml

# Copy download script
COPY scripts/download_model.py .

# Create user and set permissions
# Note: /app needs write access for trtllm-build timing cache (model.cache)
RUN useradd --create-home appuser && \
    mkdir -p /models && \
    chown -R appuser:appuser /models /app
USER appuser

# Entry point
ENTRYPOINT ["python", "download_model.py"]
