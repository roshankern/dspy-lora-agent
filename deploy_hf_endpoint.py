# ---
# pytest: false
# ---

# # Run OpenAI-compatible LLM inference with LLaMA 3.1-8B and vLLM

# LLMs do more than just model language: they chat, they produce JSON and XML, they run code, and more.
# This has complicated their interface far beyond "text-in, text-out".
# OpenAI's API has emerged as a standard for that interface,
# and it is supported by open source LLM serving frameworks like [vLLM](https://docs.vllm.ai/en/latest/).

# In this example, we show how to run a vLLM server in OpenAI-compatible mode on Modal.

# Our examples repository also includes scripts for running clients and load-testing for OpenAI-compatible APIs
# [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving/openai_compatible).

# You can find a (somewhat out-of-date) video walkthrough of this example and the related scripts on the Modal YouTube channel
# [here](https://www.youtube.com/watch?v=QmY_7ePR1hM).

# ## Set up the container image

# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/custom-container).
# vLLM can be installed with `pip`, since Modal [provides the CUDA drivers](https://modal.com/docs/guide/cuda).

# To take advantage of optimized kernels for CUDA 12.8, we install PyTorch, flashinfer, and their dependencies
# via an `extra` Python package index.

import json
from typing import Any

import aiohttp
import modal

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.9.1",
        "huggingface_hub[hf_transfer]==0.32.0",
        "flashinfer-python==0.2.6.post1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)

# ## Download the model weights

# We'll be running a pretrained foundation model -- Meta's LLaMA 3.1 8B
# in the Instruct variant that's trained to chat and follow instructions.

# Model parameters are often quantized to a lower precision during training
# than they are run at during inference.
# We'll use an eight bit floating point quantization from Neural Magic/Red Hat.
# Native hardware support for FP8 formats in [Tensor Cores](https://modal.com/gpu-glossary/device-hardware/tensor-core)
# is limited to the latest [Streaming Multiprocessor architectures](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor-architecture),
# like those of Modal's [Hopper H100/H200 and Blackwell B200 GPUs](https://modal.com/blog/announcing-h200-b200).

# You can swap this model out for another by changing the strings below.
# A single B200 GPUs has enough VRAM to store a 70,000,000,000 parameter model,
# like Llama 3.3, in eight bit precision, along with a very large KV cache.

MODEL_NAME = "rshn-krn/hotpotqa-agent-sft-llm"

# Although vLLM will download weights from Hugging Face on-demand,
# we want to cache them so we don't do it every time our server starts.
# We'll use [Modal Volumes](https://modal.com/docs/guide/volumes) for our cache.
# Modal Volumes are essentially a "shared disk" that all Modal Functions can access like it's a regular disk. For more on storing model weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).


hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# We'll also cache some of vLLM's JIT compilation artifacts in a Modal Volume.

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# ## Configuring vLLM

# ### The V1 engine

# In its 0.7 release, in early 2025, vLLM added a new version of its backend infrastructure,
# the [V1 Engine](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html).
# Using this new engine can lead to some [impressive speedups](https://github.com/modal-labs/modal-examples/pull/1064).
# It was made the default in version 0.8 and is [slated for complete removal by 0.11](https://github.com/vllm-project/vllm/issues/18571),
# in late summer of 2025.

# A small number of features, described in the RFC above, may still require the V0 engine prior to removal.
# Until deprecation, you can use it by setting the below environment variable to `0`.

vllm_image = vllm_image.env({"VLLM_USE_V1": "1"})

# ### Trading off fast boots and token generation performance

# vLLM has embraced dynamic and just-in-time compilation to eke out additional performance without having to write too many custom kernels,
# e.g. via the Torch compiler and CUDA graph capture.
# These compilation features incur latency at startup in exchange for lowered latency and higher throughput during generation.
# We make this trade-off controllable with the `FAST_BOOT` variable below.

FAST_BOOT = True

# If you're running an LLM service that frequently scales from 0 (frequent ["cold starts"](https://modal.com/docs/guide/cold-start))
# then you'll want to set this to `True`.

# If you're running an LLM service that usually has multiple replicas running, then set this to `False` for improved performance.


app = modal.App("hf-endpoint")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"A100:{N_GPU}",
    scaledown_window=5 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=32
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve_original_model():
    import subprocess

    MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        # "--revision",
        # MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # assume multiple GPUs are for splitting up large matrix multiplications
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)


@app.function(
    image=vllm_image,
    gpu=f"A100:{N_GPU}",
    scaledown_window=5 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=32
)
@modal.web_server(port=VLLM_PORT + 1, startup_timeout=10 * MINUTES)
def serve_sft_model():
    import subprocess

    MODEL_NAME = "rshn-krn/hotpotqa-agent-sft-llm"

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        # "--revision",
        # MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # assume multiple GPUs are for splitting up large matrix multiplications
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)
