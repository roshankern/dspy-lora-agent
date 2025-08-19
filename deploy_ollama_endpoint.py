"""FastAPI endpoint for Ollama chat completions with OpenAI-compatible API.

This module provides a FastAPI application that serves as a bridge between
clients and Ollama models, offering an OpenAI-compatible API interface.
It supports both streaming and non-streaming responses.
"""

import modal
import os
import subprocess
import time
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Any, Optional, AsyncGenerator
from pydantic import BaseModel, Field
import json


# Configuration
MODEL = os.environ.get("MODEL", "llama3.2:3b")
DEFAULT_MODELS = ["llama3.2:1b", "llama3.2:3b"]
API_KEY = "makora_bio_endpoint"  # Fixed API key as requested

# Security
security = HTTPBearer()


def pull() -> None:
    """Initialize and pull the Ollama model.

    Sets up the Ollama service using systemctl and pulls the specified model.
    """
    try:
        # Create systemd service file at runtime
        service_content = """[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
"""
        with open("/etc/systemd/system/ollama.service", "w") as f:
            f.write(service_content)

        subprocess.run(["systemctl", "daemon-reload"], check=False)
        subprocess.run(["systemctl", "enable", "ollama"], check=False)
        subprocess.run(["systemctl", "start", "ollama"], check=False)
        wait_for_ollama()

        # Pull all default models
        for model in DEFAULT_MODELS:
            print(f"Pulling model: {model}")
            subprocess.run(
                ["ollama", "pull", model], stdout=subprocess.PIPE, check=True
            )
            print(f"Successfully pulled: {model}")
    except Exception as e:
        print(f"Error during pull: {e}")
        raise


def wait_for_ollama(timeout: int = 30, interval: int = 2) -> None:
    """Wait for Ollama service to be ready.

    :param timeout: Maximum time to wait in seconds
    :param interval: Time between checks in seconds
    :raises TimeoutError: If the service doesn't start within the timeout period
    """
    import httpx

    # Use print instead of loguru for simplicity
    start_time = time.time()
    while True:
        try:
            response = httpx.get("http://localhost:11434/api/version")
            if response.status_code == 200:
                print("Ollama service is ready")
                return
        except httpx.ConnectError:
            if time.time() - start_time > timeout:
                raise TimeoutError("Ollama service failed to start")
            print(f"Waiting for Ollama service... ({int(time.time() - start_time)}s)")
            time.sleep(interval)


# Configure Modal image with Ollama dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("curl", "systemctl")
    .run_commands(  # from https://github.com/ollama/ollama/blob/main/docs/linux.md
        "curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz",
        "tar -C /usr -xzf ollama-linux-amd64.tgz",
        "useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama",
        "usermod -a -G ollama $(whoami)",
    )
    .pip_install("ollama", "httpx", "loguru", "fastapi")
    .run_function(pull)
)

app = modal.App(name="ollama-endpoint", image=image)
api = FastAPI()


class ChatMessage(BaseModel):
    """A single message in a chat completion request.

    Represents one message in the conversation history, following OpenAI's chat format.
    """

    role: str = Field(
        ..., description="The role of the message sender (e.g. 'user', 'assistant')"
    )
    content: str = Field(..., description="The content of the message")


class ChatCompletionRequest(BaseModel):
    """Request model for chat completions.

    Follows OpenAI's chat completion request format, supporting both streaming
    and non-streaming responses.
    """

    model: Optional[str] = Field(
        default=MODEL, description="The model to use for completion"
    )
    messages: List[ChatMessage] = Field(
        ..., description="The messages to generate a completion for"
    )
    stream: bool = Field(default=False, description="Whether to stream the response")
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum number of tokens to generate"
    )
    temperature: Optional[float] = Field(
        default=None, description="Sampling temperature"
    )
    top_p: Optional[float] = Field(default=None, description="Nucleus sampling")


def verify_api_key(credentials: HTTPAuthorizationCredentials) -> bool:
    """Verify the API key."""
    return credentials.credentials == API_KEY


@api.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Authentication middleware to check API key."""
    # Skip auth for docs and openapi endpoints
    if request.url.path in ["/docs", "/openapi.json", "/", "/health"]:
        response = await call_next(request)
        return response

    # Check for Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header",
        )

    # Extract and verify token
    token = auth_header.split(" ")[1]
    if token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )

    response = await call_next(request)
    return response


@api.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Ollama OpenAI-compatible API", "status": "running"}


@api.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@api.get("/v1/models")
async def list_models():
    """List available models in OpenAI format."""
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama",
            }
            for model in DEFAULT_MODELS
        ],
    }


@api.post("/v1/chat/completions")
async def v1_chat_completions(request: ChatCompletionRequest) -> Any:
    """Handle chat completion requests in OpenAI-compatible format.

    :param request: Chat completion parameters
    :return: Chat completion response in OpenAI-compatible format, or StreamingResponse if streaming
    :raises HTTPException: If the request is invalid or processing fails
    """
    import ollama  # Import here to ensure it's available in the Modal container

    try:
        if not request.messages:
            raise HTTPException(
                status_code=400,
                detail="Messages array is required and cannot be empty",
            )

        # Print conversation for debugging
        print("=== CHAT COMPLETION REQUEST ===")
        for i, msg in enumerate(request.messages):
            if msg.role == "system":
                print(f"SYSTEM PROMPT: {msg.content}")
            elif msg.role == "user":
                print(f"USER PROMPT: {msg.content}")
            elif msg.role == "assistant":
                print(f"ASSISTANT MSG {i}: {msg.content}")
        print("=" * 30)

        # Prepare Ollama request parameters
        ollama_params = {
            "model": request.model,
            "messages": [msg.dict() for msg in request.messages],
            "stream": request.stream,
        }

        # Add optional parameters if provided
        options = {}
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_p is not None:
            options["top_p"] = request.top_p

        if options:
            ollama_params["options"] = options

        if request.stream:

            async def generate_stream() -> AsyncGenerator[str, None]:
                """Generate streaming response chunks.

                :return: AsyncGenerator yielding SSE-formatted JSON strings
                """
                response = ollama.chat(**ollama_params)
                full_response = ""

                for chunk in response:
                    if chunk.get("message", {}).get("content"):
                        content = chunk["message"]["content"]
                        full_response += content
                        chunk_data = {
                            "id": "chatcmpl-" + str(int(time.time())),
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": content,
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"

                # Print complete response for debugging
                print(f"ASSISTANT RESPONSE: {full_response}")
                print("=" * 30)

                # Send final chunk with finish_reason
                final_chunk = {
                    "id": "chatcmpl-" + str(int(time.time())),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        # Non-streaming response
        response = ollama.chat(**ollama_params)

        # Print response for debugging
        assistant_response = response["message"]["content"]
        print(f"ASSISTANT RESPONSE: {assistant_response}")
        print("=" * 30)

        return {
            "id": "chatcmpl-" + str(int(time.time())),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": assistant_response,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": -1,  # Ollama doesn't provide token counts
                "completion_tokens": -1,
                "total_tokens": -1,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing chat completion: {str(e)}"
        )


@app.function(
    gpu="A10G",
    timeout=900,  # 15 minutes
)
@modal.asgi_app()
def ollama_api():
    """Serve the FastAPI application as a Modal function.

    :return: FastAPI application instance
    """
    # Initialize Ollama when the function starts
    try:
        subprocess.run(["systemctl", "start", "ollama"], check=False)
        wait_for_ollama()

        # Pull all default models
        for model in DEFAULT_MODELS:
            print(f"Pulling model: {model}")
            subprocess.run(["ollama", "pull", model], check=True)
            print(f"Successfully pulled: {model}")

        print(f"Ollama service started and models {DEFAULT_MODELS} loaded successfully")
    except Exception as e:
        print(f"Error starting Ollama: {e}")
        raise

    return api


if __name__ == "__main__":
    # Deploy the app
    print("Deploying Ollama endpoint to Modal...")
    print("Use the deployed URL with DSPY like this:")
    print(
        'lm = dspy.LM("openai/llama3.2:1b", api_key="makora_bio_endpoint", api_base="<YOUR_MODAL_URL>/v1")'
    )
