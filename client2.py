import base64
from pathlib import Path

from openai import OpenAI

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:9010/v1"


def encode_image(path: str) -> str:
    """Read image as base64 string."""
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


encoded = encode_image("/public/fengyupu/github/FlagScale/affordance.jpg")

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            },
            {"type": "text", "text": "the affordance area for holding the cup"},
        ],
    }
]

response = client.chat.completions.create(
    model="",
    messages=[
        {
            "role": "user",
            "content": "Give me a short introduction to large language models.",
        },
    ],
    max_tokens=1024,
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    stream=True,
)

for chunk in response:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
