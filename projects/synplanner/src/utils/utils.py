from typing import Callable, List, Dict, Any, Optional, Tuple
import os
import logging
from openai import OpenAI


def setup_logging(logging_path: str) -> None:
    """
    Setup file and console logging.
    Adapted from https://github.com/schwallergroup/saturn/blob/master/utils/utils.py.
    """
    logging.basicConfig(
        filename=logging_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logging.getLogger("").addHandler(console)

    # Silence OpenAI HTTP request logs
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def query_LLM(
    query: str,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 8192,
    on_chunk: Optional[Callable[[str], None]] = None,
    stream: bool = True,
) -> Tuple[List[Dict[str, Any]], str]:
    """Query OpenAI API for retrosynthesis planning.

    Args:
        query: The user query string.
        model: OpenAI model name.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens for response.
        on_chunk: Optional callback invoked with each text delta during streaming.
        stream: Whether to use streaming (default True).

    Returns:
        Tuple of (messages list, content string).
    """
    messages = [
        {
            "role": "system",
            "content": "You are a retrosynthesis agent who can make multi-step retrosynthesis plans based on your molecule knowledge.",
        },
        {"role": "user", "content": query},
    ]

    if stream:
        # Streaming mode (default)
        resp = CLIENT.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        text_parts = []
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                delta_content = chunk.choices[0].delta.content
                text_parts.append(delta_content)
                if on_chunk is not None:
                    on_chunk(delta_content)
        content = "".join(text_parts)
    else:
        # Non-streaming mode
        resp = CLIENT.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content

    messages.append({"role": "assistant", "content": content})
    return messages, content
