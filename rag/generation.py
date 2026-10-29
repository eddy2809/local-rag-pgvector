"""Grounded answer generation with qwen3 through Ollama."""

import re
import time
from dataclasses import dataclass

import ollama

from rag.config import LLM_MODEL, NO_ANSWER
from rag.embeddings import KEEP_ALIVE
from rag.retrieval import RetrievedChunk

SYSTEM_PROMPT = """You are an assistant that answers questions about a collection of documents.

Rules:
- Use only the information contained in the sources provided in the user message. Do not use prior knowledge.
- After each statement, cite the sources that support it with their id in square brackets, e.g. [2] or [1][3].
- If the sources do not contain the answer, reply exactly with: {no_answer}
- Answer in the same language as the question, concisely, in plain text without markdown."""

GENERATION_OPTIONS = {"temperature": 0.1, "seed": 42}


@dataclass
class Answer:
    text: str
    cited_ids: list[int]   # 1-based ids of the sources cited in the answer
    seconds: float
    prompt_tokens: int
    output_tokens: int
    tokens_per_second: float

    @property
    def is_abstention(self) -> bool:
        return self.text.strip() == NO_ANSWER


def format_sources(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f'<source id="{i}" pages="{c.pages}">\n{c.content}\n</source>'
        for i, c in enumerate(chunks, start=1)
    )


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    user_message = f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(no_answer=NO_ANSWER)},
        {"role": "user", "content": user_message},
    ]


def warm_up() -> None:
    """Load the LLM into VRAM so that the first timed call is not a cold start."""
    ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": "ok"}],
                think=False, keep_alive=KEEP_ALIVE, options={"num_predict": 1})


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> Answer:
    t0 = time.perf_counter()
    response = ollama.chat(
        model=LLM_MODEL,
        messages=build_messages(question, chunks),
        think=False,
        keep_alive=KEEP_ALIVE,
        options=GENERATION_OPTIONS,
    )
    seconds = time.perf_counter() - t0

    text = response.message.content.strip()
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(chunks)})
    eval_seconds = (response.eval_duration or 0) / 1e9
    return Answer(
        text=text,
        cited_ids=cited,
        seconds=seconds,
        prompt_tokens=response.prompt_eval_count or 0,
        output_tokens=response.eval_count or 0,
        tokens_per_second=(response.eval_count or 0) / eval_seconds if eval_seconds else 0.0,
    )