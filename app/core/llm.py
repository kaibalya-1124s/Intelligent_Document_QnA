# app/core/llm.py
import os
from typing import List

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

def _build_prompt(question: str, contexts: List[str]) -> str:
    ctx = "\n\n".join([f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts)])
    prompt = (
        "You are an assistant that answers questions using ONLY the provided context. "
        "If the answer is not contained in the context, respond with: 'I don't know based on the provided documents.'\n\n"
        f"{ctx}\n\nQuestion: {question}\nAnswer:"
    )
    return prompt

def generate_answer(question: str, contexts: List[str], max_tokens: int = 256) -> str:
    """
    Returns the model-generated answer string.
    contexts: list of text snippets retrieved from documents (ordered by relevance)
    """
    prompt = _build_prompt(question, contexts)
    if OPENAI_API_KEY:
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            # fail-safe: return context summary if API call fails
            return f"[LLM error: {e}]\n\nContext preview:\n" + "\n\n".join(c for c in contexts[:3])
    else:
        # No API key: basic heuristic fallback (concatenate contexts)
        preview = "\n\n".join(contexts[:3])
        return f"(No OPENAI_API_KEY set) Context preview:\n{preview}\n\nQuestion: {question}"

