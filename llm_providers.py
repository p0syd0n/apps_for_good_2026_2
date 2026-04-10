"""
Provider abstraction for LLM and embedding calls, to hopefully swap providers without touching inference logic.
"""

from abc import ABC, abstractmethod
from typing import Optional
import os

# Templatish thing
class LLMProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a unit-normalised embedding vector"""
        ...

    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Return a text response"""
        ...

#inherit 
class GeminiProvider(LLMProvider):
    EMBED_MODEL  = "gemini-embedding-001"
    GEN_MODEL    = "gemini-2.5-flash"

    # import google things and create instance
    # it needs api keys but they're automatically passed because the .env is loaded
    def __init__(self):
        from google import genai
        self._client = genai.Client()


    def embed(self, text: str) -> list[float]:
        result = self._client.models.embed_content(
            model=self.EMBED_MODEL,
            contents=text,
        )
        return result.embeddings[0].values

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        contents = prompt if system is None else f"{system}\n\n{prompt}"
        response = self._client.models.generate_content(
            model=self.GEN_MODEL,
            contents=contents,
        )
        return response.text

class ClaudeProvider(LLMProvider):
    # Claude doesn't expose a public embedding API yet;
    # fall back to a local sentence-transformer for embeddings.
    GEN_MODEL   = "claude-3-5-haiku-20241022"
    EMBED_MODEL = "BAAI/bge-m3"          # local fallback

    # anthropic protocl
    def __init__(self):
        import anthropic
        from sentence_transformers import SentenceTransformer
        self._client     = anthropic.Anthropic()
        self._embed_model = SentenceTransformer(self.EMBED_MODEL)

    def embed(self, text: str) -> list[float]:
        return self._embed_model.encode(text, normalize_embeddings=True).tolist()

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        kwargs = {"system": system} if system else {}
        message = self._client.messages.create(
            model=self.GEN_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return message.content[0].text


class OpenAIProvider(LLMProvider):
    EMBED_MODEL = "text-embedding-3-small"
    GEN_MODEL   = "gpt-4o-mini"

    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI()

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            model=self.EMBED_MODEL,
            input=text,
        )
        return response.data[0].embedding

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self.GEN_MODEL,
            messages=messages,
        )
        return response.choices[0].message.content


# get one of the providers
def get_provider(name: Optional[str] = None) -> LLMProvider:
    """
    Return a provider instance.  Priority:
      1. `name` argument
      2. LLM_PROVIDER env var
      3. 'gemini' default
    """
    name = (name or os.getenv("LLM_PROVIDER", "gemini")).lower()
    match name:
        case "gemini":  return GeminiProvider()
        case "claude":  return ClaudeProvider()
        case "openai":  return OpenAIProvider()
        case _: raise ValueError(f"Unknown provider: {name!r}. Choose gemini | claude | openai")