import httpx
from openai import OpenAI
from langchain_openai import ChatOpenAI


class LLMClientFactory:
    """Creates OpenAI-compatible clients pointing at ZenMux."""

    def __init__(self, api_key: str, base_url: str):
        self._api_key = api_key
        self._base_url = base_url

    def get_openai_client(self) -> OpenAI:
        """Raw OpenAI client for direct API calls (Gemini OCR with vision)."""
        return OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=httpx.Timeout(300.0, connect=30.0),
            max_retries=3,
        )

    def get_langchain_llm(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ChatOpenAI:
        """LangChain-compatible chat model for agent use."""
        return ChatOpenAI(
            model=model,
            api_key=self._api_key,
            base_url=self._base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
