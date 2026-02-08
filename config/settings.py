from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


class Settings:
    """Loads all configuration from config.yaml."""

    def __init__(self, config_path: str | Path | None = None):
        path = Path(config_path) if config_path else _CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            cfg: dict[str, Any] = yaml.safe_load(f)

        # API
        api = cfg.get("api", {})
        self.zenmux_base_url: str = api.get("base_url", "https://zenmux.ai/api/v1")
        self.zenmux_api_key: str = api.get("api_key", "")

        # Models
        models = cfg.get("models", {})
        self.gemini_model: str = models.get("ocr_model", "openai/gpt-5.2-chat")
        self.claude_model: str = models.get("agent_model", "anthropic/claude-sonnet-4")

        # Parser
        parser = cfg.get("parser", {})
        self.ocr_dpi: int = parser.get("dpi", 150)
        self.jpeg_quality: int = parser.get("jpeg_quality", 75)
        self.ocr_max_tokens: int = parser.get("max_tokens", 4096)

        # VectorDB
        vectordb = cfg.get("vectordb", {})
        self.chroma_dir: str = str(_PROJECT_ROOT / vectordb.get("persist_dir", "data/chroma_db"))
        self.collection_name: str = vectordb.get("collection_name", "hyperrag_docs")

        # Agent
        agent = cfg.get("agent", {})
        self.agent_max_iterations: int = agent.get("max_iterations", 10)

        # Paths
        paths = cfg.get("paths", {})
        self.upload_dir: str = str(_PROJECT_ROOT / paths.get("upload_dir", "data/uploads"))
        self.parsed_dir: str = str(_PROJECT_ROOT / paths.get("parsed_dir", "data/parsed"))
        self.highlighted_dir: str = str(_PROJECT_ROOT / paths.get("highlighted_dir", "data/highlighted"))
        self.prompts_dir: str = str(_PROJECT_ROOT / paths.get("prompts_dir", "prompts"))
