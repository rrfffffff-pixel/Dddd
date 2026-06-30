"""Configuration management."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    provider: str = "ollama"
    model: str = "qwen3:1.7b"
    api_key: str = ""
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 4096


@dataclass
class SandboxConfig:
    enabled: bool = True
    engine: str = "nsjail"  # nsjail, docker, subprocess
    timeout: int = 30
    memory_limit: str = "512m"


@dataclass
class IntelligenceConfig:
    embedding_model: str = "nomic-embed-text"
    index_path: str = ".coding-agent/index"
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_context_tokens: int = 12000


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    intelligence: IntelligenceConfig = field(default_factory=IntelligenceConfig)
    project_root: str = "."
    verbose: bool = False

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        config = cls()

        # Try loading from file
        if path is None:
            candidates = [
                Path("coding-agent.yaml"),
                Path("coding-agent.yml"),
                Path(".coding-agent/config.yaml"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    path = candidate
                    break

        if path and Path(path).exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            if "model" in data:
                for k, v in data["model"].items():
                    if hasattr(config.model, k):
                        setattr(config.model, k, v)
            if "sandbox" in data:
                for k, v in data["sandbox"].items():
                    if hasattr(config.sandbox, k):
                        setattr(config.sandbox, k, v)
            if "intelligence" in data:
                for k, v in data["intelligence"].items():
                    if hasattr(config.intelligence, k):
                        setattr(config.intelligence, k, v)

        # Environment overrides
        if api_key := os.environ.get("OPENAI_API_KEY"):
            config.model.api_key = api_key
        if model := os.environ.get("CODING_AGENT_MODEL"):
            config.model.model = model

        return config
