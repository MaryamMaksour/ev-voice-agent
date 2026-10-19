"""vLLM in production, Ollama for testing: only LLM_BASE_URL / LLM_MODEL change in .env."""

import pytest

from agent.config import Settings, llm_backend, no_think_body, no_think_prompt_suffix


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)


@pytest.mark.parametrize("url, backend", [
    ("http://localhost:11434/v1", "ollama"),
    ("http://127.0.0.1:11434/v1", "ollama"),
    ("http://10.0.0.5:8000/v1", "vllm"),
    ("http://localhost:8000/v1", "vllm"),
])
def test_backend_follows_the_url(monkeypatch, url, backend):
    monkeypatch.setenv("LLM_BASE_URL", url)
    assert llm_backend() == backend
    assert Settings().llm_backend == backend


def test_backend_can_be_forced(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://gpu-box:9999/v1")
    monkeypatch.setenv("LLM_BACKEND", "Ollama")
    assert Settings().llm_backend == "ollama"
    monkeypatch.setenv("LLM_BACKEND", "tgi")
    with pytest.raises(ValueError):
        llm_backend()


def test_thinking_is_turned_off_the_way_each_server_understands():
    assert no_think_body("qwen3.5-9b", "vllm") == {"chat_template_kwargs": {"enable_thinking": False}}
    # Ollama ignores "think": false on its OpenAI API for qwen3.5; reasoning_effort "none" works
    assert no_think_body("qwen3.5:9b", "ollama") == {"think": False, "reasoning_effort": "none"}
    assert no_think_body("llama3.1:8b", "ollama") == {}  # not a thinking model: send nothing extra
    assert no_think_prompt_suffix("qwen3.5:9b", "ollama") == "\n/no_think"
    assert no_think_prompt_suffix("qwen3.5-9b", "vllm") == ""


def test_default_is_production_vllm():
    s = Settings()
    assert s.llm_backend == "vllm" and s.llm_base_url.endswith(":8000/v1")
