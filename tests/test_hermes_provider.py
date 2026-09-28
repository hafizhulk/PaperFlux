import asyncio
import json

import fitz
import pytest
import yaml

from paperflux.config import Config, load
from paperflux.providers import available_providers, get_provider
from paperflux.providers import hermes_provider as hp

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]
PROMPTS = REPO / "prompts"


def _write(path, content):
    path.write_text(content, encoding="utf-8")


def _paperflux_config(tmp_path, hermes_block=""):
    return f"""provider: "hermes"
{hermes_block}
ui:
  detail_level: "medium"
  reasoning_effort: "low"
  max_output_tokens: 4096
  highlight_colors:
    contributions: [1.0, 1.0, 0.0]
    limitations:   [1.0, 0.6, 0.0]
    claims:        [0.2, 0.4, 1.0]
    evidence:      [0.0, 0.8, 0.3]

matching:
  min_similarity: 0.9
  max_window_tokens: 120

rag:
  category_prompt_file: "{PROMPTS / 'rag_category_prompt.j2'}"
  summary_prompt_file: "{PROMPTS / 'rag_summary_prompt.j2'}"
  category_system_prompt_file: "{PROMPTS / 'rag_category_system_prompt.txt'}"
  max_quotes_per_category: 4
"""


def test_hermes_registered():
    assert "hermes" in available_providers()
    assert isinstance(get_provider("hermes"), hp.HermesProvider)


def test_hermes_config_parses_without_block(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    _write(cfg_file, _paperflux_config(tmp_path))
    cfg = load(cfg_file)
    assert isinstance(cfg, Config)
    assert cfg.provider == "hermes"
    assert cfg.hermes is not None
    assert cfg.hermes.model is None


def _fake_hermes_home(tmp_path, key_env="PAPERFLUX_TEST_HERMES_KEY"):
    home = tmp_path / "hermes-home"
    home.mkdir()
    _write(home / "config.yaml", yaml.safe_dump({
        "model": {
            "default": "test-model",
            "provider": "test-provider",
            "base_url": "https://llm.example.test/v1",
            "key_env": key_env,
        },
        "providers": {
            "alt": {
                "base_url": "https://alt.example.test/v1",
                "model": "alt-model",
                "key_env": key_env,
            }
        },
    }))
    _write(home / ".env", f"{key_env}=secret-123\n")
    return home


def test_resolve_autodetects_any_hermes_home(tmp_path, monkeypatch):
    home = _fake_hermes_home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("PAPERFLUX_TEST_HERMES_KEY", raising=False)
    cfg_file = tmp_path / "config.yaml"
    _write(cfg_file, _paperflux_config(tmp_path))
    conn = hp.resolve_connection(load(cfg_file))
    assert conn.base_url == "https://llm.example.test/v1"
    assert conn.model == "test-model"
    assert conn.api_key == "secret-123"


def test_resolve_named_hermes_provider(tmp_path, monkeypatch):
    home = _fake_hermes_home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("PAPERFLUX_TEST_HERMES_KEY", raising=False)
    cfg_file = tmp_path / "config.yaml"
    _write(cfg_file, _paperflux_config(tmp_path, 'hermes:\n  provider: "alt"\n'))
    conn = hp.resolve_connection(load(cfg_file))
    assert conn.base_url == "https://alt.example.test/v1"
    assert conn.model == "alt-model"


def test_resolve_explicit_overrides_need_no_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "nonexistent"))
    cfg_file = tmp_path / "config.yaml"
    _write(cfg_file, _paperflux_config(tmp_path, (
        "hermes:\n"
        '  base_url: "https://override.example.test/v1"\n'
        '  model: "override-model"\n'
        '  api_key: "override-key"\n'
    )))
    conn = hp.resolve_connection(load(cfg_file))
    assert (conn.base_url, conn.model, conn.api_key) == (
        "https://override.example.test/v1", "override-model", "override-key")


def test_resolve_missing_key_raises(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    _write(home / "config.yaml", yaml.safe_dump({
        "model": {"default": "m", "base_url": "https://x.test/v1",
                  "key_env": "PAPERFLUX_TEST_MISSING_KEY"},
    }))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("PAPERFLUX_TEST_MISSING_KEY", raising=False)
    cfg_file = tmp_path / "config.yaml"
    _write(cfg_file, _paperflux_config(tmp_path))
    with pytest.raises(ValueError, match="api_key"):
        hp.resolve_connection(load(cfg_file))


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)
        self.finish_reason = "stop"


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._script.pop(0)


class _FakeClient:
    instances = []
    script: list = []

    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeCompletions(list(_FakeClient.script))
        _FakeClient.instances.append(self)


QUOTE = "The membrane removed 99.2 percent of microplastics in pilot trials."


def _sample_pdf(path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), f"Introduction. {QUOTE} Further discussion follows.")
    doc.save(path)
    doc.close()


def test_analyze_pdf_end_to_end(tmp_path, monkeypatch):
    home = _fake_hermes_home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("PAPERFLUX_TEST_HERMES_KEY", raising=False)
    monkeypatch.setattr(hp, "AsyncOpenAI", _FakeClient)
    _FakeClient.instances.clear()
    _FakeClient.script = [
        _FakeResp(json.dumps({"categories": [{
            "name": "contributions",
            "quotes": [{"text": QUOTE, "pages": [1],
                        "prefix": "Introduction.", "suffix": "Further"}],
            "category_summary": "Pilot-scale removal result.",
        }]})),
        _FakeResp("## Takeaways\nPilot trials look promising."),
    ]
    cfg_file = tmp_path / "config.yaml"
    _write(cfg_file, _paperflux_config(tmp_path))
    pdf = tmp_path / "paper.pdf"
    _sample_pdf(pdf)

    result = asyncio.run(
        hp.HermesProvider().analyze_pdf(pdf, load(cfg_file))
    )
    assert result["key_takeaways"].startswith("## Takeaways")
    assert result["quotes"]["contributions"][0]["text"] == QUOTE
    client = _FakeClient.instances[0]
    assert client.base_url == "https://llm.example.test/v1"
    assert client.chat.completions.calls[0].get("response_format") == {
        "type": "json_object"}
