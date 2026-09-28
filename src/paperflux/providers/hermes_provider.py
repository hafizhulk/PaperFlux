"""Hermes backend: reuse any Hermes Agent LLM via its OpenAI-compatible API.

This provider is generic — it is not tied to one specific Hermes setup.
It reads the Hermes configuration of whichever machine it runs on:

1. Hermes home: ``hermes.home`` from PaperFlux config, else ``$HERMES_HOME``,
   else ``~/.hermes`` (profiles included, e.g. ``~/.hermes/profiles/<name>``).
2. Connection: ``<home>/config.yaml`` — top-level ``model`` (``base_url``,
   ``key_env``, ``default``) or a named entry under ``providers`` when
   ``hermes.provider`` is set. Explicit ``hermes.*`` fields in PaperFlux
   config always win over auto-detection.
3. API key: ``hermes.api_key``, else ``$<key_env>`` from the environment,
   else ``<home>/.env``.

Because Hermes endpoints are OpenAI-compatible, the already-required
``openai`` SDK is reused — no extra dependency. Unlike the OpenAI backend
(which needs the Responses API + a server-side vector store) the PDF text
is extracted locally with PyMuPDF and sent in-context, so this works with
any model Hermes is configured to use.
"""

import json
import logging
import os
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional

import fitz  # PyMuPDF
import yaml
from openai import AsyncOpenAI

from ..config import Config
from .base import (
    ProgressCallback,
    dump_failed_response,
    load_template,
    load_text_file,
    normalize_category_bundle,
    resolve_config_path,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HermesConnection:
    """Resolved OpenAI-compatible endpoint for a Hermes installation."""

    base_url: str
    api_key: str
    model: str
    api_mode: str = "chat_completions"
    extra_headers: Dict[str, str] = field(default_factory=dict)


def _read_dotenv(path: Path) -> Dict[str, str]:
    """Parse a dotenv file into a dict (no interpolation, no export handling)."""
    values: Dict[str, str] = {}
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("\"'")
        if key:
            values[key] = val
    return values


def _pool_key_env(home: Path, base_url: Optional[str],
                  *names: Optional[str]) -> Optional[str]:
    """Find an ``env:VAR`` credential source in Hermes' credential pool.

    The pool is Hermes' live key rotation: it wins over the key named in
    ``config.yaml``, which may be stale. Candidates score by base-URL match,
    then provider-name match, preferring entries whose last probe was ok.
    """
    try:
        pool = json.loads((home / "auth.json").read_text(encoding="utf-8"))
        pool = pool.get("credential_pool") or {}
    except (OSError, ValueError):
        return None
    wanted = {n for n in names if n}
    best = None
    for pool_name, creds in pool.items():
        if not creds or not isinstance(creds, list):
            continue
        entry = creds[0] if isinstance(creds[0], dict) else {}
        source = entry.get("source") or ""
        if not source.startswith("env:"):
            continue
        if base_url and entry.get("base_url") == base_url:
            score = 2
        elif pool_name in wanted:
            score = 1
        else:
            continue
        if entry.get("last_status") == "ok":
            score += 1
        if best is None or score > best[0]:
            best = (score, source[len("env:"):])
    return best[1] if best else None


def _lookup_key(key_env: Optional[str], dotenv: Dict[str, str]) -> Optional[str]:
    """Return ``$<key_env>`` from the environment or the Hermes dotenv."""
    if not key_env:
        return None
    return os.environ.get(key_env) or dotenv.get(key_env)


def resolve_hermes_home(explicit_home: Optional[str] = None) -> Path:
    """Return the Hermes home directory, honouring explicit config first."""
    if explicit_home:
        return Path(explicit_home).expanduser()
    env_home = os.environ.get("HERMES_HOME")
    if env_home:
        return Path(env_home).expanduser()
    return Path.home() / ".hermes"


def resolve_connection(cfg: Config) -> HermesConnection:
    """Resolve base_url/api_key/model for any Hermes installation.

    Precedence per field: PaperFlux ``hermes:`` block > Hermes ``config.yaml``
    > ``HERMES_*`` environment. Raises ``ValueError`` with an actionable
    message when something is missing.
    """
    hermes_cfg = getattr(cfg, "hermes", None)
    home = resolve_hermes_home(
        getattr(hermes_cfg, "home", None) if hermes_cfg else None
    )

    # Explicit overrides from the PaperFlux config win outright.
    override_base = getattr(hermes_cfg, "base_url", None) if hermes_cfg else None
    override_key = getattr(hermes_cfg, "api_key", None) if hermes_cfg else None
    override_model = getattr(hermes_cfg, "model", None) if hermes_cfg else None
    override_key_env = getattr(hermes_cfg, "key_env", None) if hermes_cfg else None
    wanted_provider = getattr(hermes_cfg, "provider", None) if hermes_cfg else None

    hermes_yaml = home / "config.yaml"
    data: dict = {}
    if hermes_yaml.exists():
        with open(hermes_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

    model_section = data.get("model") or {}
    provider_section: dict = {}
    if wanted_provider:
        providers = data.get("providers") or {}
        if wanted_provider not in providers:
            known = sorted(providers) if isinstance(providers, dict) else []
            hint = f" Known providers: {', '.join(known)}." if known else ""
            raise ValueError(
                f"Hermes provider '{wanted_provider}' not found in "
                f"{hermes_yaml}.{hint}"
            )
        provider_section = providers[wanted_provider] or {}

    base_url = (
        override_base
        or provider_section.get("base_url")
        or model_section.get("base_url")
    )
    model = (
        override_model
        or provider_section.get("model")
        or model_section.get("model")
        or model_section.get("default")
    )
    key_env = (
        override_key_env
        or provider_section.get("key_env")
        or model_section.get("key_env")
    )

    dotenv = _read_dotenv(home / ".env")
    # The credential pool is Hermes' live rotation source and wins over the
    # key named in config.yaml (which may be stale); the config key_env is
    # the fallback. Explicit hermes.api_key always wins.
    pool_env = _pool_key_env(
        home, base_url, wanted_provider, model_section.get("provider")
    )
    api_key = (
        override_key
        or _lookup_key(pool_env, dotenv)
        or _lookup_key(key_env, dotenv)
    )
    if api_key and not override_key:
        key_env = pool_env if _lookup_key(pool_env, dotenv) == api_key else key_env

    missing = []
    if not base_url:
        missing.append("base_url")
    if not model:
        missing.append("model")
    if not api_key:
        missing.append(f"api_key (key_env='{key_env}')" if key_env else "api_key")
    if missing:
        raise ValueError(
            f"Hermes connection incomplete (missing: {', '.join(missing)}). "
            f"Hermes home: {home}. Set them via the PaperFlux 'hermes:' block "
            "or fix the Hermes config + key."
        )
    extra_headers: Dict[str, str] = {}
    if base_url and "opencode.ai" in base_url:
        # OpenCode Go relay requires x-opencode-session on every request
        # (HTTP 400 MissingSessionID otherwise). Per Hermes docs the value is
        # an opaque per-conversation key carrying no personal data; without a
        # live Hermes session an ephemeral key per PaperFlux run is correct.
        extra_headers["x-opencode-session"] = f"paperflux-{uuid.uuid4().hex[:16]}"
    # Hermes talks to opencode.ai relays via the Responses API (its responses
    # transport), not chat/completions — mirror that unless overridden.
    override_mode = getattr(hermes_cfg, "api_mode", None) if hermes_cfg else None
    if override_mode and override_mode not in ("chat_completions", "responses"):
        raise ValueError(
            f"Unknown hermes.api_mode '{override_mode}'. "
            "Use 'chat_completions' or 'responses'."
        )
    api_mode = override_mode or (
        "responses" if (base_url and "opencode.ai" in base_url)
        else "chat_completions"
    )
    return HermesConnection(
        base_url=str(base_url), api_key=str(api_key), model=str(model),
        api_mode=api_mode, extra_headers=extra_headers,
    )


def extract_pdf_text(path: Path) -> str:
    """Extract page-marked plain text from a PDF for in-context prompting."""
    chunks = []
    with fitz.open(path) as doc:
        for i in range(len(doc)):
            text = str(doc[i].get_text("text") or "").strip()
            if text:
                chunks.append(f"[Page {i + 1}]\n{text}")
    return "\n\n".join(chunks)


async def _complete_text(client: AsyncOpenAI, conn: HermesConnection,
                         system: str, user: str, max_tokens: int,
                         json_mode: bool) -> str:
    """Run one extraction/summary call and return the plain text result.

    Uses the Responses API when the Hermes endpoint needs it (opencode.ai
    relays, like Hermes itself) and chat/completions everywhere else.
    """
    headers = conn.extra_headers or None
    if conn.api_mode == "responses":
        kwargs: dict = {
            "model": conn.model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_output_tokens": max_tokens,
            "store": False,
        }
        if json_mode:
            kwargs["text"] = {"format": {"type": "json_object"}}
        if headers:
            kwargs["extra_headers"] = headers
        resp = await client.responses.create(**kwargs)
        status = getattr(resp, "status", None)
        if status and status != "completed":
            raise ValueError(
                f"Hermes response ended with status='{status}'. Consider "
                f"increasing ui.max_output_tokens (currently {max_tokens})."
            )
        return getattr(resp, "output_text", "") or ""
    kwargs = {
        "model": conn.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if headers:
        kwargs["extra_headers"] = headers
    resp = await client.chat.completions.create(**kwargs)
    choice = (getattr(resp, "choices", [None])[0] or None)
    if getattr(choice, "finish_reason", None) == "length":
        raise ValueError(
            "Hermes response was truncated (finish_reason='length'). Consider "
            f"increasing ui.max_output_tokens (currently {max_tokens})."
        )
    try:
        return choice.message.content if choice else ""
    except AttributeError:
        return ""


class HermesProvider:
    """Analyze a PDF with whatever LLM the local Hermes installation uses.

    Pipeline mirrors the Anthropic backend (whole paper in context, one
    bundled extraction call + one summary call) but talks to the Hermes
    OpenAI-compatible endpoint, so no separate LLM key setup is needed.
    """

    async def analyze_pdf(
        self,
        path: Path,
        cfg: Config,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> dict:
        """Extract quotes and a summary from a PDF via the Hermes LLM.

        Args:
            path: Local path to the PDF file to analyze.
            cfg: Resolved application configuration.
            progress_callback: Optional callable receiving status strings.

        Returns:
            A dict with keys ``"key_takeaways"`` (str) and ``"quotes"``
            (Dict[category_name, list]).
        """
        conn = resolve_connection(cfg)
        client = AsyncOpenAI(
            base_url=str(conn.base_url), api_key=str(conn.api_key)
        )

        if progress_callback:
            progress_callback(f"Extracting text from {path.name}")
        paper_text = extract_pdf_text(path)
        if not paper_text.strip():
            raise ValueError(f"No extractable text found in {path}")

        categories = cfg.extraction_categories.categories
        category_template = load_template(
            resolve_config_path(cfg.rag.category_prompt_file, cfg)
        )
        category_system_prompt = load_text_file(
            resolve_config_path(cfg.rag.category_system_prompt_file, cfg)
        )
        category_entries = [
            {"name": cat, "description": desc} for cat, desc in categories.items()
        ]
        user_msg = category_template.render(
            categories=category_entries,
            max_quotes_per_category=cfg.rag.max_quotes_per_category,
        )
        user_msg += "\n\nPAPER TEXT (page markers are 1-indexed):\n" + paper_text

        if progress_callback:
            progress_callback(f"Extracting quotes with Hermes model {conn.model}")
        text_val = await _complete_text(
            client, conn, category_system_prompt, user_msg,
            cfg.ui.max_output_tokens, json_mode=True,
        )
        if not text_val or not text_val.strip():
            dump_path = dump_failed_response(
                "hermes_category_bundle_no_text", text_val
            )
            hint = f" (raw saved to {dump_path})" if dump_path else ""
            raise ValueError(
                f"Hermes category bundle response missing text output{hint}"
            )
        try:
            result = json.loads(text_val)
        except json.JSONDecodeError as exc:
            dump_path = dump_failed_response("hermes_category_bundle_bad_json", text_val)
            hint = f" (raw saved to {dump_path})" if dump_path else ""
            raise ValueError(
                f"Hermes category bundle response was not valid JSON{hint}"
            ) from exc
        if not isinstance(result, dict):
            raise ValueError(
                f"Hermes category bundle returned non-dict JSON: {type(result)}"
            )

        quotes, category_summaries = normalize_category_bundle(result)

        summary_template = load_template(
            resolve_config_path(cfg.rag.summary_prompt_file, cfg)
        )
        summary_msg = summary_template.render(
            detail_level=cfg.ui.detail_level,
            category_summaries=category_summaries,
        )
        if progress_callback:
            progress_callback("Generating summary")
        key_takeaways = await _complete_text(
            client, conn,
            "You are a meticulous research assistant writing a Markdown summary.",
            summary_msg, cfg.ui.max_output_tokens, json_mode=False,
        )

        return {"key_takeaways": key_takeaways, "quotes": quotes}
