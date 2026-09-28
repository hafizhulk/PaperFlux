---
layout: default
title: PaperFlux
description: AI-powered PDF annotation for research papers
---

# PaperFlux

PaperFlux helps you read scientific papers faster by automatically extracting exact quotations, organizing them by category (e.g., contributions, limitations, claims, evidence), and annotating your PDFs with precise highlights. It also produces a concise, structured summary you can share or extend. PaperFlux works with OpenAI, Anthropic (Claude), or any Hermes Agent LLM; you select the backend in `config.yaml` and provide the matching API key (no key needed for Hermes).

## The Idea

- Give the model the paper and let it find the most relevant passages: OpenAI uses server-side file search over a temporary vector store, while Anthropic (Claude) and Hermes read the paper in context (PDF natively for Claude, extracted text for Hermes).
- Ask the model to return structured results (JSON) with exact quotes and page numbers per category.
- Turn that into actionable artifacts: an annotated PDF with color-coded highlights and a clean Markdown summary.
- Keep everything reproducible: save the extracted quotes alongside your outputs so you can re-annotate without re-running extraction.

## Features

- Pluggable LLM backend: OpenAI, Anthropic (Claude), or Hermes Agent, selected via `provider` in config
- Batch CLI: [options] *.pdf
- YAML config with LLMs, prompts, colors, defaults
- Three detail levels (low / medium / high)
- RAG retrieval and summarization pipeline
- User-editable prompt templates (Jinja2)
- RGB highlight colors editable in config
- Optional output directory for generated artifacts
- Stage-level CLI progress with elapsed time during extraction and annotation
- Markdown summary injected as sticky note on page 1
- Color-coded highlights: contributions (Y), limitations (O), claims (B), evidence (G)
- Quote-match report with matched/skipped counts, pages, scores, and verbose layout-gap diagnostics

## Getting Started

### 1. Installation

Install the latest release from PyPI:

```bash
python -m pip install paperflux
```

For local development from a cloned repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Choose a Provider and Configure the API Key

PaperFlux supports three backends, selected by the `provider` key in `config.yaml`:

```yaml
# "openai" (default), "anthropic", or "hermes"
provider: "openai"

openai:
	api_key: "ENV:PAPERFLUX_OPENAI_API_KEY"
	model: "gpt-5.4-mini"

# Used when provider is "anthropic"
anthropic:
	api_key: "ENV:PAPERFLUX_ANTHROPIC_API_KEY"
	model: "claude-opus-4-8"

# Used when provider is "hermes": reuse any Hermes Agent LLM, no key setup.
# hermes: {}  # all overrides optional; auto-detects $HERMES_HOME / ~/.hermes
```

Only the selected provider's block is required; PaperFlux validates that the chosen `provider` has a matching configuration block (`hermes` needs no block — it auto-detects the Hermes installation on the machine). The `ENV:` prefix means PaperFlux expands the named environment variable at runtime. A PaperFlux-specific key makes it easier to monitor package-related usage and cost separately.

### 2a. Hermes backend (no API key setup)

If [Hermes Agent](https://github.com/NousResearch/hermes-agent) is installed on the machine, set `provider: "hermes"` and skip the key setup entirely. PaperFlux reads the Hermes home (`hermes.home` in config, else `$HERMES_HOME`, else `~/.hermes` — profiles included), takes `base_url`/`model`/`key_env` from its `config.yaml`, and the key from the environment or its `.env`. This works with any Hermes setup — OpenRouter, Anthropic, OpenAI, local models, or whatever the machine is configured to use — because Hermes endpoints are OpenAI-compatible. The PDF text is extracted locally and sent in context, so unlike the OpenAI backend no vector store or `file_search` is involved. Optional overrides (`hermes.base_url/model/api_key/key_env/provider/home/api_mode`) always win over auto-detection. The credential pool (`auth.json`) is consulted before the configured `key_env`, mirroring Hermes' own key rotation; opencode.ai relays get an ephemeral `x-opencode-session` header and the Responses API, exactly like Hermes core.

Choose one of the following setups (substitute `PAPERFLUX_ANTHROPIC_API_KEY` and an `sk-ant-` key when using Anthropic):

- Temporary (current shell only):

	```bash
	export PAPERFLUX_OPENAI_API_KEY="sk-your-key"
	```

- Persistent for zsh (macOS default):

	```bash
	echo 'export PAPERFLUX_OPENAI_API_KEY="sk-your-key"' >> ~/.zshrc
	source ~/.zshrc
	```

- Using a `.env` file (auto-loaded):

	Create a file named `.env` in the repository root:

	```bash
	echo 'PAPERFLUX_OPENAI_API_KEY=sk-your-key' > .env
	```

	PaperFlux loads `.env` automatically because it calls `load_dotenv()` during config load.

- Inline in `config.yaml` (not recommended for commits):

	```yaml
	openai:
		api_key: "sk-your-key"
	```

### 3. Usage

Run the CLI:

```bash
paperflux init
paperflux --config config.yaml path/to/your.pdf
# or reuse prior quotes without extraction
paperflux --config config.yaml --quotes-file your_paper_quotes.json path/to/your.pdf
```

#### Options

- `--config`, `-c`: Path to configuration file (required)
- `--detail`, `-d`: Detail level (low/medium/high, overrides config)
- `--verbose`: Enable internal logs, per-quote match details, and layout-gap diagnostics
- `--output-dir`, `-o`: Directory where annotated PDFs and summaries will be saved
- `--progress/--no-progress`: Show or hide stage-level progress updates with elapsed time (default: shown)
- `--quotes-file`: Path to JSON quotes file to annotate without rerunning extraction
- `--version`, `-V`: Show the PaperFlux version and exit

Use `paperflux init [directory]` to create a starter `config.yaml` and editable prompt templates in `prompts/`. Existing files are not overwritten unless you pass `--force`.

## Customizing Prompts

PaperFlux uses three editable Jinja2 templates in the `prompts/` directory to control how the AI extracts and summarizes information:

### `rag_category_system_prompt.txt` and `rag_category_system_prompt_anthropic.txt`

The system prompt that instructs the AI assistant on its role and behavior. PaperFlux uses a provider-specific variant: `rag_category_system_prompt.txt` for OpenAI (which retrieves passages via file_search) and `rag_category_system_prompt_anthropic.txt` for Anthropic (which reads the attached PDF directly). Both define the same extraction rules:
- Find the relevant passages (file_search for OpenAI; the attached PDF for Anthropic)
- Extract near-verbatim quotations with accurate page numbers
- Avoid including section/table/figure references in quotes
- Return structured JSON without code fences

This is the "personality" of the extraction assistant. The file used for each provider is configurable via `rag.category_system_prompt_file` and `rag.category_system_prompt_file_anthropic`.

### `rag_category_prompt.j2`

The user prompt template for category extraction. It:
- Lists all extraction categories (contributions, limitations, claims, evidence) with their descriptions
- Specifies the JSON output format with categories, quotes, pages, and category summaries
- Gets rendered with variables from `config.yaml` (categories list)

Edit `config.yaml` (under `extraction_categories`) to change what categories are extracted. Edit this template to modify the output structure or instructions.

### `rag_summary_prompt.j2`

The prompt template for generating the final Markdown summary. It:
- Receives all category summaries as input
- Uses the detail level (low/medium/high) to control summary length
- Produces a cohesive narrative summary integrating all categories

Edit this to change how the final summary is structured or what information it emphasizes.

All three templates can be customized without modifying any Python code—just edit the files in `prompts/` and rerun PaperFlux.

## Model and Retrieval Settings

### OpenAI

The default OpenAI model is `gpt-5.4-mini`, which supports the Responses API, structured outputs, and file search. You can change it in `config.yaml`:

```yaml
openai:
  model: "gpt-5.4-mini"
```

File-search retrieval can also be tuned without changing Python code:

```yaml
ui:
  reasoning_effort: "medium"
  max_output_tokens: 32768

matching:
  min_similarity: 0.88
  max_window_tokens: 80

rag:
  max_num_results:        # leave empty to let OpenAI choose
  max_quotes_per_category: 6
  include_search_results: false
  vector_store_expires_after_days: 1
```

Use `max_num_results` to cap retrieved passages when latency or cost matters. Use `max_quotes_per_category` to keep the structured JSON response bounded. The local highlighter uses `matching.min_similarity` and `matching.max_window_tokens` to align returned quotes to real PDF word spans; raising the similarity threshold improves precision, while lowering it improves recall. Enabling `include_search_results` is useful for debugging retrieval, but it increases the response payload.

### Anthropic (Claude)

When `provider: "anthropic"`, set the model under the `anthropic` block:

```yaml
provider: "anthropic"

anthropic:
  model: "claude-opus-4-8"
```

Claude reads the PDF directly (sent as a document in the request), so there is no vector store and the `rag.max_num_results`, `rag.include_search_results`, and `rag.vector_store_expires_after_days` settings do not apply. The shared `ui` and `rag` knobs still apply:

- `ui.reasoning_effort` maps to Claude's thinking: `"none"` disables thinking, and `low`/`medium`/`high`/`xhigh` enable adaptive thinking at the corresponding effort.
- `ui.max_output_tokens` caps the response (requests stream, so large values are supported).
- `rag.max_quotes_per_category` bounds the structured JSON response, and the same `matching.*` settings align quotes to PDF word spans.

Page numbers are returned as a field in the structured JSON (citations and structured output cannot be combined in the Messages API), keeping the output identical to the OpenAI path.

### Quote matching

The highlighter first tries exact and fuzzy contiguous matching. If those fail, it can fall back to `layout-gap` matching for quotes split by page layout interruptions such as tables, figures, captions, or column breaks. Layout-gap matches highlight only the words that belong to the quote and skip intervening layout artifacts.

## Brief History

- v1.0.20250525: Initial version using local text extraction, LLM-based processing of extracted text, and fuzzy quote matching for PDF annotation.
- v2.0.20250625: Migrated to OpenAI Responses API, adopted file_search with vector stores.
- v2.1.20251115: Added GPT-5 support, and bundled multi-category extraction into a single structured call.
- v2.2.20251115: Added n-gram quote matching for improved PDF annotation.
- v3.0.20251116: Added token-based quote matching for improved PDF annotation. Removed fuzzy- and n-gram-based matching.
- v3.1.20251116: Introduced a `--quotes-file` flow to re-annotate quickly without re-running extraction.
- v3.2.20260527: Updated default model to GPT-5.4 mini and added file-search tuning options.
- v3.3.20260527: Added local quote span alignment for more accurate PDF highlights.
- v3.4.20260528: Added layout-gap quote matching for quotes split by tables, figures, captions, or column breaks; added quote-match reports, stage-level CLI progress, concise default CLI output with verbose diagnostics, package metadata, CLI validation improvements, and the `PAPERFLUX_OPENAI_API_KEY` default for easier cost tracking.
- v4.0.20260530: Added Anthropic (Claude) as a selectable LLM backend via a `provider` config key, alongside a pluggable provider registry. The Anthropic path reads the PDF directly (no vector store), returns the same structured JSON with page numbers, and maps `reasoning_effort` to adaptive thinking.
- v4.1.20260613: Added a `--version`/`-V` CLI flag that reports the installed package version.

## Contributing

Contributions are very welcome! Whether you’re fixing a bug, improving extraction accuracy, or adding a new workflow, here’s how to get started:

1. Fork the repository and create a feature branch.
2. Set up the project and run tests locally.
3. Open a PR describing the change and its impact.

Issues, ideas, or questions? Please open an issue or start a discussion—feedback helps shape the roadmap.

---

For more information about configuration options and advanced features, see `config.yaml` in the repository.
