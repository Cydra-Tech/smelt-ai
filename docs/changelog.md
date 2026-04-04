# Changelog

All notable changes to smelt-ai.

## v0.4.0 — 2026-04-04

### Added
- **`AggregateJob`** — many-to-one aggregation via tree-parallel reduction
- Map phase runs batches concurrently, merge phase combines outputs pairwise until one result remains
- Supports both structured output (`output_model=SummaryModel`) and free-text mode (`output_model=None`)
- New `execute_aggregate()` engine in `smelt.aggregate` module
- Aggregate-specific prompt templates for map and merge phases
- `build_aggregate_system_message()` and `build_aggregate_human_message()` prompt builders
- 60-company portfolio and 50-employee survey test datasets
- Live tests verifying aggregation accuracy against ground truth

## v0.3.0 — 2026-04-03

### Added
- **Free-text output mode** — set `output_model=None` (or omit it) to get plain `list[str]` responses instead of structured Pydantic models
- `_TextRow` internal model for free-text row tracking
- `create_text_batch_wrapper()` in validation module
- `text_mode` parameter on `build_system_message()` for text-optimized prompts
- `output_model` now defaults to `None` — zero-config free-text is the simplest path

### Changed
- `Job.output_model` is now `Optional[Type[BaseModel]]` with default `None` (backward compatible — existing code with explicit `output_model=MyModel` works unchanged)
- `execute_batches()` branches on `output_model is None` for text vs structured paths
- `strip_row_id()` returns `str` when `user_model is None`

## v0.2.0 — 2026-03-07

### Added
- **Vision / image support** — pass `PIL.Image.Image` objects directly in data dicts for vision-capable LLMs
- New `smelt.image` module with image detection, base64 encoding, and extraction utilities
- `PILLOW_AVAILABLE` flag exported from `smelt` package
- `[vision]` optional dependency extra (`pip install smelt-ai[vision]`)
- `has_images` parameter on `build_system_message()` for image-aware prompts
- Multimodal content blocks in `build_human_message()` — auto-detects PIL images, replaces with placeholders, and appends base64-encoded image blocks
- `UserWarning` when `batch_size > 5` with image data (large payloads)
- Vision cookbook with ECG analysis example

### Changed
- `build_system_message()` now accepts optional `has_images` keyword argument (backward compatible)
- `build_human_message()` returns multimodal content blocks when images are detected, plain text otherwise (backward compatible)

## v0.1.1 — 2025-02-18

### Added
- `job.test()` and `job.atest()` methods for single-row validation before full runs
- PyPI metadata: readme, authors, keywords, classifiers, project URLs
- Documentation URL in PyPI package metadata

### Fixed
- Empty `data` list now raises `SmeltConfigError` in `test()`/`atest()` instead of silently succeeding

## v0.1.0 — 2025-02-18

### Added
- Initial release
- `Model` class wrapping LangChain's `init_chat_model` with lazy initialization and caching
- `Job` class with `run()`, `arun()` for sync/async execution
- Automatic batching with configurable `batch_size`
- Concurrent batch processing via `asyncio.Semaphore` with configurable `concurrency`
- Row ID tracking for guaranteed output ordering
- Automatic retry with exponential backoff and jitter
- `SmeltResult[T]` with typed data, errors, and metrics
- `SmeltMetrics` with token counts, timing, retry counts
- `BatchError` with per-batch diagnostic information
- `stop_on_exhaustion` parameter for fail-fast vs collect-errors modes
- `shuffle` parameter for randomizing batch composition
- Exception hierarchy: `SmeltError`, `SmeltConfigError`, `SmeltValidationError`, `SmeltAPIError`, `SmeltExhaustionError`
- Support for OpenAI, Anthropic, and Google Gemini providers
- Dynamic Pydantic model creation (inherits user validators)
- Pydantic v2 native validation
