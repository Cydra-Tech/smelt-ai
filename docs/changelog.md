# Changelog

All notable changes to smelt-ai.

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
