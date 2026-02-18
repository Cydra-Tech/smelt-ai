# Changelog

All notable changes to smelt-ai.

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
