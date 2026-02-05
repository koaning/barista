# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "diskcache>=5.6.0",
#     "marimo>=0.19.7",
#     "pytest==9.0.2",
#     "tenacity==9.1.3",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="columns")

with app.setup:
    import asyncio
    import concurrent.futures
    import csv
    import hashlib
    import inspect
    import json
    from dataclasses import dataclass, field
    from pathlib import Path
    from typing import Any, Awaitable, Callable, Generic, Iterator, Literal, TypeVar

    from diskcache import Cache
    from tenacity import (
        AsyncRetrying,
        stop_after_attempt,
        wait_exponential,
    )

    T = TypeVar("T")


@app.cell(column=0)
def _():
    import pytest
    return


@app.cell
def _(pipeline):
    # Example 1: sync functions
    def test_basic_example():
        def _add_greeting(item: dict) -> dict:
            return {**item, "greeting": f"Hello, {item['name']}!"}

        _data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]

        _results = (
            pipeline(_data, show_progress=False)
            .then(_add_greeting)
            .collect()
        )
        assert len(list(_results)) == 3
    return


@app.cell
def _(pipeline):
    # Example 2: async functions (run concurrently)
    async def _slow_transform(item: dict) -> dict:
        await asyncio.sleep(0.1)
        return {**item, "processed": True}

    _data = [{"id": i} for i in range(10)]

    _results = (
        pipeline(_data, max_concurrency=5, show_progress=False)
        .then(_slow_transform)
        .head(3)
    )
    _results
    return


@app.cell
def _(pipeline):
    # Example 3: retry on failure
    _call_count = {"n": 0}

    def _flaky_func(item: dict) -> dict:
        _call_count["n"] += 1
        if _call_count["n"] < 3:
            raise ValueError("Simulated failure")
        return {**item, "success": True}

    _data = [{"id": 1}]
    _results = (
        pipeline(_data, show_progress=False)
        .then(_flaky_func, retries=3)
        .collect()
    )
    (list(_results), f"Total calls: {_call_count['n']}")
    return


@app.cell
def _(pipeline):
    # Example 4: error handling (None for failed items)
    def _failing_func(item: dict) -> dict:
        if item["id"] == 2:
            raise ValueError("ID 2 always fails")
        return {**item, "ok": True}

    _data = [{"id": 1}, {"id": 2}, {"id": 3}]
    _results = (
        pipeline(_data, show_progress=False)
        .then(_failing_func, on_error="none")  # None for failed, keep in results
        .collect()
    )
    _results  # [{'id': 1, 'ok': True}, None, {'id': 3, 'ok': True}]
    return


@app.class_definition
@dataclass
class PipelineStep:
    """Represents a single step in the pipeline."""

    func: Callable[..., Any]
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    cache_key: str | None = None
    is_async: bool = False
    max_concurrency: int = 10
    retries: int = 0
    on_error: Literal["none", "skip"] = "none"


@app.class_definition
@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""

    cache_dir: str = ".pipeline-cache"
    max_concurrency: int = 10
    rate_limit: float | None = None
    show_progress: bool = True


@app.cell
def _():
    class Pipeline(Generic[T]):
        """
        A lazy, chainable pipeline for processing JSONL data.

        Usage:
            pipeline("data.jsonl").then(transform).then(enrich, cache=True).head(10)
        """

        def __init__(
            self,
            source: str | Path | list[T],
            config: PipelineConfig | None = None,
        ):
            self._source = source
            self._steps: list[PipelineStep] = []
            self._config = config or PipelineConfig()
            self._cache: Cache | None = None

        def then(
            self,
            func: Callable[..., T | Awaitable[T]],
            *args: Any,
            cache: bool | str = False,
            retries: int = 0,
            on_error: Literal["none", "skip"] = "none",
            **kwargs: Any,
        ) -> "Pipeline[T]":
            """
            Add a transformation step to the pipeline (lazy).

            Args:
                func: Function that takes a dict and returns a dict (sync or async)
                *args: Additional positional arguments for func
                cache: True for auto-caching, or string for custom cache key
                retries: Number of retry attempts on failure (uses exponential backoff)
                on_error: "none" returns None for failed items, "skip" excludes them
                **kwargs: Additional keyword arguments for func

            Returns:
                New Pipeline with the step added
            """
            is_async = inspect.iscoroutinefunction(func)

            cache_key = None
            if cache:
                if isinstance(cache, str):
                    cache_key = cache
                else:
                    cache_key = self._generate_cache_key(func, args, kwargs)

            step = PipelineStep(
                func=func,
                args=args,
                kwargs=kwargs,
                cache_key=cache_key,
                is_async=is_async,
                max_concurrency=self._config.max_concurrency,
                retries=retries,
                on_error=on_error,
            )

            new_pipeline: Pipeline[T] = Pipeline(self._source, self._config)
            new_pipeline._steps = self._steps + [step]
            new_pipeline._cache = self._cache
            return new_pipeline

        def _iter_source(self) -> Iterator[dict]:
            """Lazily iterate over the source (JSONL, CSV, or list)."""
            if isinstance(self._source, (str, Path)):
                path = Path(self._source)
                suffix = path.suffix.lower()

                if suffix == ".csv":
                    with open(path, "r", newline="") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            yield dict(row)
                elif suffix in (".jsonl", ".ndjson"):
                    with open(path, "r") as f:
                        for line in f:
                            if line.strip():
                                yield json.loads(line)
                else:
                    # Default: try JSONL
                    with open(path, "r") as f:
                        for line in f:
                            if line.strip():
                                yield json.loads(line)
            else:
                yield from self._source

        def _get_cache(self) -> Cache:
            """Lazily initialize cache."""
            if self._cache is None:
                self._cache = Cache(self._config.cache_dir)
            return self._cache

        def _generate_cache_key(
            self, func: Callable, args: tuple, kwargs: dict
        ) -> str:
            """Generate a cache key based on function identity."""
            func_id = f"{func.__module__}.{func.__qualname__}"
            args_hash = hashlib.md5(
                json.dumps((args, kwargs), sort_keys=True, default=str).encode()
            ).hexdigest()[:8]
            return f"{func_id}:{args_hash}"

        def _cache_item_key(self, step_key: str, item: dict) -> str:
            """Generate cache key for a specific item."""
            item_hash = hashlib.md5(
                json.dumps(item, sort_keys=True, default=str).encode()
            ).hexdigest()
            return f"{step_key}:{item_hash}"

        async def _apply_step(self, step: PipelineStep, item: dict) -> dict:
            """Apply a single step to an item, with optional retries."""
            if step.retries > 0:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(step.retries + 1),
                    wait=wait_exponential(multiplier=1, min=1, max=10),
                    reraise=True,
                ):
                    with attempt:
                        if step.is_async:
                            return await step.func(item, *step.args, **step.kwargs)
                        else:
                            return step.func(item, *step.args, **step.kwargs)
            else:
                if step.is_async:
                    return await step.func(item, *step.args, **step.kwargs)
                else:
                    return step.func(item, *step.args, **step.kwargs)

        async def _apply_step_with_cache(
            self, step: PipelineStep, item: dict
        ) -> dict:
            """Apply a step with optional caching."""
            if step.cache_key:
                cache = self._get_cache()
                cache_key = self._cache_item_key(step.cache_key, item)

                if cache_key in cache:
                    return cache[cache_key]

                result = await self._apply_step(step, item)
                cache[cache_key] = result
                return result

            return await self._apply_step(step, item)

        async def _process_item(self, item: dict) -> tuple[dict | None, bool]:
            """
            Process a single item through all steps.

            Returns:
                (result, should_include) - result is None on failure,
                should_include is False if on_error="skip"
            """
            current = item
            for step in self._steps:
                try:
                    current = await self._apply_step_with_cache(step, current)
                except Exception:
                    if step.on_error == "skip":
                        return None, False
                    else:  # on_error == "none"
                        return None, True
            return current, True

        async def _execute(self, limit: int | None = None) -> list[T | None]:
            """Execute the pipeline with concurrency control."""
            results: list[T | None] = []
            has_async = any(step.is_async for step in self._steps)

            if has_async:
                semaphore = asyncio.Semaphore(self._config.max_concurrency)

                async def process_with_semaphore(
                    item: dict,
                ) -> tuple[dict | None, bool]:
                    async with semaphore:
                        if self._config.rate_limit:
                            await asyncio.sleep(self._config.rate_limit)
                        return await self._process_item(item)

                batch = []
                count = 0
                batch_size = self._config.max_concurrency * 2

                for item in self._iter_source():
                    if limit and count >= limit:
                        break

                    batch.append(item)

                    if len(batch) >= batch_size:
                        batch_results = await asyncio.gather(
                            *[process_with_semaphore(i) for i in batch]
                        )
                        for result, should_include in batch_results:
                            if should_include:
                                results.append(result)
                                count += 1
                        if self._config.show_progress:
                            print(f"Processed {count} items...", end="\r")
                        batch = []

                if batch:
                    remaining = limit - count if limit else len(batch)
                    batch = batch[:remaining]
                    batch_results = await asyncio.gather(
                        *[process_with_semaphore(i) for i in batch]
                    )
                    for result, should_include in batch_results:
                        if should_include:
                            results.append(result)
                            count += 1
                    if self._config.show_progress:
                        print(f"Processed {count} items.    ")
            else:
                count = 0
                for item in self._iter_source():
                    if limit and count >= limit:
                        break
                    result, should_include = await self._process_item(item)
                    if should_include:
                        results.append(result)
                        count += 1
                    if self._config.show_progress and count % 10 == 0:
                        print(f"Processed {count} items...", end="\r")
                if self._config.show_progress:
                    print(f"Processed {count} items.    ")

            return results[:limit] if limit else results

        def _run_sync(self, coro):
            """Run coroutine from sync context, handling existing event loops."""
            try:
                asyncio.get_running_loop()
                # Already in an event loop (e.g., marimo) - run in thread
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result()
            except RuntimeError:
                # No running loop - safe to use asyncio.run
                return asyncio.run(coro)

        # Terminal operations - work from scripts and notebooks
        def head(self, n: int = 10) -> list[T | None]:
            """Execute pipeline and return first n results."""
            return self._run_sync(self._execute(limit=n))

        def collect(self) -> list[T | None]:
            """Execute pipeline and collect all results."""
            return self._run_sync(self._execute(limit=None))

        def first(self) -> T | None:
            """Execute pipeline and return first result, or None."""
            result = self.head(1)
            return result[0] if result else None

        def run(self) -> int:
            """Execute pipeline for side effects, return count."""
            return len(self.collect())
    return (Pipeline,)


@app.cell
def _(Pipeline):
    def pipeline(
        source: str | Path | list[dict],
        *,
        cache_dir: str = ".pipeline-cache",
        max_concurrency: int = 10,
        rate_limit: float | None = None,
        show_progress: bool = True,
    ) -> Pipeline:
        """
        Create a new lazy pipeline from a JSONL file or list of dicts.

        Args:
            source: Path to JSONL file, or list of dicts
            cache_dir: Directory for caching results
            max_concurrency: Max concurrent async operations
            rate_limit: Seconds between operations (for API throttling)
            show_progress: Show progress during execution

        Example:
            pipeline("data.jsonl").then(transform).then(enrich, cache=True).head(10)
        """
        config = PipelineConfig(
            cache_dir=cache_dir,
            max_concurrency=max_concurrency,
            rate_limit=rate_limit,
            show_progress=show_progress,
        )
        return Pipeline(source, config)
    return (pipeline,)


if __name__ == "__main__":
    app.run()
