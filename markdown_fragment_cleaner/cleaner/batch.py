from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence

from .config import CleanerConfig, OutputConfig
from .pipeline import CleaningSummary, clean_jsonl


@dataclass(frozen=True)
class BatchConfig:
    """Shard-level orchestration settings for a directory of JSONL files."""

    input_dir: str
    output_dir: str
    input_glob: str = "**/*.jsonl"
    max_workers: int = 4
    skip_completed_shards: bool = True
    continue_on_error: bool = True
    write_shard_previews: bool = False

    def validate(self) -> None:
        if not self.input_dir:
            raise ValueError("input_dir is required")
        if not self.output_dir:
            raise ValueError("output_dir is required")
        if not self.input_glob:
            raise ValueError("input_glob is required")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")


@dataclass
class ShardResult:
    relative_path: str
    input_path: str
    status: str
    elapsed_seconds: float = 0.0
    summary: Optional[CleaningSummary] = None
    error_type: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "input_path": self.input_path,
            "status": self.status,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "summary": self.summary.to_dict() if self.summary is not None else None,
            "error_type": self.error_type,
            "error": self.error,
        }


@dataclass
class BatchSummary:
    discovered_shards: int = 0
    scheduled_shards: int = 0
    skipped_shards: int = 0
    completed_shards: int = 0
    failed_shards: int = 0
    elapsed_seconds: float = 0.0
    aggregate: CleaningSummary = field(default_factory=CleaningSummary)
    failures: List[Dict[str, str]] = field(default_factory=list)

    def add(self, result: ShardResult) -> None:
        if result.status == "skipped":
            self.skipped_shards += 1
            if result.summary is not None:
                _merge_cleaning_summaries(self.aggregate, result.summary)
        elif result.status == "completed":
            self.completed_shards += 1
            if result.summary is not None:
                _merge_cleaning_summaries(self.aggregate, result.summary)
        elif result.status == "failed":
            self.failed_shards += 1
            self.failures.append(
                {
                    "relative_path": result.relative_path,
                    "input_path": result.input_path,
                    "error_type": result.error_type,
                    "error": result.error,
                }
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "discovered_shards": self.discovered_shards,
            "scheduled_shards": self.scheduled_shards,
            "skipped_shards": self.skipped_shards,
            "completed_shards": self.completed_shards,
            "failed_shards": self.failed_shards,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "aggregate": self.aggregate.to_dict(),
            "failures": self.failures,
        }


@dataclass(frozen=True)
class _ShardJob:
    relative_path: str
    input_path: str
    marker_path: str
    expected_outputs: Sequence[str]
    input_size: int
    input_mtime_ns: int
    config_fingerprint: str
    config: CleanerConfig


ProgressCallback = Callable[[ShardResult, BatchSummary], None]


def clean_jsonl_shards(
    base_config: CleanerConfig,
    batch_config: BatchConfig,
    on_result: Optional[ProgressCallback] = None,
) -> BatchSummary:
    """Clean JSONL shards concurrently while preserving their relative paths."""

    batch_config.validate()
    input_root = Path(batch_config.input_dir).expanduser().resolve()
    output_root = Path(batch_config.output_dir).expanduser().resolve()
    if not input_root.is_dir():
        raise NotADirectoryError("input directory does not exist: %s" % input_root)

    input_paths = _discover_inputs(input_root, output_root, batch_config.input_glob)
    if not input_paths:
        raise FileNotFoundError(
            "no JSONL shards matched %s under %s" % (batch_config.input_glob, input_root)
        )

    started = time.monotonic()
    summary = BatchSummary(discovered_shards=len(input_paths))
    jobs: List[_ShardJob] = []
    for input_path in input_paths:
        job = _build_job(base_config, batch_config, input_root, output_root, input_path)
        completed_summary = (
            _read_completed_summary(job) if batch_config.skip_completed_shards else None
        )
        if completed_summary is not None:
            result = ShardResult(
                relative_path=job.relative_path,
                input_path=job.input_path,
                status="skipped",
                summary=completed_summary,
            )
            summary.add(result)
            if on_result is not None:
                on_result(result, summary)
        else:
            jobs.append(job)

    summary.scheduled_shards = len(jobs)
    failure: Optional[ShardResult] = None
    if batch_config.max_workers == 1:
        for job in jobs:
            result = _process_shard(job)
            summary.add(result)
            if on_result is not None:
                on_result(result, summary)
            if result.status == "failed" and not batch_config.continue_on_error:
                failure = result
                break
    else:
        failure = _run_parallel(jobs, batch_config, summary, on_result)

    summary.elapsed_seconds = time.monotonic() - started
    _write_batch_summary(output_root, batch_config, base_config, summary)
    if failure is not None:
        raise RuntimeError(
            "shard %s failed: %s: %s"
            % (failure.relative_path, failure.error_type, failure.error)
        )
    return summary


def _run_parallel(
    jobs: Sequence[_ShardJob],
    batch_config: BatchConfig,
    summary: BatchSummary,
    on_result: Optional[ProgressCallback],
) -> Optional[ShardResult]:
    if not jobs:
        return None

    job_iterator = iter(jobs)
    pending: Dict[Future[ShardResult], _ShardJob] = {}
    max_pending = max(batch_config.max_workers, batch_config.max_workers * 2)
    failure: Optional[ShardResult] = None

    with ProcessPoolExecutor(max_workers=batch_config.max_workers) as executor:
        _fill_pending(executor, pending, job_iterator, max_pending)
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                job = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = ShardResult(
                        relative_path=job.relative_path,
                        input_path=job.input_path,
                        status="failed",
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                summary.add(result)
                if on_result is not None:
                    on_result(result, summary)
                if result.status == "failed" and not batch_config.continue_on_error:
                    failure = result
                    for remaining in pending:
                        remaining.cancel()
                    pending.clear()
                    break
            if failure is not None:
                break
            _fill_pending(executor, pending, job_iterator, max_pending)
    return failure


def _fill_pending(
    executor: ProcessPoolExecutor,
    pending: Dict[Future[ShardResult], _ShardJob],
    jobs: Iterator[_ShardJob],
    limit: int,
) -> None:
    while len(pending) < limit:
        try:
            job = next(jobs)
        except StopIteration:
            return
        pending[executor.submit(_process_shard, job)] = job


def _process_shard(job: _ShardJob) -> ShardResult:
    started = time.monotonic()
    try:
        summary = clean_jsonl(job.config)
        _write_completion_marker(job, summary)
        return ShardResult(
            relative_path=job.relative_path,
            input_path=job.input_path,
            status="completed",
            elapsed_seconds=time.monotonic() - started,
            summary=summary,
        )
    except Exception as exc:
        return ShardResult(
            relative_path=job.relative_path,
            input_path=job.input_path,
            status="failed",
            elapsed_seconds=time.monotonic() - started,
            error_type=type(exc).__name__,
            error=str(exc),
        )


def _discover_inputs(input_root: Path, output_root: Path, pattern: str) -> List[Path]:
    inputs = []
    for path in input_root.glob(pattern):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if _is_relative_to(resolved, output_root):
            continue
        inputs.append(resolved)
    return sorted(set(inputs), key=lambda path: path.as_posix())


def _build_job(
    base_config: CleanerConfig,
    batch_config: BatchConfig,
    input_root: Path,
    output_root: Path,
    input_path: Path,
) -> _ShardJob:
    relative = input_path.relative_to(input_root)
    relative_without_suffix = relative.with_suffix("")
    fallback_prefix = relative_without_suffix.as_posix()
    if base_config.input.fallback_id_prefix:
        fallback_prefix = "%s:%s" % (
            base_config.input.fallback_id_prefix,
            fallback_prefix,
        )

    output = OutputConfig(
        accepted_path=str(output_root / "fragments" / "accepted" / relative),
        review_path=str(output_root / "fragments" / "review" / relative),
        rejected_path=str(output_root / "fragments" / "rejected" / relative),
        templates_path=str(output_root / "metadata" / "templates" / relative.with_suffix(".json")),
        statistics_path=str(output_root / "metadata" / "statistics" / relative.with_suffix(".json")),
        preview_path=str(output_root / "metadata" / "previews" / "fragments" / relative.with_suffix(".md")),
        preview_fragments=base_config.output.preview_fragments,
        document_accepted_path=str(output_root / "documents" / "accepted" / relative),
        document_review_path=str(output_root / "documents" / "review" / relative),
        document_rejected_path=str(output_root / "documents" / "rejected" / relative),
        document_preview_path=str(output_root / "metadata" / "previews" / "documents" / relative.with_suffix(".md")),
        preview_documents=base_config.output.preview_documents,
        write_fragment_preview=batch_config.write_shard_previews,
        write_document_preview=batch_config.write_shard_previews,
    )
    shard_config = replace(
        base_config,
        input_path=str(input_path),
        input=replace(base_config.input, fallback_id_prefix=fallback_prefix),
        output=output,
    )
    fingerprint = _config_fingerprint(shard_config)
    stat = input_path.stat()
    marker = output_root / "metadata" / "completed" / relative.with_suffix(".done.json")
    return _ShardJob(
        relative_path=relative.as_posix(),
        input_path=str(input_path),
        marker_path=str(marker),
        expected_outputs=_expected_outputs(shard_config),
        input_size=stat.st_size,
        input_mtime_ns=stat.st_mtime_ns,
        config_fingerprint=fingerprint,
        config=shard_config,
    )


def _expected_outputs(config: CleanerConfig) -> List[str]:
    outputs = [config.output.statistics_path]
    if config.templates.enabled:
        outputs.append(config.output.templates_path)
    if config.assembly.output_mode in {"fragment", "both"}:
        outputs.extend(
            [
                config.output.accepted_path,
                config.output.review_path,
                config.output.rejected_path,
            ]
        )
        if config.output.write_fragment_preview:
            outputs.append(config.output.preview_path)
    if config.assembly.output_mode in {"document", "both"}:
        outputs.extend(
            [
                config.output.document_accepted_path,
                config.output.document_review_path,
                config.output.document_rejected_path,
            ]
        )
        if config.output.write_document_preview:
            outputs.append(config.output.document_preview_path)
    return outputs


def _config_fingerprint(config: CleanerConfig) -> str:
    encoded = json.dumps(
        asdict(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def _read_completed_summary(job: _ShardJob) -> Optional[CleaningSummary]:
    marker = Path(job.marker_path)
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    matches = (
        value.get("input_size") == job.input_size
        and value.get("input_mtime_ns") == job.input_mtime_ns
        and value.get("config_fingerprint") == job.config_fingerprint
        and all(Path(path).is_file() for path in job.expected_outputs)
    )
    stored_summary = value.get("summary")
    if not matches or not isinstance(stored_summary, Mapping):
        return None
    return _summary_from_dict(stored_summary)


def _write_completion_marker(job: _ShardJob, summary: CleaningSummary) -> None:
    payload = {
        "version": 1,
        "relative_path": job.relative_path,
        "input_path": job.input_path,
        "input_size": job.input_size,
        "input_mtime_ns": job.input_mtime_ns,
        "config_fingerprint": job.config_fingerprint,
        "outputs": list(job.expected_outputs),
        "summary": summary.to_dict(),
    }
    _atomic_write_json(Path(job.marker_path), payload)


def _write_batch_summary(
    output_root: Path,
    batch_config: BatchConfig,
    base_config: CleanerConfig,
    summary: BatchSummary,
) -> None:
    payload = {
        "version": 1,
        "batch_config": asdict(batch_config),
        "cleaner_config": asdict(base_config),
        "summary": summary.to_dict(),
    }
    _atomic_write_json(output_root / "metadata" / "batch_summary.json", payload)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _merge_cleaning_summaries(target: CleaningSummary, source: CleaningSummary) -> None:
    for item in fields(CleaningSummary):
        current = getattr(target, item.name)
        incoming = getattr(source, item.name)
        if isinstance(current, Counter):
            current.update(incoming)
        else:
            setattr(target, item.name, current + incoming)


def _summary_from_dict(value: Mapping[str, Any]) -> CleaningSummary:
    summary = CleaningSummary()
    for item in fields(CleaningSummary):
        if item.name not in value:
            continue
        current = getattr(summary, item.name)
        incoming = value[item.name]
        if isinstance(current, Counter):
            if isinstance(incoming, Mapping):
                current.update(incoming)
        elif isinstance(incoming, int):
            setattr(summary, item.name, incoming)
    return summary


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
