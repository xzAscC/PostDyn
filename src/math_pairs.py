from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


CONCEPT_NAME = "concise_math_reasoning_vs_verbose_math_reasoning"

GenerationMode = Literal["verbose", "concise"]
ProblemRecord = Mapping[str, object]
GenerateFn = Callable[[ProblemRecord, GenerationMode, str | None], str]
VerifyFn = Callable[[str, str], bool]
MathId = int | str


@dataclass(frozen=True, slots=True)
class MathPair:
    unique_id: MathId
    problem: str
    gold_answer: str
    verbose_solution: str
    concise_solution: str


PairCallback = Callable[[MathPair], None]


def is_meaningfully_concise(concise: str, verbose: str) -> bool:
    concise_length = len(concise.strip())
    verbose_length = len(verbose.strip())
    return bool(
        concise_length
        and verbose_length
        and concise_length <= int(verbose_length * 0.9)
    )


def sort_math_pairs(pairs: Iterable[MathPair]) -> list[MathPair]:
    return sorted(pairs, key=lambda pair: _id_sort_key(pair.unique_id))


def _required(record: ProblemRecord, key: str) -> object:
    if key not in record:
        raise ValueError(f"MATH-500 record is missing required field {key!r}")
    return record[key]


def _math_id(value: object) -> MathId:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("MATH-500 unique_id must be a string or integer")
    return value


def _required_id(record: ProblemRecord) -> MathId:
    return _math_id(_required(record, "unique_id"))


def _id_sort_key(unique_id: MathId) -> tuple[int, int | str]:
    if isinstance(unique_id, int):
        return (0, unique_id)
    return (1, unique_id)


def build_math_pairs(
    problems: Iterable[ProblemRecord],
    generate_fn: GenerateFn,
    verify_fn: VerifyFn,
    n_pairs: int = 50,
    initial_pairs: Iterable[MathPair] = (),
    on_pair: PairCallback | None = None,
) -> list[MathPair]:
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")

    pairs = list(initial_pairs)
    if len(pairs) > n_pairs:
        raise ValueError(
            f"Found {len(pairs)} existing pairs for a {n_pairs}-pair build"
        )
    seen_ids = {pair.unique_id for pair in pairs}
    if len(seen_ids) != len(pairs):
        raise ValueError("Existing MATH-500 pairs contain duplicate unique IDs")
    if len(pairs) == n_pairs:
        return sort_math_pairs(pairs)

    ordered = sorted(problems, key=lambda record: _id_sort_key(_required_id(record)))

    for record in ordered:
        unique_id = _required_id(record)
        if unique_id in seen_ids:
            continue
        problem = str(_required(record, "problem"))
        gold_answer = str(_required(record, "answer"))

        verbose = generate_fn(record, "verbose", None).strip()
        if not verbose or not verify_fn(gold_answer, verbose):
            continue

        concise = generate_fn(record, "concise", verbose).strip()
        if concise and verify_fn(gold_answer, concise):
            pair = MathPair(
                unique_id=unique_id,
                problem=problem,
                gold_answer=gold_answer,
                verbose_solution=verbose,
                concise_solution=concise,
            )
            pairs.append(pair)
            seen_ids.add(unique_id)
            if on_pair is not None:
                on_pair(pair)
        if len(pairs) == n_pairs:
            return sort_math_pairs(pairs)

    raise ValueError(
        f"Required {n_pairs} verified MATH-500 pairs, but only built {len(pairs)}"
    )


def write_math_pairs_jsonl(path: str | Path, pairs: Iterable[MathPair]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def append_math_pair_jsonl(path: str | Path, pair: MathPair) -> None:
    destination = Path(path)
    lock_path = destination.with_name(destination.name + ".lock")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(lock_path):
        existing = read_math_pairs_jsonl(destination) if destination.exists() else []
        existing.append(pair)
        write_math_pairs_jsonl(destination, existing)


def _file_lock(lock_path: Path):
    import fcntl
    import contextlib

    @contextlib.contextmanager
    def _cm():
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "w")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            lock_path.unlink(missing_ok=True)

    return _cm()


def read_math_pairs_jsonl(path: str | Path) -> list[MathPair]:
    source = Path(path)
    pairs: list[MathPair] = []
    lines = source.read_text(encoding="utf-8").splitlines()
    last_index = len(lines)
    while last_index > 0 and not lines[last_index - 1].strip():
        last_index -= 1
    for line_number, line in enumerate(lines[:last_index], start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}") from exc
        try:
            pair = MathPair(
                unique_id=_math_id(record["unique_id"]),
                problem=str(record["problem"]),
                gold_answer=str(record["gold_answer"]),
                verbose_solution=str(record["verbose_solution"]),
                concise_solution=str(record["concise_solution"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid math pair on line {line_number}") from exc
        pairs.append(pair)
    return pairs
