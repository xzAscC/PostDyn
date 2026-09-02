"""Typed configuration for the RL-Zero-Code *Python syntax* concept experiment.

This module centralizes every knob of the 11-checkpoint syntax experiment so
that extraction, sensitivity, and downstream-evaluation code share one source
of truth. It is **data/config only**: it runs no model, writes no results,
mutates no checkpoint config, and never touches ``postdyn/config.py``.

What lives here
---------------
* **Isolated results root** -- ``RL_ZERO_CODE_RESULTS_ROOT`` is deliberately
  distinct from ``logs/concept_dynamics_multi`` (the paired-concept run).
* **Eleven checkpoints** -- ``olmo3-base`` revision ``"main"`` plus the ten
  uniformly-spaced RL-Zero-Code steps, reused verbatim from
  ``postdyn.config.MODEL_CHECKPOINTS`` (never re-derived here).
* **Ten layers** -- ``postdyn.config.EXPERIMENT_LAYERS_7B``, reused verbatim.
* **Six concepts** -- one target (``python_valid_vs_syntax_error``) + four
  related code-language concepts + one gender control.
* **Eight probe classes** -- ``python_valid``, ``python_syntax_error``,
  ``cpp``, ``js``, ``java``, ``go``, ``she``, ``he``.
* **Sample count & sensitivity sub-steps** -- 50 paired records; the three
  RL steps ``step_100`` / ``step_1700`` / ``step_2900``.
* **Protocol** -- primary extraction is **raw text** across all eleven
  checkpoints; existing chat-template vectors are read-only sensitivity
  inputs only.
* **Downstream-data helpers** -- load and validate the 50 downstream
  HumanEval-X items and confirm target ids are disjoint.

Model facts (architecture, layer count, revisions, checkpoint schedules) are
imported from :mod:`postdyn.config`; this module never duplicates them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from postdyn.config import (
    EXPERIMENT_LAYERS_7B,
    MODEL_CHECKPOINTS,
    OLMO3_VARIANTS,
    PROJECT_ROOT,
    LOGS_DIR,
)
from postdyn.config import ModelConfig  # re-exported type only

# =============================================================================
# Results isolation
# =============================================================================

#: Distinct results root for this experiment. Must NEVER equal any paired-concept
#: results directory -- the syntax experiment is fully isolated.
RL_ZERO_CODE_RESULTS_ROOT: str = os.path.join(LOGS_DIR, "rl_zero_code_syntax")

#: Quick / smoke variant of the results root.
RL_ZERO_CODE_RESULTS_ROOT_QUICK: str = RL_ZERO_CODE_RESULTS_ROOT + "_quick"

#: The existing paired-concept results directories. These are **read-only
#: sensitivity inputs only** (their chat-template vectors feed the sensitivity
#: sweep); this experiment never writes into them.
PAIRED_CONCEPT_RESULTS_ROOT: str = os.path.join(LOGS_DIR, "concept_dynamics_multi")
PAIRED_CONCEPT_RESULTS_ROOT_QUICK: str = os.path.join(
    LOGS_DIR, "concept_dynamics_multi_quick"
)
#: Legacy directory names that the syntax root must also avoid colliding with.
_PAIRED_CONCEPT_LEGACY_ROOTS: frozenset[str] = frozenset(
    {
        os.path.join(LOGS_DIR, "concept_dynamics"),
        os.path.join(LOGS_DIR, "concept_dynamics_paired"),
    }
)


def results_root(*, quick: bool = False, override: str | None = None) -> str:
    """Resolve the results root for this experiment.

    An explicit ``override`` always wins; otherwise the quick/full default is
    used. The resolved path is guaranteed to differ from every paired-concept
    results directory.
    """
    if override is not None:
        return override
    return RL_ZERO_CODE_RESULTS_ROOT_QUICK if quick else RL_ZERO_CODE_RESULTS_ROOT


# =============================================================================
# Model & checkpoint schedule (reused from postdyn.config -- not duplicated)
# =============================================================================

#: Shared starting point of every OLMo-3 post-training branch.
BASE_MODEL_KEY: str = "olmo3-base"
#: The single post-training trajectory under study.
TARGET_MODEL_KEY: str = "olmo3-rl-zero-code"

#: Frozen views of the two model configs (architecture / hf_id / layers ...).
BASE_MODEL: ModelConfig = OLMO3_VARIANTS[BASE_MODEL_KEY]
TARGET_MODEL: ModelConfig = OLMO3_VARIANTS[TARGET_MODEL_KEY]

#: Base checkpoint name (``olmo3-base`` ships only the ``main`` revision).
BASE_CHECKPOINT: str = "main"

#: Ten uniformly-spaced RL-Zero-Code checkpoints, reused verbatim from
#: ``postdyn.config.MODEL_CHECKPOINTS``. These are the ``step_100 .. step_2900``
#: branch names selected by ``select_uniform_checkpoints``.
RL_CHECKPOINTS: list[str] = list(MODEL_CHECKPOINTS[TARGET_MODEL_KEY])

#: Full eleven-checkpoint schedule: base ``main`` first, then ten RL steps,
#: in training order.
EXPERIMENT_CHECKPOINTS: list[str] = [BASE_CHECKPOINT, *RL_CHECKPOINTS]


def is_base_checkpoint(checkpoint: str) -> bool:
    """True for the single base checkpoint (``main``)."""
    return checkpoint == BASE_CHECKPOINT


def is_rl_checkpoint(checkpoint: str) -> bool:
    """True for any of the ten RL-Zero-Code step checkpoints."""
    return checkpoint in RL_CHECKPOINTS


# =============================================================================
# Layers (reused verbatim from postdyn.config -- never re-derived)
# =============================================================================

#: Ten slide-formula layer indices for OLMo-3 7B (32 transformer blocks).
#: Identical object content to ``postdyn.config.EXPERIMENT_LAYERS_7B``; copied
#: into a fresh list so callers cannot mutate the shared constant.
EXPERIMENT_LAYERS: list[int] = list(EXPERIMENT_LAYERS_7B)


# =============================================================================
# Sample count & sensitivity sub-steps
# =============================================================================

#: Paired records per concept class. Matches the data-only builder's 50
#: ``python_syntax_pairs.json`` records and ``DEFAULT_N_SAMPLES`` in
#: :mod:`postdyn.contrastive_datasets`.
N_SAMPLES: int = 50

#: Three RL checkpoints reserved for the sensitivity sweep. These are the
#: first, middle, and last of the ten RL steps, and are always a subset of
#: ``RL_CHECKPOINTS``.
SENSITIVITY_STEPS: tuple[str, ...] = ("step_100", "step_1700", "step_2900")


# =============================================================================
# Protocol
# =============================================================================

#: Primary extraction protocol: **raw text** across all eleven checkpoints
#: (no chat template). Per the experiment brief this is the canonical mode.
PRIMARY_USE_CHAT_TEMPLATE: bool = False

#: Existing chat-template concept vectors are read-only sensitivity inputs
#: only; they feed ``SENSITIVITY_STEPS`` analysis and are never overwritten.
SENSITIVITY_INPUT_RESULTS_ROOT: str = PAIRED_CONCEPT_RESULTS_ROOT


# =============================================================================
# Probe classes & concepts
# =============================================================================

#: The eight probe classes exercised across the experiment, in canonical order.
#: ``python_valid`` / ``python_syntax_error`` are the target concept's two
#: classes; ``cpp`` / ``js`` / ``java`` / ``go`` are the four non-python
#: HumanEval-X languages; ``she`` / ``he`` are the gender control classes.
PROBE_CLASSES: tuple[str, ...] = (
    "python_valid",
    "python_syntax_error",
    "cpp",
    "js",
    "java",
    "go",
    "she",
    "he",
)

#: The target syntax-validity concept (domain ``"syntax"`` in the shared
#: registry; materialized by the data-only builder under
#: ``data/allenai/Dolci-RL-Zero-Code-7B``).
TARGET_CONCEPT: str = "python_valid_vs_syntax_error"

#: Four related code-language concepts. Python (here labelled ``python_valid``)
#: is the negative/A class; each non-python language is the positive/B class.
#: Order follows ``PROBE_CLASSES`` (cpp, js, java, go).
RELATED_CONCEPTS: tuple[str, ...] = (
    "code_python_vs_cpp",
    "code_python_vs_js",
    "code_python_vs_java",
    "code_python_vs_go",
)

#: Control concept: gender pronouns, unrelated to code -- a sanity probe for
#: whether the syntax concept localizes to code at all.
CONTROL_CONCEPT: str = "gender_she_vs_he"

#: All six concepts in experiment order: target + four related + control.
EXPERIMENT_CONCEPTS: tuple[str, ...] = (
    TARGET_CONCEPT,
    *RELATED_CONCEPTS,
    CONTROL_CONCEPT,
)

#: Concept role lookup (``"target"`` / ``"related"`` / ``"control"``).
CONCEPT_ROLE: dict[str, str] = {
    TARGET_CONCEPT: "target",
    **{key: "related" for key in RELATED_CONCEPTS},
    CONTROL_CONCEPT: "control",
}

#: ``(positive_class, negative_class)`` for each concept in experiment polarity
#: (``+B -A``, arrow ``A -> B``). The python code class is uniformly labelled
#: ``python_valid`` so it coincides with the target concept's positive class.
CONCEPT_PROBE_CLASSES: dict[str, tuple[str, str]] = {
    TARGET_CONCEPT: ("python_valid", "python_syntax_error"),
    "code_python_vs_cpp": ("cpp", "python_valid"),
    "code_python_vs_js": ("js", "python_valid"),
    "code_python_vs_java": ("java", "python_valid"),
    "code_python_vs_go": ("go", "python_valid"),
    CONTROL_CONCEPT: ("he", "she"),
}


@dataclass(frozen=True)
class ConceptSpec:
    """Typed description of one experiment concept."""

    key: str
    role: str  # "target" | "related" | "control"
    positive_class: str
    negative_class: str
    domain: str
    #: Registered in ``postdyn.contrastive_datasets``? The target concept is not.
    registered: bool


def concept_specs() -> dict[str, ConceptSpec]:
    """Build :class:`ConceptSpec` objects, reusing the contrastive registry.

    All six concepts are looked up in :mod:`postdyn.contrastive_datasets` so their
    domain is never duplicated here; probe-class polarity comes from
    :data:`CONCEPT_PROBE_CLASSES` (the experiment's own ``python_valid`` /
    ``python_syntax_error`` labels, which alias the registry's
    ``syntax_valid`` / ``syntax_error`` classes).
    """
    # Imported lazily so this module stays importable in minimal envs and so
    # the registry is only touched when actually needed.
    from postdyn.contrastive_datasets import CONCEPTS as _REGISTRY

    specs: dict[str, ConceptSpec] = {}
    for key in EXPERIMENT_CONCEPTS:
        meta = _REGISTRY.get(key)
        if meta is None:
            raise KeyError(
                f"experiment concept {key!r} is missing from "
                "postdyn.contrastive_datasets.CONCEPTS"
            )
        positive_class, negative_class = CONCEPT_PROBE_CLASSES[key]
        specs[key] = ConceptSpec(
            key=key,
            role=CONCEPT_ROLE[key],
            positive_class=positive_class,
            negative_class=negative_class,
            domain=meta["domain"],
            registered=True,
        )
    return specs


def probe_classes_used() -> set[str]:
    """Every probe class referenced by the six concepts."""
    used: set[str] = set()
    for pos, neg in CONCEPT_PROBE_CLASSES.values():
        used.add(pos)
        used.add(neg)
    return used


# =============================================================================
# Data-only artifacts & downstream-data helpers
# =============================================================================

#: Directory holding the builder's auditable JSON artifacts.
CONCEPT_DATA_DIR: Path = (
    Path(PROJECT_ROOT) / "data" / "allenai" / "Dolci-RL-Zero-Code-7B"
)
#: 50 paired records for the target concept.
PAIRS_FILE: Path = CONCEPT_DATA_DIR / "python_syntax_pairs.json"
#: Downstream HumanEval-X (50) + MMLU (50) items, disjoint from the target ids.
DOWNSTREAM_FILE: Path = CONCEPT_DATA_DIR / "downstream.json"

#: Downstream HumanEval-X item count enforced by the helpers.
DOWNSTREAM_HUMANEVAL_X_ITEMS: int = N_SAMPLES
#: Downstream MMLU question count enforced by the helpers.
DOWNSTREAM_MMLU_ITEMS: int = N_SAMPLES


class DownstreamDataError(ValueError):
    """Raised when a downstream-data invariant is violated."""


def load_pairs() -> dict[str, Any]:
    """Load ``python_syntax_pairs.json``.

    Raises ``FileNotFoundError`` with a build-hint if the artifact is absent.
    """
    if not PAIRS_FILE.exists():
        raise FileNotFoundError(
            f"{PAIRS_FILE} not built; run scripts/build_rl_zero_syntax_concept.py --only pairs"
        )
    return json.loads(PAIRS_FILE.read_text(encoding="utf-8"))


def load_downstream() -> dict[str, Any]:
    """Load ``downstream.json``.

    Raises ``FileNotFoundError`` with a build-hint if the artifact is absent.
    """
    if not DOWNSTREAM_FILE.exists():
        raise FileNotFoundError(
            f"{DOWNSTREAM_FILE} not built; run scripts/build_rl_zero_syntax_concept.py --only downstream"
        )
    return json.loads(DOWNSTREAM_FILE.read_text(encoding="utf-8"))


def target_ids(pairs: dict[str, Any]) -> set[str]:
    """Target HumanEval-X task ids used to build the concept pairs."""
    return set(pairs["selection"]["target_ids"])


def downstream_humaneval_ids(downstream: dict[str, Any]) -> set[str]:
    """Pinned downstream HumanEval-X task ids (must be disjoint from target)."""
    return set(downstream["humaneval_x"]["task_ids"])


def humaneval_x_item_count(downstream: dict[str, Any]) -> int:
    """Reported downstream HumanEval-X item count."""
    return int(downstream["humaneval_x"]["n_items"])


def mmlu_question_count(downstream: dict[str, Any]) -> int:
    """Reported downstream MMLU question count."""
    return int(downstream["mmlu"]["n_questions"])


def validate_downstream_counts(
    downstream: dict[str, Any],
    *,
    expected_humaneval_x: int = DOWNSTREAM_HUMANEVAL_X_ITEMS,
    expected_mmlu: int = DOWNSTREAM_MMLU_ITEMS,
) -> None:
    """Validate the downstream item counts.

    Ensures the HumanEval-X block carries exactly ``expected_humaneval_x``
    items and that the reported ``n_items`` matches the materialized list,
    and likewise for the MMLU block.
    """
    hx = downstream.get("humaneval_x")
    if not isinstance(hx, dict):
        raise DownstreamDataError("downstream.json missing 'humaneval_x' block")
    items = hx.get("items")
    if not isinstance(items, list):
        raise DownstreamDataError("downstream 'humaneval_x.items' is not a list")
    if hx.get("n_items") != expected_humaneval_x:
        raise DownstreamDataError(
            f"downstream humaneval_x n_items={hx.get('n_items')!r}, expected {expected_humaneval_x}"
        )
    if len(items) != expected_humaneval_x:
        raise DownstreamDataError(
            f"downstream humaneval_x has {len(items)} items, expected {expected_humaneval_x}"
        )
    mmlu = downstream.get("mmlu")
    if not isinstance(mmlu, dict):
        raise DownstreamDataError("downstream.json missing 'mmlu' block")
    mitems = mmlu.get("items")
    if not isinstance(mitems, list):
        raise DownstreamDataError("downstream 'mmlu.items' is not a list")
    if mmlu.get("n_questions") != expected_mmlu:
        raise DownstreamDataError(
            f"downstream mmlu n_questions={mmlu.get('n_questions')!r}, expected {expected_mmlu}"
        )
    if len(mitems) != expected_mmlu:
        raise DownstreamDataError(
            f"downstream mmlu has {len(mitems)} items, expected {expected_mmlu}"
        )


def validate_target_downstream_disjoint(
    pairs: dict[str, Any],
    downstream: dict[str, Any],
) -> None:
    """Validate that target ids and downstream HumanEval-X ids are disjoint.

    Also honours the builder's own ``disjointness_verified`` flag.
    """
    selection = pairs.get("selection")
    if not isinstance(selection, dict):
        raise DownstreamDataError("pairs missing 'selection' block")
    target = set(selection.get("target_ids", []))
    downstream_ids_set = downstream_humaneval_ids(downstream)
    overlap = target & downstream_ids_set
    if overlap:
        raise DownstreamDataError(
            f"target ids overlap downstream humaneval_x ids: {sorted(overlap)}"
        )
    if selection.get("disjointness_verified") is not True:
        raise DownstreamDataError("pairs 'selection.disjointness_verified' is not True")
    excluded = set(selection.get("excluded_downstream_ids", []))
    if excluded != downstream_ids_set:
        raise DownstreamDataError(
            "pairs 'selection.excluded_downstream_ids' does not match the downstream humaneval_x task ids"
        )


def validate_downstream(
    pairs: dict[str, Any] | None = None,
    downstream: dict[str, Any] | None = None,
) -> None:
    """End-to-end downstream validation: 50 items + disjoint target ids.

    Loads the artifacts lazily when the caller does not supply them, so this
    is the single entry point extraction / eval code should call.
    """
    if pairs is None:
        pairs = load_pairs()
    if downstream is None:
        downstream = load_downstream()
    validate_downstream_counts(downstream)
    validate_target_downstream_disjoint(pairs, downstream)


# =============================================================================
# Structural invariants
# =============================================================================


def self_check() -> None:
    """Assert every structural invariant of the experiment configuration.

    Runs no model and touches no data files. The contrastive-registry cross
    check imports :mod:`postdyn.contrastive_datasets` lazily. Raises ``AssertionError``
    on the first violation; returns ``None`` on success.
    """
    # --- Results isolation -------------------------------------------------
    assert RL_ZERO_CODE_RESULTS_ROOT != PAIRED_CONCEPT_RESULTS_ROOT, (
        "syntax results root must be isolated from the paired-concept run"
    )
    assert RL_ZERO_CODE_RESULTS_ROOT not in _PAIRED_CONCEPT_LEGACY_ROOTS, (
        "syntax results root must not collide with any legacy paired root"
    )
    assert RL_ZERO_CODE_RESULTS_ROOT_QUICK != PAIRED_CONCEPT_RESULTS_ROOT_QUICK, (
        "quick syntax results root must be isolated from the quick paired run"
    )
    assert not RL_ZERO_CODE_RESULTS_ROOT.endswith("concept_dynamics_multi"), (
        "syntax results root must live outside concept_dynamics_multi"
    )

    # --- Model facts reused, not duplicated --------------------------------
    assert BASE_MODEL_KEY in OLMO3_VARIANTS, (
        "base model key must exist in OLMO3_VARIANTS"
    )
    assert TARGET_MODEL_KEY in OLMO3_VARIANTS, (
        "target model key must exist in OLMO3_VARIANTS"
    )
    assert TARGET_MODEL_KEY in MODEL_CHECKPOINTS, (
        "RL checkpoints must be reused from MODEL_CHECKPOINTS"
    )
    assert BASE_MODEL_KEY not in MODEL_CHECKPOINTS, (
        "base must not carry a step schedule (it only has 'main')"
    )

    # --- Checkpoint schedule (11 = main + 10) ------------------------------
    assert len(RL_CHECKPOINTS) == 10, (
        f"expected 10 RL checkpoints, got {len(RL_CHECKPOINTS)}"
    )
    assert BASE_CHECKPOINT == "main", "base checkpoint must be the 'main' revision"
    assert len(EXPERIMENT_CHECKPOINTS) == 11, (
        f"expected 11 checkpoints, got {len(EXPERIMENT_CHECKPOINTS)}"
    )
    assert EXPERIMENT_CHECKPOINTS[0] == BASE_CHECKPOINT, (
        "schedule must start at the base checkpoint"
    )
    assert EXPERIMENT_CHECKPOINTS[1:] == RL_CHECKPOINTS, (
        "schedule tail must equal the reused RL_CHECKPOINTS"
    )
    assert len(set(EXPERIMENT_CHECKPOINTS)) == 11, "checkpoints must be unique"

    # --- Sensitivity sub-steps --------------------------------------------
    rl_set = set(RL_CHECKPOINTS)
    assert set(SENSITIVITY_STEPS) <= rl_set, (
        f"sensitivity steps {set(SENSITIVITY_STEPS) - rl_set} are not RL checkpoints"
    )
    assert len(SENSITIVITY_STEPS) == 3, "exactly three sensitivity steps"
    # Ordered: first < middle < last of the RL schedule.
    positions = [RL_CHECKPOINTS.index(s) for s in SENSITIVITY_STEPS]
    assert positions == sorted(positions), "sensitivity steps must be in RL order"
    assert positions[0] == 0, "first sensitivity step must be the first RL checkpoint"
    assert positions[-1] == len(RL_CHECKPOINTS) - 1, (
        "last sensitivity step must be the final RL checkpoint"
    )

    # --- Layers (reused) ---------------------------------------------------
    assert EXPERIMENT_LAYERS == list(EXPERIMENT_LAYERS_7B), (
        "layers must be reused verbatim from postdyn.config.EXPERIMENT_LAYERS_7B"
    )
    assert len(EXPERIMENT_LAYERS) == 10, "exactly ten layers"

    # --- Samples -----------------------------------------------------------
    assert N_SAMPLES == 50, "sample count must be 50"
    assert DOWNSTREAM_HUMANEVAL_X_ITEMS == N_SAMPLES, (
        "downstream item count must track the sample count"
    )

    # --- Protocol ----------------------------------------------------------
    assert PRIMARY_USE_CHAT_TEMPLATE is False, (
        "primary protocol must be raw text (no chat template)"
    )

    # --- Probe classes (8, exact set) -------------------------------------
    assert PROBE_CLASSES == (
        "python_valid",
        "python_syntax_error",
        "cpp",
        "js",
        "java",
        "go",
        "she",
        "he",
    ), "probe classes must be exactly the eight specified labels"
    assert len(set(PROBE_CLASSES)) == 8, "probe classes must be unique"

    # --- Concepts (6 = target + 4 related + control) ----------------------
    assert len(EXPERIMENT_CONCEPTS) == 6, "exactly six concepts"
    assert EXPERIMENT_CONCEPTS[0] == TARGET_CONCEPT, "first concept is the target"
    assert EXPERIMENT_CONCEPTS[-1] == CONTROL_CONCEPT, "last concept is the control"
    assert len(RELATED_CONCEPTS) == 4, "exactly four related concepts"
    assert tuple(EXPERIMENT_CONCEPTS[1:-1]) == RELATED_CONCEPTS, (
        "middle concepts must be the four related ones"
    )
    assert len(set(EXPERIMENT_CONCEPTS)) == 6, "concepts must be unique"

    # --- Concept -> probe-class coverage ----------------------------------
    assert set(CONCEPT_PROBE_CLASSES) == set(EXPERIMENT_CONCEPTS), (
        "every concept must declare its probe classes"
    )
    assert probe_classes_used() == set(PROBE_CLASSES), (
        "union of concept probe classes must equal the eight PROBE_CLASSES"
    )
    for key, (pos, neg) in CONCEPT_PROBE_CLASSES.items():
        assert pos in PROBE_CLASSES, (
            f"concept {key!r} positive class {pos!r} not in PROBE_CLASSES"
        )
        assert neg in PROBE_CLASSES, (
            f"concept {key!r} negative class {neg!r} not in PROBE_CLASSES"
        )
        assert pos != neg, f"concept {key!r} has identical positive/negative classes"

    # --- Registry membership ----------------------------------------------
    from postdyn.contrastive_datasets import CONCEPTS as _REGISTRY  # lazy

    for key in EXPERIMENT_CONCEPTS:
        assert key in _REGISTRY, (
            f"concept {key!r} must be registered in contrastive_datasets"
        )
    for key in RELATED_CONCEPTS:
        meta = _REGISTRY[key]
        assert meta["negative"] == "python", (
            f"related concept {key!r} must contrast python against another language"
        )
    assert _REGISTRY[CONTROL_CONCEPT]["domain"] == "general", (
        "control concept must have domain 'general'"
    )
    assert _REGISTRY[TARGET_CONCEPT]["domain"] == "syntax", (
        "target concept must have domain 'syntax'"
    )

    # --- Concept specs -----------------------------------------------------
    specs = concept_specs()
    assert set(specs) == set(EXPERIMENT_CONCEPTS), "spec keys must match concepts"
    for key in EXPERIMENT_CONCEPTS:
        assert specs[key].registered is True, (
            f"concept {key!r} spec must report registered=True"
        )
    assert specs[TARGET_CONCEPT].domain == "syntax", (
        "target spec domain must be 'syntax'"
    )
    assert specs[CONTROL_CONCEPT].domain == "general", (
        "control spec domain must be 'general'"
    )


# Run the cheap, dependency-free invariants at import so misconfiguration is
# caught immediately. The registry cross-check runs inside ``self_check()``.
assert RL_ZERO_CODE_RESULTS_ROOT != PAIRED_CONCEPT_RESULTS_ROOT
assert len(RL_CHECKPOINTS) == 10 and BASE_CHECKPOINT == "main"
assert len(EXPERIMENT_CHECKPOINTS) == 11
assert set(SENSITIVITY_STEPS) <= set(RL_CHECKPOINTS)
assert len(EXPERIMENT_LAYERS) == 10
assert len(PROBE_CLASSES) == 8 and len(set(PROBE_CLASSES)) == 8
assert len(EXPERIMENT_CONCEPTS) == 6
assert probe_classes_used() == set(PROBE_CLASSES)


__all__ = [
    "BASE_CHECKPOINT",
    "BASE_MODEL",
    "BASE_MODEL_KEY",
    "CONCEPT_DATA_DIR",
    "CONCEPT_PROBE_CLASSES",
    "CONCEPT_ROLE",
    "CONTROL_CONCEPT",
    "DOWNSTREAM_FILE",
    "DOWNSTREAM_HUMANEVAL_X_ITEMS",
    "DOWNSTREAM_MMLU_ITEMS",
    "DownstreamDataError",
    "EXPERIMENT_CHECKPOINTS",
    "EXPERIMENT_CONCEPTS",
    "EXPERIMENT_LAYERS",
    "PAIRS_FILE",
    "PAIRED_CONCEPT_RESULTS_ROOT",
    "PAIRED_CONCEPT_RESULTS_ROOT_QUICK",
    "PRIMARY_USE_CHAT_TEMPLATE",
    "PROBE_CLASSES",
    "RELATED_CONCEPTS",
    "RL_CHECKPOINTS",
    "RL_ZERO_CODE_RESULTS_ROOT",
    "RL_ZERO_CODE_RESULTS_ROOT_QUICK",
    "SENSITIVITY_INPUT_RESULTS_ROOT",
    "SENSITIVITY_STEPS",
    "TARGET_CONCEPT",
    "TARGET_MODEL",
    "TARGET_MODEL_KEY",
    "N_SAMPLES",
    "ConceptSpec",
    "concept_specs",
    "downstream_humaneval_ids",
    "humaneval_x_item_count",
    "is_base_checkpoint",
    "is_rl_checkpoint",
    "load_downstream",
    "load_pairs",
    "mmlu_question_count",
    "probe_classes_used",
    "results_root",
    "self_check",
    "target_ids",
    "validate_downstream",
    "validate_downstream_counts",
    "validate_target_downstream_disjoint",
]
