#!/usr/bin/env python3
"""CLI entry point for the raw-vs-chat direction SENSITIVITY analysis.

This is a deliberately ISOLATED, secondary diagnostic. It does NOT touch the
primary concept-dynamics pipeline (``logs/concept_dynamics_multi``) or any
metrics output. It reads:

* the isolated PRIMARY RAW vectors (``--raw-vectors-dir``), and
* the OLD chat vectors for the related/control concepts
  (``--chat-old-vectors-dir``, READ-ONLY), and
* the EXTRACTED chat direction for the target concept
  (``--chat-target-vectors-dir``, under ``sensitivity/`` only),

then writes one atomic file: ``<--output-dir>/sensitivity.json``.

Scope is hard-coded to the sensitivity protocol:

* model        : ``olmo3-rl-zero-code``
* checkpoints  : ``step_100``, ``step_1700``, ``step_2900``
* concepts (6) : ``python_valid_vs_syntax_error`` (target, NOT in old chat
                 results) + 4 related code concepts + ``gender_she_vs_he``
                 (control)
* layers       : the 10 uniform Olmo-3-7B layers

Usage
-----

Default (compare whatever raw + chat vectors already exist; record target as
``chat_missing`` until extracted)::

    uv run python scripts/run_sensitivity_analysis.py

Override the isolated raw store::

    uv run python scripts/run_sensitivity_analysis.py \\
        --raw-vectors-dir logs/concept_dynamics_raw/vectors

Gated, resumable extraction of the MISSING chat-TARGET direction only. This
loads a real model + tokenizer and requires ``tokenizer.apply_chat_template``
to be available; it FAILS CLEARLY otherwise and never silently falls back to
raw text::

    uv run python scripts/run_sensitivity_analysis.py --extract-target-chat

The CLI NEVER writes into ``concept_dynamics_multi`` or any metrics location.
A path-isolation guard refuses output dirs that lie inside a primary store.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable

# Make ``src`` importable when run as a script.

from postdyn.sensitivity_analysis import (  # noqa: E402
    ChatTemplateUnavailableError,
    LIMITATIONS,
    PROTOCOL,
    RELATED_CONCEPTS,
    SENSITIVITY_CHECKPOINTS,
    SENSITIVITY_CONCEPTS,
    SENSITIVITY_LAYERS,
    SENSITIVITY_MODEL,
    SENSITIVITY_N_SAMPLES,
    SENSITIVITY_D_MODEL,
    TARGET_CONCEPT,
    CONTROL_CONCEPT,
    extract_missing_chat_target,
    run_sensitivity_analysis,
)
from postdyn.rl_zero_experiment import RL_ZERO_CODE_RESULTS_ROOT  # noqa: E402

#: Default isolated primary raw store (layout mirrors concept_dynamics_multi).
DEFAULT_RAW_VECTORS_DIR = os.path.join(RL_ZERO_CODE_RESULTS_ROOT, "concept_vectors")

#: Default old chat results (READ-ONLY reuse for related/control).
DEFAULT_CHAT_OLD_VECTORS_DIR = os.path.join(
    "logs", "concept_dynamics_multi", "vectors"
)

#: Default sensitivity output root (NEVER inside concept_dynamics_multi).
DEFAULT_SENSITIVITY_DIR = os.path.join("logs", "sensitivity")

#: Chat-TARGET extraction root, kept strictly under the sensitivity dir.
DEFAULT_CHAT_TARGET_VECTORS_DIR = os.path.join(
    DEFAULT_SENSITIVITY_DIR, "chat_target_vectors"
)

DEFAULT_OUTPUT_FILENAME = "sensitivity.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Raw-vs-chat direction sensitivity analysis (SECONDARY, isolated). "
            "Produces sensitivity/sensitivity.json; never touches primary metrics."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--raw-vectors-dir",
        default=DEFAULT_RAW_VECTORS_DIR,
        help=(
            "Isolated PRIMARY RAW vectors root (layout mirrors "
            "concept_dynamics_multi/vectors). Default: "
            f"{DEFAULT_RAW_VECTORS_DIR}"
        ),
    )
    parser.add_argument(
        "--chat-old-vectors-dir",
        default=DEFAULT_CHAT_OLD_VECTORS_DIR,
        help=(
            "OLD chat results root, READ-ONLY (related/control reuse). "
            f"Default: {DEFAULT_CHAT_OLD_VECTORS_DIR}"
        ),
    )
    parser.add_argument(
        "--chat-target-vectors-dir",
        default=DEFAULT_CHAT_TARGET_VECTORS_DIR,
        help=(
            "Root for the EXTRACTED chat-TARGET direction (under sensitivity/ "
            "only). Default: "
            f"{DEFAULT_CHAT_TARGET_VECTORS_DIR}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_SENSITIVITY_DIR,
        help=(
            "Where to write sensitivity.json. MUST be isolated from the "
            "primary stores (a guard rejects nesting). "
            f"Default: {DEFAULT_SENSITIVITY_DIR}"
        ),
    )
    parser.add_argument(
        "--output-filename",
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Output filename (default: {DEFAULT_OUTPUT_FILENAME})",
    )
    parser.add_argument(
        "--extract-target-chat",
        action="store_true",
        help=(
            "Gated step: load a REAL model + tokenizer and extract the MISSING "
            "chat direction for the target concept (python_valid_vs_syntax_error) "
            "into --chat-target-vectors-dir, using the tokenizer's actual chat "
            "template. Fails clearly if the model or chat template is unavailable. "
            "Off by default (sensitivity records target as 'chat_missing')."
        ),
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Max tokenization length for the gated target extraction (default 2048).",
    )
    return parser.parse_args(argv)


def _print_protocol_banner() -> None:
    print("=" * 72)
    print(
        f"SENSITIVITY ANALYSIS (SECONDARY) — {PROTOCOL['name']} v{PROTOCOL['version']}"
    )
    print("=" * 72)
    print(f"  model        : {SENSITIVITY_MODEL}")
    print(f"  checkpoints  : {SENSITIVITY_CHECKPOINTS}")
    print(f"  concepts (6) : target={TARGET_CONCEPT}")
    print(f"                  related={RELATED_CONCEPTS}")
    print(f"                  control={CONTROL_CONCEPT}")
    print(f"  layers       : {SENSITIVITY_LAYERS}")
    print(f"  samples/class: {SENSITIVITY_N_SAMPLES}  d_model={SENSITIVITY_D_MODEL}")
    print()
    print("  LIMITATIONS (recorded in output):")
    for lim in LIMITATIONS:
        print(f"    - {lim}")
    print("=" * 72)


def _extract_target_step(
    args: argparse.Namespace,
    *,
    extract_fn: Callable[..., dict[str, Any]] | None = None,
    load_model_and_tokenizer: Callable[..., tuple[Any, Any]] | None = None,
) -> dict[str, Any] | None:
    """Run the gated chat-target extraction, failing clearly on problems.

    Returns the extraction summary dict, or ``None`` if there was nothing to do
    (left to the comparison pass to report as ``chat_missing``).
    """
    if not args.extract_target_chat:
        return None

    print("\n--- Gated extraction: chat-TARGET direction ---")
    print(f"  concept   : {TARGET_CONCEPT}")
    print(f"  output    : {args.chat_target_vectors_dir}")
    print(
        "  NOTE: this loads a real model and requires the tokenizer's "
        "apply_chat_template. It will FAIL CLEARLY if either is unavailable."
    )
    try:
        summary = extract_missing_chat_target(
            chat_target_vectors_dir=args.chat_target_vectors_dir,
            sensitivity_output_root=args.output_dir,
            raw_vectors_dir=args.raw_vectors_dir,
            chat_old_vectors_dir=args.chat_old_vectors_dir,
            max_seq_len=args.max_seq_len,
            extract_fn=extract_fn,
            load_model_and_tokenizer=load_model_and_tokenizer,
        )
    except ChatTemplateUnavailableError as exc:
        print(
            f"\nERROR: chat-template extraction unavailable:\n  {exc}\n\n"
            "The sensitivity driver refuses to fall back to raw text for a "
            "CHAT direction. Install/load the model + tokenizer with a usable "
            "apply_chat_template, then re-run with --extract-target-chat.",
            file=sys.stderr,
        )
        sys.exit(3)
    except FileNotFoundError as exc:
        print(
            f"\nERROR: model or tokenizer file not found during gated "
            f"extraction:\n  {exc}",
            file=sys.stderr,
        )
        sys.exit(4)

    extracted = summary["checkpoints"]["extracted"]
    skipped = summary["checkpoints"]["skipped_present"]
    print(
        f"  extracted : {len(extracted)} checkpoint(s); "
        f"skipped (already present): {len(skipped)}"
    )
    return summary


def main(
    argv: list[str] | None = None,
    *,
    extract_fn: Callable[..., dict[str, Any]] | None = None,
    load_model_and_tokenizer: Callable[..., tuple[Any, Any]] | None = None,
) -> int:
    """CLI entry point.

    The optional ``extract_fn`` / ``load_model_and_tokenizer`` hooks exist so
    tests can drive the full CLI with mocked model I/O instead of a real GPU.

    Returns:
        Process exit code (0 on success).
    """
    args = parse_args(argv)

    _check_output_isolation(args)
    _check_chat_target_isolation(args)

    _print_protocol_banner()

    _extract_target_step(
        args,
        extract_fn=extract_fn,
        load_model_and_tokenizer=load_model_and_tokenizer,
    )

    print("\n--- Raw-vs-chat comparison ---")
    print(f"  raw vectors (isolated) : {args.raw_vectors_dir}")
    print(f"  chat old (READ-ONLY)   : {args.chat_old_vectors_dir}")
    print(f"  chat target (extracted): {args.chat_target_vectors_dir}")
    print(
        f"  output                 : {os.path.join(args.output_dir, args.output_filename)}"
    )
    print()

    payload = run_sensitivity_analysis(
        raw_vectors_dir=args.raw_vectors_dir,
        chat_old_vectors_dir=args.chat_old_vectors_dir,
        chat_target_vectors_dir=args.chat_target_vectors_dir,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
    )

    s = payload["summary"]
    print(f"Sensitivity written: {os.path.join(args.output_dir, args.output_filename)}")
    print(
        f"  entries={s['n_entries']}  compared={s['n_compared']}  "
        f"raw_missing={s['n_raw_missing']}  chat_missing={s['n_chat_missing']}  "
        f"metadata_rejected={s['n_metadata_rejected']}"
    )
    if s["n_chat_missing"]:
        print(
            "  NOTE: chat_missing entries need the target chat direction; "
            "re-run with --extract-target-chat once a model + chat template "
            "are available."
        )
    return 0


def _check_output_isolation(args: argparse.Namespace) -> None:
    """Refuse output dirs nested inside any primary store; validate filename.

    Validates ``output_filename`` is a plain basename early so the CLI fails
    before any banner / model work, and checks ``output_dir`` is isolated from
    the primary stores.
    """
    from postdyn.sensitivity_analysis import (
        _assert_output_containment,
        _assert_path_isolation,
    )

    try:
        _assert_path_isolation(
            args.output_dir, args.chat_old_vectors_dir, args.raw_vectors_dir
        )
        _assert_output_containment(args.output_dir, args.output_filename)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


def _check_chat_target_isolation(args: argparse.Namespace) -> None:
    """Refuse chat-target dirs that escape the sensitivity root or hit a store.

    The chat-target dir is the ONLY place the CLI ever WRITES during the gated
    extraction, so it must nest strictly under ``--output-dir`` (the designated
    sensitivity root) and never inside the raw or old-chat primary stores.
    """
    from postdyn.sensitivity_analysis import _assert_chat_target_isolation

    try:
        _assert_chat_target_isolation(
            args.chat_target_vectors_dir,
            sensitivity_output_root=args.output_dir,
            raw_vectors_dir=args.raw_vectors_dir,
            chat_old_vectors_dir=args.chat_old_vectors_dir,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    sys.exit(main())
