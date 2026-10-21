import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "data" / "evaluation_results.csv"
OUTPUT_FILE = ROOT / "data" / "scored_evaluation.csv"
SUMMARY_FILE = ROOT / "data" / "evaluation_summary.txt"


def split_requirements(value):
    """Semicolon = separate requirements; slash = acceptable alternatives."""
    if not value:
        return []
    requirements = []
    for item in value.split(";"):
        alternatives = [
            part.strip().lower()
            for part in item.split("/")
            if part.strip()
        ]
        if alternatives:
            requirements.append(alternatives)
    return requirements


def required_terms_pass(answer, requirements):
    answer = (answer or "").lower()
    return all(
        any(option in answer for option in alternatives)
        for alternatives in requirements
    )


def forbidden_terms_found(answer, requirements):
    answer = (answer or "").lower()
    return [
        option
        for alternatives in requirements
        for option in alternatives
        if option in answer
    ]


def evidence_hit(gold_evidence, retrieved_sources):
    """Check whether retrieved section IDs or source filenames match gold evidence."""
    if not gold_evidence.strip():
        return ""

    try:
        sources = json.loads(retrieved_sources or "[]")
    except (json.JSONDecodeError, TypeError):
        sources = []

    retrieved_ids = {
        str(item.get("section_id", "")).strip().lower()
        for item in sources
    }
    retrieved_names = {
        Path(str(item.get("source", "")).replace("\\", "/"))
        .name.lower()
        for item in sources
    }

    gold_items = [
        item.strip()
        for item in gold_evidence.split(";")
        if item.strip()
    ]

    for item in gold_items:
        if ":" in item:
            filename, identifier = item.split(":", 1)
            if (
                identifier.strip().lower() in retrieved_ids
                or filename.strip().lower() in retrieved_names
            ):
                return 1
        elif item.lower() in retrieved_ids:
            return 1

    return 0


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {INPUT_FILE}"
        )

    with INPUT_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Evaluation CSV has no result rows.")

    scored_rows = []
    metrics = {
        "baseline_required_pass": 0,
        "rag_required_pass": 0,
        "baseline_no_forbidden": 0,
        "rag_no_forbidden": 0,
        "retrieval_hits": 0,
        "retrieval_eligible": 0,
        "rag_refusal_on_unanswerable": 0,
        "unanswerable_count": 0,
    }

    refusal_phrases = [
        "could not find",
        "don't have information",
        "do not have information",
        "not sure",
        "no information",
        "can't find",
        "cannot find",
        "not able to find",
        "couldn't find",
        "pata nahi",
        "jaankari nahi",
        "connect you to a colleague",
        "connect you to a human",
    ]

    for row in rows:
        baseline = row.get("baseline_answer", "")
        rag = row.get("rag_answer", "")

        required = split_requirements(row.get("must_include", ""))
        forbidden = split_requirements(row.get("must_not_include", ""))

        baseline_pass = required_terms_pass(baseline, required)
        rag_pass = required_terms_pass(rag, required)

        baseline_bad = forbidden_terms_found(baseline, forbidden)
        rag_bad = forbidden_terms_found(rag, forbidden)

        gold = row.get("gold_evidence", "").strip()
        retrieval = evidence_hit(
            gold, row.get("retrieved_sources", "[]")
        )

        answerable = row.get("answerable", "").strip().lower()
        is_unanswerable = answerable in {"false", "0", "no"}

        refusal_detected = any(
            phrase in rag.lower() for phrase in refusal_phrases
        )

        if required:
            metrics["baseline_required_pass"] += int(baseline_pass)
            metrics["rag_required_pass"] += int(rag_pass)

        if forbidden:
            metrics["baseline_no_forbidden"] += int(not baseline_bad)
            metrics["rag_no_forbidden"] += int(not rag_bad)

        if gold:
            metrics["retrieval_eligible"] += 1
            metrics["retrieval_hits"] += int(retrieval == 1)

        if is_unanswerable:
            metrics["unanswerable_count"] += 1
            metrics["rag_refusal_on_unanswerable"] += int(
                refusal_detected
            )

        scored_rows.append({
            **row,
            "baseline_required_pass": int(baseline_pass),
            "rag_required_pass": int(rag_pass),
            "baseline_forbidden_terms": " | ".join(baseline_bad),
            "rag_forbidden_terms": " | ".join(rag_bad),
            "retrieval_hit_at_4": retrieval,
            "rag_refusal_detected": int(refusal_detected),
        })

    extra_columns = [
        "baseline_required_pass",
        "rag_required_pass",
        "baseline_forbidden_terms",
        "rag_forbidden_terms",
        "retrieval_hit_at_4",
        "rag_refusal_detected",
    ]
    columns = list(rows[0].keys()) + extra_columns

    with OUTPUT_FILE.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(scored_rows)

    def percentage(numerator, denominator):
        if denominator == 0:
            return "N/A (no applicable questions)"
        return f"{100 * numerator / denominator:.2f}% ({numerator}/{denominator})"

    summary = [
        f"Total evaluated questions: {len(rows)}",
        "",
        "QUALITY CHECKS (keyword-based proxies, not human-verified accuracy)",
        "Baseline required-term pass rate: "
        + percentage(
            metrics["baseline_required_pass"],
            sum(bool(split_requirements(r.get("must_include", ""))) for r in rows),
        ),
        "RAG required-term pass rate: "
        + percentage(
            metrics["rag_required_pass"],
            sum(bool(split_requirements(r.get("must_include", ""))) for r in rows),
        ),
        "Baseline no-forbidden-term rate: "
        + percentage(
            metrics["baseline_no_forbidden"],
            sum(bool(split_requirements(r.get("must_not_include", ""))) for r in rows),
        ),
        "RAG no-forbidden-term rate: "
        + percentage(
            metrics["rag_no_forbidden"],
            sum(bool(split_requirements(r.get("must_not_include", ""))) for r in rows),
        ),
        "Gold-evidence retrieval hit rate: "
        + percentage(
            metrics["retrieval_hits"], metrics["retrieval_eligible"]
        ),
        "RAG refusal phrase rate on unanswerable questions: "
        + percentage(
            metrics["rag_refusal_on_unanswerable"],
            metrics["unanswerable_count"],
        ),
        "",
        "Caution: these are automatic indicators, not final correctness scores.",
        "Manually inspect answers, policy precedence, source relevance, and language.",
        "",
        f"Scored CSV: {OUTPUT_FILE}",
    ]

    SUMMARY_FILE.write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))
    print(f"\nSaved scored rows to: {OUTPUT_FILE}")
    print(f"Saved summary to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()