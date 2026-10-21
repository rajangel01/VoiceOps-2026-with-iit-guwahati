
import argparse
import csv
import json
from pathlib import Path

from src.agent import respond as baseline_respond
from src.rag_agent import respond_with_sources

ROOT = Path(__file__).resolve().parent
QUESTIONS_FILE = ROOT / "practice" / "questions.csv"
OUTPUT_FILE = ROOT / "data" / "evaluation_results.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Test only the first N questions; 0 means all questions.",
    )
    args = parser.parse_args()

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Questions file not found: {QUESTIONS_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with QUESTIONS_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        questions = list(csv.DictReader(file))

    if args.limit > 0:
        questions = questions[:args.limit]

    columns = [
        "id",
        "language",
        "question",
        "expected_decision",
        "answerable",
        "reference_answer",
        "gold_evidence",
        "must_include",
        "must_not_include",
        "baseline_answer",
        "rag_answer",
        "retrieved_sources",
        "baseline_error",
        "rag_error",
    ]

    # Write each result immediately so progress is preserved.
    with OUTPUT_FILE.open(
        "w", encoding="utf-8-sig", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()

        for number, row in enumerate(questions, start=1):
            question = row["question"]
            print(
                f"\n[{number}/{len(questions)}] "
                f"{row['id']}: {question}"
            )

            baseline_answer = ""
            rag_answer = ""
            sources_json = "[]"
            baseline_error = ""
            rag_error = ""

            try:
                baseline_answer = baseline_respond(question)
            except Exception as exc:
                baseline_error = str(exc)

            try:
                answer, sources = respond_with_sources(question)
                rag_answer = answer
                sources_json = json.dumps(
                    [
                        {
                            "source": source.get("source"),
                            "section_id": source.get("section_id"),
                            "score": source.get("score"),
                        }
                        for source in sources
                    ],
                    ensure_ascii=False,
                )
            except Exception as exc:
                rag_error = str(exc)

            writer.writerow({
                "id": row.get("id", ""),
                "language": row.get("language", ""),
                "question": question,
                "expected_decision": row.get("expected_decision", ""),
                "answerable": row.get("answerable", ""),
                "reference_answer": row.get("reference_answer", ""),
                "gold_evidence": row.get("gold_evidence", ""),
                "must_include": row.get("must_include", ""),
                "must_not_include": row.get("must_not_include", ""),
                "baseline_answer": baseline_answer,
                "rag_answer": rag_answer,
                "retrieved_sources": sources_json,
                "baseline_error": baseline_error,
                "rag_error": rag_error,
            })
            output.flush()

            print("Baseline:", baseline_answer[:250])
            print("RAG:", rag_answer[:250])

    print(f"\nEvaluation finished. Results saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
