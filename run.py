
from src.rag_agent import respond_with_sources


def main():
    print("=" * 55)
    print("ZentroMart AI Customer Support — RAG")
    print("Type 'exit' to stop.")
    print("=" * 55)

    history = []

    while True:
        try:
            user_message = input("\nYou: ").strip()

            if user_message.lower() in {"exit", "quit"}:
                print("Assistant: Thank you for contacting ZentroMart!")
                break

            if not user_message:
                continue

            answer, sources = respond_with_sources(
                user_message, history
            )

            print(f"\nAssistant: {answer}")
            print("\nRetrieved sources:")

            for source in sources:
                print(
                    f"- {source.get('source')} | "
                    f"Section: {source.get('section_id')} | "
                    f"Score: {source.get('score')}"
                )

            history.append({
                "role": "user",
                "content": user_message
            })
            history.append({
                "role": "assistant",
                "content": answer
            })

        except KeyboardInterrupt:
            print("\nChat ended.")
            break

        except Exception as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    main()
