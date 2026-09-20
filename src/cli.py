"""
Interactive multi-turn chat with the support assistant.

Usage:
    export GOOGLE_API_KEY=your_key   # optional -- omit to run with MockLLM
    python src/cli.py
"""
from assistant import SupportAssistant


def main():
    assistant = SupportAssistant()
    print("LearnForge Support Assistant (type 'exit' to quit)\n")
    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            break

        turn = assistant.ask(user_message)
        print(f"\nAssistant: {turn.answer}")
        if turn.retrieved_ids:
            print(f"  (retrieved: {', '.join(turn.retrieved_ids)}, top_score={turn.top_score:.3f})")
        if turn.escalate:
            print(f"  [ESCALATE -> human agent | reasons: {', '.join(turn.escalation_reasons)}]")
        print()


if __name__ == "__main__":
    main()
