import sys
from retriever import retrieve
from llm import answer


def main():
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        raise SystemExit(1)

    question = " ".join(sys.argv[1:])

    contexts = retrieve(question, k=4)

    print("\n--- Retrieved context ---\n")
    for i, c in enumerate(contexts, start=1):
        print(f"[{i}] {c}\n")

    print("\n--- Answer ---\n")
    print(answer(question, contexts))


if __name__ == "__main__":
    main()
