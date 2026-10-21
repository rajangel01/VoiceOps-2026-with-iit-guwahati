from src.rag import search_knowledge_base

tests = [
    "SKU ELE-0012 pe ZentroMart Assured hai kya? Claim kaise karein?",
    "credit card wala refund 17 October 2026 ko shuru hua, kab tak aa jaana chahiye?",
    "I opened the Neelam Eau de Parfum 50 ml and tried it once. Can I still return it?",
]

for question in tests:
    print("\n" + "=" * 90)
    print("QUESTION:", question)
    print("=" * 90)

    results = search_knowledge_base(question, top_k=4)

    for rank, item in enumerate(results, start=1):
        print(f"\n--- Rank {rank} ---")
        print("Source:", item.get("source"))
        print("Section:", item.get("section_id"))
        print("Status:", item.get("status"))
        print("Effective from:", item.get("effective_from"))
        print("Similarity:", item.get("score"))
        print("Text:\n", item.get("text", ""))