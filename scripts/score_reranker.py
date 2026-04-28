"""
Quick diagnostic: score specific chunks against a query using current reranker.
Usage: uv run python scripts/score_reranker.py
"""
from flashrank import Ranker, RerankRequest

MODELS = ["rank-T5-flan", "ms-marco-MiniLM-L-12-v2"]
CACHE = ".cache/flashrank"

QUERY = "What does Section 18 say about refund obligations when a promoter fails to give possession?"

# Chunks to score — Section 18 relevant + Karnataka-13 irrelevant
PASSAGES = [
    {
        "id": 0,
        "label": "Section 18 - return of amount and compensation",
        "text": (
            "Section 18. Return of amount and compensation. "
            "(1) If the promoter fails to complete or is unable to give possession of an apartment, "
            "plot or building,— (a) in accordance with the terms of the agreement for sale or, "
            "as the case may be, duly completed by the date specified therein; or (b) due to "
            "discontinuance of his business as a developer on account of suspension or revocation "
            "of the registration under this Act or for any other reason, the promoter shall be "
            "liable on demand to the allottees, in case the allottee wishes to withdraw from the "
            "project, without prejudice to any other remedy available, to return the amount "
            "received by him in respect of that apartment, plot, building, as the case may be, "
            "with interest at such rate as may be prescribed in this behalf including compensation "
            "in the manner as provided under this Act."
        ),
    },
    {
        "id": 1,
        "label": "Section 18(2) - interest on delayed possession",
        "text": (
            "Section 18(2) Where an allottee does not intend to withdraw from the project, "
            "he shall be paid, by the promoter, interest for every month of delay, till the "
            "handing over of the possession, at such rate as may be prescribed."
        ),
    },
    {
        "id": 2,
        "label": "Karnataka Rule 13 - Maintenance of books of accounts (IRRELEVANT)",
        "text": (
            "Rule 13. Maintenance and preservation of books of accounts, records and documents. "
            "Every promoter shall maintain proper books of accounts, records and documents in "
            "relation to each registered real estate project, separately, and such books of "
            "accounts, records and documents shall be preserved for a period of not less than "
            "five years after the completion of the project."
        ),
    },
    {
        "id": 3,
        "label": "Section 19 - rights of allottees (somewhat related)",
        "text": (
            "Section 19. Rights and duties of allottees. (1) The allottee shall be entitled "
            "to obtain the information relating to sanctioned plans, layout plans along with "
            "the specifications, approved by the competent authority and such other information "
            "as provided under this Act or the rules and regulations made thereunder. "
            "(4) The allottee shall be entitled to claim the possession of apartment, plot or "
            "building, as the case may be, and the promoter shall be liable to pay interest "
            "for any delay in handing over such possession at the rate specified under the Act."
        ),
    },
]


def score_model(model_name: str):
    print(f"\n{'='*80}")
    print(f"Model: {model_name}")
    print(f"{'='*80}")
    ranker = Ranker(model_name=model_name, cache_dir=CACHE)

    passages = [{"id": p["id"], "text": p["text"]} for p in PASSAGES]
    request = RerankRequest(query=QUERY, passages=passages)
    results = ranker.rerank(request)

    id_to_label = {p["id"]: p["label"] for p in PASSAGES}

    print(f"{'Score':>8}  {'ID':>3}  Label")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        score = round(float(r["score"]), 4)
        label = id_to_label[r["id"]]
        marker = " ← IRRELEVANT" if r["id"] == 2 else ""
        print(f"{score:>8.4f}  {r['id']:>3}  {label}{marker}")

    karnataka_score = next(round(float(r["score"]), 4) for r in results if r["id"] == 2)
    sec18_score = next(round(float(r["score"]), 4) for r in results if r["id"] == 0)
    print(f"\nKarnataka-13: {karnataka_score}  |  Section 18: {sec18_score}")
    if karnataka_score > sec18_score:
        print("✗ FAIL — irrelevant chunk ranked ABOVE Section 18")
    else:
        gap = round(sec18_score - karnataka_score, 4)
        print(f"✓ PASS — Section 18 ranks above Karnataka-13 (gap={gap})")


def main():
    print(f"Query: {QUERY}")
    for model in MODELS:
        score_model(model)


if __name__ == "__main__":
    main()
