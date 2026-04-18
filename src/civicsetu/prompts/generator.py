GENERATOR_PROMPT = """{conversation_history_block}Answer the following question about RERA rules.

Your answer must:
1. Open with a plain-English summary of what the rule means in practice (1-3 sentences, no jargon)
2. Explain the key points as a short bulleted list — focus on what it means for the person asking, using only information from the provided context
3. Note any connections to other rules, contradictions between jurisdictions, or important exceptions
4. Close with section references anchoring each point (e.g. "Under Section 18...")

Do NOT open with "According to Section X..." - explain first, cite second.
Do NOT paste raw clause text - paraphrase and explain.
Use ONLY the provided context.

Question: {query}

Context:
{context}

Respond with JSON:
{{
  "answer": "<markdown-formatted answer>",
  "confidence_score": <0.0 to 1.0 based on how well context supports the answer>,
  "cited_chunks": [<1-based indices of ONLY the context items you actually used in your answer>],
  "amendment_notice": null,
  "conflict_warnings": []
}}

Rules:
- Only use information present in the context above — never use external knowledge or training data
- If the context does not contain the answer, set confidence_score below 0.3 and cited_chunks: [], and state "The provided context does not contain sufficient information to answer this question."
- Never invent section numbers, legal provisions, or specific figures not present in the context
- Format the answer field as GitHub-flavored Markdown
- Prefer concise markdown paragraphs and bullet lists; use tables only when comparing provisions or jurisdictions
- Do not wrap the answer in markdown code fences
- Reference specific section numbers in your answer ONLY if those sections appear in the context
- In cited_chunks, list ONLY the [N] indices you directly drew from — not every item in context
- If you used [1] and [3] but not [2], [4], [5] → cited_chunks: [1, 3]
- If cited_chunks is empty, confidence_score must be below 0.3
- conflict_warnings: list any direct contradictions found between provisions across jurisdictions
- amendment_notice: note if any provision appears superseded or amended, otherwise null
- If context is sparse or does not directly answer the question, do NOT construct a legal conclusion by reasoning from general principles — instead say "Based on the available context: [what you CAN say]" and explicitly note "The context does not contain sufficient information to determine [missing element]"
- For conflict_detection queries: only assert a conflict exists if BOTH conflicting provisions appear in the context. If only one side is present, describe what you found and note the missing side: "The context contains [jurisdiction X's position] but does not include [jurisdiction Y's position] to confirm or deny a conflict"
"""
