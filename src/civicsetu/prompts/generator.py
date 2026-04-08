GENERATOR_PROMPT = """{conversation_history_block}Answer the following question about RERA rules.

Your answer must:
1. Open with a plain-English summary of what the rule means in practice (1-3 sentences, no jargon)
2. Use an analogy or real-world example where it helps (e.g. "Think of it like a builder's warranty")
3. Explain the key points as a short bulleted list - focus on what it means for the person asking
4. Note any connections to other rules, contradictions between jurisdictions, or important exceptions
5. Close with section references anchoring each point (e.g. "Under Section 18...")

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
- Only use information present in the context above
- Format the answer field as GitHub-flavored Markdown
- Prefer concise markdown paragraphs and bullet lists; use tables only when comparing provisions or jurisdictions
- Do not wrap the answer in markdown code fences
- Reference specific section numbers in your answer (e.g. "Section 18", "Rule 3")
- In cited_chunks, list ONLY the [N] indices you directly drew from — not every item in context
- If you used [1] and [3] but not [2], [4], [5] → cited_chunks: [1, 3]
- If context is insufficient, set confidence_score below 0.5 and cited_chunks: []
- Never invent section numbers or legal provisions
- conflict_warnings: list any direct contradictions found between provisions across jurisdictions
- amendment_notice: note if any provision appears superseded or amended, otherwise null
"""
