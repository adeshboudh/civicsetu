VALIDATOR_PROMPT = """You are a legal answer quality scorer.

Query Type: {query_type}
Answer: {answer}
Context: {context}

Score how well the answer is supported by the context.
Respond with JSON only — no other text:
{{
  "confidence_score": <0.0 to 1.0>,
  "reason": "<one sentence>"
}}

Scoring guide:
- 0.9–1.0 : Answer is fully grounded; all claims trace to the context
- 0.6–0.8 : Answer is mostly grounded; minor gaps or paraphrasing
- 0.3–0.5 : Answer is partially grounded or says "context does not contain X"
- 0.0–0.2 : Answer contradicts context or invents specific facts not present

For cross_reference / penalty_lookup / temporal queries:
  The answer synthesises across multiple chunks — low citation count is normal.
  Score 0.7+ if the answer only names sections/jurisdictions present in the context.
  Score 0.2 only if the answer names sections or rules that do NOT appear in the context.
"""
