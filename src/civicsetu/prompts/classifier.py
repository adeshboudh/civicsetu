CLASSIFIER_PROMPT = """Classify this legal query and rewrite it for better document retrieval.

Query: {query}

Respond with JSON:
{{
  "query_type": "fact_lookup" | "cross_reference" | "temporal" | "penalty_lookup" | "conflict_detection",
  "rewritten_query": "<cleaner, expanded version of the query for semantic search>"
}}

Classification rules (apply in order — first match wins):
- cross_reference: query mentions a specific section number (e.g. "Section 18", "section 4", "s. 11"),
                   OR asks how sections relate, reference, or interact with each other
- penalty_lookup:  asks about fines, punishments, jail, imprisonment, consequences of violation
- temporal:        asks about amendments, changes, history, "before/after", "as amended"
- conflict_detection: asks if two laws or provisions contradict each other
- fact_lookup:     all other queries — asking what a law generally says without citing a section

Examples:
- "What does Section 18 say?" → cross_reference
- "What are the duties of a promoter?" → fact_lookup
- "What is the penalty for not registering?" → penalty_lookup
- "How does Section 11 relate to Section 4?" → cross_reference
- "Was RERA amended in 2020?" → temporal
"""
