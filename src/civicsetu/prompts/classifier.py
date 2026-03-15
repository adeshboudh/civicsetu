CLASSIFIER_PROMPT = """Classify this legal query and rewrite it for better document retrieval.

Query: {query}

Respond with JSON:
{{
  "query_type": "fact_lookup" | "cross_reference" | "temporal" | "penalty_lookup" | "conflict_detection",
  "rewritten_query": "<cleaner, expanded version of the query>"
}}

Rules:
- fact_lookup: asking what a law says
- cross_reference: asking how sections relate to each other
- temporal: asking about amendments, changes over time
- penalty_lookup: asking about fines, punishments, consequences
- conflict_detection: asking if two laws contradict
"""
