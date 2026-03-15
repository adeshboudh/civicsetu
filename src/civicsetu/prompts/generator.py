GENERATOR_PROMPT = """Answer the following legal question using ONLY the provided context.

Question: {query}

Context:
{context}

Respond with JSON:
{{
  "answer": "<plain English answer citing specific sections>",
  "confidence_score": <0.0 to 1.0 based on how well context supports the answer>,
  "amendment_notice": "<if context mentions any amendments, note them here, else null>",
  "conflict_warnings": ["<any contradictions noticed between sections>"]
}}

Rules:
- Only use information present in the context above
- Reference specific section numbers in your answer
- If context is insufficient, set confidence_score below 0.5
- Never invent section numbers or legal provisions
"""
