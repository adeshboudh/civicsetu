VALIDATOR_PROMPT = """Check if this legal answer is grounded in the provided context.

Answer: {answer}

Context: {context}

Respond with JSON:
{{
  "hallucination_detected": true | false,
  "confidence_score": <revised 0.0 to 1.0>,
  "reason": "<brief explanation if hallucination detected>"
}}

Rules:
- hallucination_detected=true if the answer makes claims not supported by context
- Lower confidence_score if answer is only partially supported
- hallucination_detected=false if all claims trace to the context
"""
