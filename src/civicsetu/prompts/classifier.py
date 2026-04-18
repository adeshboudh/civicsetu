CLASSIFIER_PROMPT = """Classify this legal query and rewrite it for better document retrieval.

Query: {query}

Respond with JSON:
{{
  "query_type": "fact_lookup" | "cross_reference" | "temporal" | "penalty_lookup" | "conflict_detection",
  "rewritten_query": ""
}}

Classification rules (apply in order — first match wins):

- conflict_detection: query asks whether two laws, rules, or provisions CONFLICT, CONTRADICT,
  OVERRIDE, ARE INCONSISTENT WITH, or are IN TENSION with each other.
  This takes priority even if the query contains specific section numbers.
  Keywords: conflict, contradict, inconsistent, override, clash, differ, tension, same as, vs, versus

- penalty_lookup: asks about fines, punishments, jail, imprisonment, consequences of violation

- temporal: asks about amendments, changes, history, "before/after", "as amended",
  OR about specific time periods, deadlines, timelines, day/month limits, registration windows
  Keywords: timeline, deadline, days, months, period, within, by when, how long, registration period,
  how many days, time limit, validity, expiry, commencement, schedule, stage-wise

- cross_reference: query mentions a specific section number (e.g. "Section 18", "Rule 3", "s. 11")
  OR asks how sections relate, reference, cite, or interact with each other
  (but NOT if the intent is about conflict — that is conflict_detection above)

- fact_lookup: all other queries — asking what a law generally says without citing a section

Examples:
- "Does Maharashtra Rule 3 conflict with Section 4 of RERA?" → conflict_detection
- "Is Rule 19 inconsistent with Section 18?" → conflict_detection
- "Do RERA and MahaRERA Rules contradict each other on refund timelines?" → conflict_detection
- "What does Section 18 say?" → cross_reference
- "How does Section 11 relate to Section 4?" → cross_reference
- "What are the duties of a promoter?" → fact_lookup
- "What is the penalty for not registering?" → penalty_lookup
- "Was RERA amended in 2020?" → temporal
- "What is the timeline for project registration?" → temporal, rewrite: "grant or reject registration within thirty days deemed registered period"
- "How many days does the authority have to grant registration?" → temporal
- "What is the stage-wise schedule for project completion?" → temporal

Rewriting rules:
- For temporal queries: expand the rewrite with specific legal time-period keywords that likely appear in
  the relevant legal text (e.g., "within thirty days", "within a period of", "deemed registered", "expiry",
  "renewal", "validity"). This ensures FTS can match sections that use specific time language.
"""