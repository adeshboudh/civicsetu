CLASSIFIER_PROMPT = """Classify this legal query and rewrite it for better document retrieval.

Query: {query}

Respond with JSON:
{{
"query_type": "fact_lookup" | "cross_reference" | "temporal" | "penalty_lookup" | "conflict_detection",
"rewritten_query": ""
}}

Classification rules (apply in order — first match wins):

- conflict_detection: query asks whether two laws, rules, or provisions CONFLICT, CONTRADICT,
  OVERRIDE, ARE INCONSISTENT WITH, or are IN TENSION with each other — OR compares how two
  jurisdictions/laws handle the same topic differently.
  This takes priority even if the query contains specific section numbers.
  Keywords: conflict, contradict, inconsistent, override, clash, differ, differently, tension,
  same as, vs, versus, compared to, compare, how X handle Y differently, how X vs Y

- penalty_lookup: asks about fines, punishments, jail, imprisonment, consequences of violation

- temporal: asks about amendments, changes, history, "before/after", "as amended",
  OR about specific time periods, deadlines, timelines, day/month limits, registration windows
  Keywords: timeline, deadline, days, months, period, within, by when, how long, registration period,
  how many days, time limit, validity, expiry, commencement, schedule, stage-wise

- cross_reference: query mentions a specific section number (e.g. "Section 18", "Rule 3", "s. 11")
  OR asks how sections relate, reference, cite, or interact with each other
  (but NOT if the intent is about conflict — that is conflict_detection above)

- fact_lookup: all other queries — asking what a law generally says without citing a section

Rewriting rules — APPLY THE FIRST MATCHING RULE BELOW, then write rewritten_query:

RULE 1 — TEMPORAL queries about PROJECT REGISTRATION timelines ONLY:
Apply this rule ONLY when the query explicitly asks about:
- How long the Authority has to grant or reject a project registration application
- The "deemed registered" provision for projects
- The deadline for ongoing projects to apply for registration at commencement
NOT for: complaint filing deadlines, agent registration/renewal periods, association
formation timelines, refund timelines, or any other non-project-registration timeline.

When applicable, the rewritten_query MUST contain:
"grant or reject registration within thirty days deemed registered section 5
period of thirty days application receipt authority"

Example: "What is the timeline for project registration under central RERA?"
-> "grant or reject registration within thirty days deemed registered section 5
   period of thirty days application receipt authority"

For all OTHER temporal queries — skip to RULE 7 and expand with domain-specific time vocabulary.
IMPORTANT: Do NOT include section numbers or rule numbers in these rewritten queries —
section numbers cause incorrect state rules to be pinned during retrieval.
Use these as templates (do NOT copy verbatim — adapt to the query):
- complaint filing deadline → "complaint filing period limitation cause of action
  allottee RERA Authority prescribed format fee three years"
- refund / compensation timeline → "promoter fails complete give possession date agreement
  allottee return amount refund interest compensation"
- project completion / how long to complete registered project → "registered project completion
  period years time schedule promoter undertakes deliver force majeure extension handover declared"
- agent registration or renewal period → "real estate agent registration renewal validity
  period application fee expiry continuity sixty days"

RULE 2 — FACT_LOOKUP queries about allottee / homebuyer / buyer rights:
The rewritten_query MUST include:
"section 19 rights duties allottee entitled claim possession documents plans
interest compensation obtain information stage-wise schedule"
Example: "What rights does an allottee have under RERA?"
-> "section 19 rights duties allottee entitled claim possession documents plans
   interest compensation obtain information stage-wise schedule"

RULE 3D — FACT_LOOKUP queries about what documents a promoter must SUBMIT with a project
registration application (disclosures required at the time of applying for registration):
Triggers: query asks what documents, certificates, or declarations must be submitted with
a registration application. NOT for ongoing post-registration duties (use RULE 3 for those).
The rewritten_query MUST NOT include any section number. Use:
"promoter application registration information furnished documents submit commencement
certificate sanctioned plan agreement sale land title encumbrance approvals ongoing disclosure"
Example: "What documents must a promoter submit for project registration in Maharashtra?"
-> "promoter application registration information furnished documents submit commencement
   certificate sanctioned plan agreement sale land title encumbrance approvals ongoing disclosure"

RULE 3 — FACT_LOOKUP queries about promoter obligations / builder duties:
NOT for: queries about what documents to submit with a registration application (use RULE 3D).
The rewritten_query MUST include:
"section 11 obligations duties promoter outgoings insurance structural defect
transfer consent allottees functions responsibilities"
Example: "What are the obligations of a promoter under RERA?"
-> "section 11 obligations duties promoter outgoings insurance structural defect
   transfer consent allottees functions responsibilities"

RULE 3A — FACT_LOOKUP queries about revocation of project registration:
Triggers: query asks about grounds, criteria, or conditions for revoking / cancelling
a project registration (NOT agent registration revocation).
The rewritten_query MUST include these terms but MUST NOT include any section number:
"revocation registration grounds criteria real estate project suomotu complaint satisfied
wilful default false statement unfair practice non-compliance failure pay penalty dues
promoter cancellation circumstances authority"
Example: "What are the grounds for revocation of project registration in Karnataka?"
-> "revocation registration grounds criteria real estate project suomotu complaint satisfied
   wilful default false statement unfair practice non-compliance failure pay penalty dues
   promoter cancellation circumstances authority"

RULE 3C — FACT_LOOKUP queries about the purpose, mandate, or functions of the RERA Authority:
Triggers: query asks what RERA Authority does, its role, objectives, or powers.
The rewritten_query MUST include:
"section 20 functions Real Estate Regulatory Authority regulate promote real estate sector
consumer protection adjudicating mechanism dispute redressal transparent efficient sale allottee"
Example: "What is the purpose of the Real Estate Regulatory Authority under RERA?"
-> "section 20 functions Real Estate Regulatory Authority regulate promote real estate sector
   consumer protection adjudicating mechanism dispute redressal transparent efficient sale allottee"

RULE 3B — FACT_LOOKUP or CROSS_REFERENCE queries about project account / separate account /
fund maintenance / bank account / 70% deposit:
Triggers: query asks which rule or provision governs the project separate bank account,
maintenance of funds, withdrawal conditions, or audit requirements.
The rewritten_query MUST include these terms but MUST NOT include any section number:
"separate designated bank account seventy percent amounts allottee deposited
audit requirements withdrawal conditions project maintain state rules implement"
Example: "Which Karnataka RERA rule implements the central act provisions on project account maintenance?"
-> "separate designated bank account seventy percent amounts allottee deposited
   audit requirements withdrawal conditions project maintain state rules implement"

RULE 4 — CROSS_REFERENCE queries:
Include the section ID and its common title vocabulary only.
Do NOT add other section numbers — graph traversal finds related sections automatically.
Example: "What does Section 18 say about refund obligations?"
-> "section 18 return amount refund compensation possession promoter fails interest allottee withdraw"

RULE 5 — CONFLICT_DETECTION queries:
Decompose into BOTH sides of the comparison. Do NOT include bare section numbers — use
descriptive vocabulary only so state rules with matching numbers are not incorrectly pinned.
For state-vs-central comparisons: include central Act vocabulary AND named state vocabulary.
For state-vs-state comparisons: MUST include BOTH state names AND central Act vocabulary.
Example: "How do state RERA rules differ from the central Act on project registration?"
-> "central RERA Act prior registration baseline requirements
   state rules maharashtra karnataka additional documents fee structure local formats"
Example: "How does Karnataka handle extension of project registration compared to the central RERA Act?"
-> "extension registration project period force majeure central RERA Act one year reasonable circumstances
   karnataka state rules extension fee application procedure conditions comparison"
Example: "How does Karnataka handle extension of project registration compared to Maharashtra?"
-> "extension registration project period force majeure central RERA Act
   karnataka state rules maharashtra state rules extension fee application procedure comparison"
Example: "How does Tamil Nadu specify the interest rate on refunds differently from the central RERA Act?"
-> "interest rate refund allottee SBI MCLR marginal cost lending rate two percent
   central RERA Act prescribed government Tamil Nadu Rule comparison return amount"

RULE 6 — PENALTY_LOOKUP queries:
Include both the penalty clause AND the trigger condition.
Example: "What is the penalty for not registering a project?"
-> "penalty promoter advertise market book sell real estate project without prior registration
   ten percent estimated cost imprisonment contravention"
Example: "What is the penalty for a real estate agent operating without registration?"
-> "penalty real estate agent unregistered section 62 ten thousand rupees per day
   five percent cost contravention facilitates sale without registration"

RULE 7 — ALL other queries:
Expand with relevant legal section vocabulary from the RERA Act domain.
Never copy the query verbatim as the rewritten_query.
"""