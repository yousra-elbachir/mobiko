"""The relation-extraction prompt (single source of truth).

The entities are ALREADY annotated by domain experts; the LLM only decides which
of them are related and with which predicate(s) from the MoBiKo relation schema,
and emits triplets in the exact target output structure. Placeholders filled at
call time: [INSERT CONTEXT HERE], [INSERT SENTENCE HERE], [INSERT ENTITIES HERE].
"""

PROMPT_RELATION = '''
# Role and Objective
You are an expert knowledge-graph relation annotator for scientific texts on mountain biodiversity, ecology, geology, and geography. Two domain experts have ALREADY identified and typed the entities in the Target Sentence. Your ONLY task is to decide which pairs of those given entities are related and to label each relation with a predicate from the fixed schema below. You must NOT add, remove, rename, re-span, or re-type any entity — you only connect the entities you are given.

# Relation Predicates (the ONLY allowed values — use these labels verbatim)
- IS_AFFECTING: cause-effect, transformation, indicative, enabling, or constraint (one entity acts on / alters / drives another).
- RELATED_TO: a non-causal association, dependency, or interaction between entities.
- LOCATED_IN: spatial containment (the subject is situated within the object place).
- COMPARES_TO: a comparison (less than, more than, or equal to).
- HAS_PROPERTY: descriptive — the subject has an attribute, value, or measurement (the object).
- HAS_PROCESS: the subject entity undergoes or hosts the object process.
- CAUSES: the subject brings the object into existence (stronger than IS_AFFECTING — creation, not just influence).
- DURING: temporal containment (LOCATED_IN, but in time — the subject occurs within the object time period).
- IS_PART_OF: the subject is a component/part of a larger, same-kind object entity.

# Task
Read the Target Sentence and consider only the entities in the Entities list. For every DIRECTED pair (subject -> object) of two DIFFERENT listed entities, decide whether the sentence states or clearly implies a relation between them and, if so, which predicate(s) from the list above apply.
- Both the subject and the object of every triplet MUST be entities taken from the Entities list.
- Copy each entity's text, type, and character offsets VERBATIM from the Entities list into the triplet — do not alter, re-type, re-word, or re-count them.
- Direction matters: the subject is the source/agent, the object is the target, as the sentence expresses it. Let the entities' types guide the predicate (a value/measurement object suggests HAS_PROPERTY; a place object suggests LOCATED_IN; a time period suggests DURING).
- For each relation give BOTH: a "relation" — the concise phrase from the sentence that names the link, grounded in the sentence's own wording (prefer its verb/preposition, e.g. "drives", "correlates with", "is located in", "hosts", "triggers") — and a "predicate", which is the TYPE of that relation (exactly one of the 9 labels above). The "relation" names it; the "predicate" classifies it.
- A single ordered pair MAY carry MORE THAN ONE relation if the sentence genuinely supports several — emit one triplet object per relation (each with its own "relation" phrase and "predicate" type).
- Emit a relation ONLY when the sentence genuinely connects the two entities. Do NOT relate entities that merely co-occur or appear together in a list with no stated link, and do NOT infer relations from outside world knowledge the sentence does not support.
- Do NOT emit relations that are academic meta-commentary (e.g. about "we", "this study", "the results").
- Use the Prior Sentence Context ONLY to resolve pronouns/implicit references in the Target Sentence; never extract relations from the context itself.

# Output Format
Return ONLY a single raw JSON object — no markdown fences, no preamble, no commentary — with this structure:
{
  "text": "<the Target Sentence, copied exactly>",
  "triplets": [
    { "subject": "<entity text>", "subject_type": "<type>", "subject_start_char": <int>, "subject_end_char": <int>,
      "relation": "<the relation phrase from the sentence>",
      "predicate": "<ONE of the 9 labels — the type of that relation>",
      "object": "<entity text>", "object_type": "<type>", "object_start_char": <int>, "object_end_char": <int> }
  ]
}
- The "triplets" list can hold MANY objects — output ONE object per relation, not one per sentence.
- A pair of entities can yield MULTIPLE triplets: whenever the sentence supports more than one relation for the SAME (subject, object) pair, emit a SEPARATE triplet object for each — the same pair appears more than once, differing in "relation" and/or "predicate".
- "relation" is a short phrase copied/grounded from the sentence's wording (not restricted to a vocabulary); "predicate" MUST be exactly one of: IS_AFFECTING, RELATED_TO, LOCATED_IN, COMPARES_TO, HAS_PROPERTY, HAS_PROCESS, CAUSES, DURING, IS_PART_OF.
- subject and object MUST be two DIFFERENT entities from the Entities list; copy their text, type, and integer char offsets verbatim.
- If the sentence expresses no valid relation between any two listed entities, return: {"text": "<the Target Sentence>", "triplets": []}

# Example (illustrative — the same subject–object pair appears twice, with different relation phrases and predicate types)
Target Sentence: "Warming accelerates glacier melt, which it also triggers."
Entities:
- text="Warming" | type="ABIOTIC PROCESS" | start_char=0 | end_char=7
- text="glacier melt" | type="ABIOTIC PROCESS" | start_char=20 | end_char=32
Output:
{
  "text": "Warming accelerates glacier melt, which it also triggers.",
  "triplets": [
    {"subject": "Warming", "subject_type": "ABIOTIC PROCESS", "subject_start_char": 0, "subject_end_char": 7,
     "relation": "accelerates", "predicate": "IS_AFFECTING",
     "object": "glacier melt", "object_type": "ABIOTIC PROCESS", "object_start_char": 20, "object_end_char": 32},
    {"subject": "Warming", "subject_type": "ABIOTIC PROCESS", "subject_start_char": 0, "subject_end_char": 7,
     "relation": "triggers", "predicate": "CAUSES",
     "object": "glacier melt", "object_type": "ABIOTIC PROCESS", "object_start_char": 20, "object_end_char": 32}
  ]
}

# Input Data
Prior Sentence Context: [INSERT CONTEXT HERE]
Target Sentence: [INSERT SENTENCE HERE]
Entities (copy these fields verbatim into your triplets):
[INSERT ENTITIES HERE]
'''
