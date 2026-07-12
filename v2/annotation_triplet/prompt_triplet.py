"""The JOINT triplet-extraction prompt (single source of truth).

Unlike annotation_relation (where domain experts had ALREADY annotated the entities and the LLM
only chose predicates), here the LLM performs JOINT extraction: it must find the
entities in the sentence, type them with the 16-label MoBiKo schema, decide which
pairs are related, name the relation, and classify it with one of the 9 predicates.

Character offsets are NOT requested from the model — they are unreliable for spans
the model itself proposes. The runner recomputes subject_start_char/end_char and
object_start_char/end_char by locating each entity's exact text in the sentence, so
the on-disk record matches the target structure in v3_annotation_relation/
prompt_relation.py. For that to work, the model MUST copy each entity's text as an
EXACT substring of the Target Sentence.

Placeholders filled at call time: [INSERT CONTEXT HERE], [INSERT SENTENCE HERE].
"""

PROMPT_TRIPLET = '''
# Role and Objective
You are a team of expert knowledge-graph annotators — ecologists, geographers, geologists, and bio-informaticians — building a knowledge graph of mountain biodiversity (species and ecosystems, their states and trends, the drivers of change, and their relations in space and time). Your task is JOINT triplet extraction from one scientific sentence: find the entities, type each one with the controlled schema, and connect related pairs as directed triplets (subject -> relation -> object). Extract ONLY what the sentence itself states or clearly implies — never add world knowledge.

# What a triplet is
A triplet links two DIFFERENT entities found in the Target Sentence:
- subject: the source/agent entity (its text + one type from the schema).
- object: the target entity (its text + one type from the schema).
- relation: a short phrase grounded in the sentence's own wording that names the link (prefer its verb/preposition, e.g. "drives", "correlates with", "is located in", "hosts", "triggers", "increases with").
- predicate: the TYPE of that relation — EXACTLY one of the 9 labels below. The "relation" names the link; the "predicate" classifies it.

# Controlled Entity-Type Vocabulary (closed set of 16 — copy a label VERBATIM)
- ABIOTIC ENTITY — a non-living natural physical object or component of the environment (e.g. rock, glacier, soil, river, moraine, snowpack). Physical landforms/geological features/environmental components are ABIOTIC ENTITY regardless of whether they can be mapped.
- ABIOTIC PROCESS — a physical or chemical change in the environment (e.g. erosion, landslide, glacial melt, climate change, weathering, sedimentation).
- ABIOTIC PROPERTY — a measurable physical/chemical attribute (e.g. temperature, snow cover, salinity, radiation, soil pH, cold climate).
- ANTHROPOGENIC ENTITY — a human-made physical object, infrastructure, or modification (e.g. road, dam, fence, emergence trap, hydropower infrastructure).
- ANTHROPOGENIC PROCESS — an activity/action performed by humans that affects socioecological systems (e.g. grazing, mining, logging, land-use change, tourism, conservation planning). NOT the researchers' own analysis/sampling/measurement.
- ANTHROPOGENIC PROPERTY — a measurable social, economic, or governance attribute (e.g. human population density, tourism pressure, land tenure type, literacy rate, human wellbeing).
- BIOTIC ENTITY — a living organism or taxon, OR an assemblage of organisms functioning as a unit (e.g. snow leopard, Vulpes vulpes, plants, mammals, species, pollinator community, alpine grassland, ecosystem, population, leaf, carcass).
- BIOTIC PROCESS — a biological action/interaction by or involving organisms (e.g. pollination, predation, migration, flowering, decomposition, population decrease, succession).
- BIOTIC PROPERTY — a trait/attribute/measurable characteristic of organisms or biotic assemblages (e.g. biomass, species richness, biodiversity, genetic diversity, body size, canopy height, population growth rate).
- SPATIAL ENTITY — a NAMED geographic region, administrative boundary, or operational spatial extent used as a reference frame or mapping unit (e.g. Tatra National Park, the Alps, Poland, protected area, study region, watershed, transect, grid cell).
- SPATIAL PROPERTY — a geometric or positional descriptor (e.g. elevation, slope, aspect, area, distance to road, altitudinal gradient, spatial extent).
- TEMPORAL ENTITY — a named/identifiable time period or phase (e.g. Holocene, Pleistocene, growing season, winter, dry season).
- TEMPORAL PROPERTY — a temporal descriptor or metric (e.g. annual, decadal, duration, phenological timing, interannual variability, long-term).
- QUANTITATIVE PROPERTY — a NAMED measurable ecological/biological/environmental attribute that can be quantified (e.g. abundance, rate, density expressed as a measure). Do NOT label bare numbers or quantifiers (see rules below).
- QUALITATIVE PROPERTY — a non-numeric attribute describing a quality/kind/state, not mappable to a biotic/abiotic/spatial/human property (e.g. antibacterial, parasitic, big/small, unstable/stable).
- CONCEPT — an abstract or theoretical construct used in analysis or discourse (e.g. climate, conservation status, resilience, scenario, vulnerability, trend, ecosystem service, a statistical/methodological framework, an abstract policy/agreement).

Type formatting rules (STRICT): use the label EXACTLY as written above — ALL CAPS, single ASCII space between words, "ANTHROPOGENIC" spelled in full, NO underscores, NO Title Case, NO invented labels. There is no SPATIAL PROCESS, TEMPORAL PROCESS, QUANTITATIVE ENTITY, or CONCEPT PROPERTY. If an entity does not perfectly fit, choose the single closest valid label.

# How to choose the entity SPAN (the text)
- Keep spans MINIMAL: the core noun phrase with NO function words (no articles "the", no prepositions "of", no conjunctions "and"). Example: from "the province of Parinacota" the span is "Parinacota".
- BUT keep together compound terms that express one commonly-used concept: "species richness", "habitat quality", "mountain biodiversity", "population density".
- Unfold AND/OR conjunctions into separate entities that share the partner: "birds and insects declined" -> two subjects "birds" and "insects". "antibacterial and antifungal properties" -> two properties.
- The subject/object text MUST be an EXACT, contiguous substring of the Target Sentence, copied verbatim (same characters, same casing, same word form as it appears). Do NOT paraphrase, lemmatize, pluralize, or merge distant words — the text is used to locate the entity's character position in the sentence.

# How to TYPE an entity (two layers: ontological role x domain)
- Role: ENTITY (something that exists as itself / an assemblage) · PROCESS (something that happens, unfolding over time) · PROPERTY (something an entity HAS — a state/measurement/characteristic) · CONCEPT (an abstract construct).
- Domain: abiotic · biotic · anthropogenic · spatial · temporal · quantitative · qualitative (CONCEPT has no domain).
- For a PROPERTY, decide the domain from what it belongs to: "population density" -> BIOTIC PROPERTY; "soil density" -> ABIOTIC PROPERTY; "habitat quality" -> SPATIAL PROPERTY or BIOTIC PROPERTY.
- Role can depend on context: "heavy rainfall caused erosion" -> rainfall is ABIOTIC PROCESS; "receives 100 mm of rainfall" -> rainfall is ABIOTIC ENTITY.
- Taxonomic names and taxonomic groups (generic OR enumerable) -> BIOTIC ENTITY: "Vulpes vulpes", "mammals", "predators", "species", "relict species".
- Spatial vs Abiotic ENTITY: a physical landform / geological feature / environmental component -> ABIOTIC ENTITY (e.g. mountains, glaciers, mountain slopes, lakes, soil). A NAMED geographic region / administrative boundary / operational mapping unit -> SPATIAL ENTITY (e.g. the Alps, Tatra National Park, study region).
- Ignore bare numbers, quantifiers, and statistical/methodological terms (e.g. "46", "several", "many", "at least", "p-value", "confidence interval", "model fit"): these are NOT entities. Only label a QUANTITATIVE PROPERTY when it NAMES a quantifiable ecological/biological/environmental attribute.
- CONCEPT vs others: an abstract construct, framework, policy/agreement, or a statistical "trend/response/pattern/dynamics" derived from data -> CONCEPT (even if human-made). A directly observable state/kind of the entity -> the matching PROPERTY. A directly observable biological action -> BIOTIC PROCESS.

# Relation Predicates (the ONLY allowed values — use verbatim)
- IS_AFFECTING: cause-effect, transformation, indication, enabling, or constraint (one thing acts on / alters / drives another).
- RELATED_TO: a non-causal association, dependency, or interaction.
- LOCATED_IN: spatial containment (subject is situated within the object place).
- COMPARES_TO: a comparison (less than, more than, or equal to).
- HAS_PROPERTY: descriptive — the subject has an attribute/value/measurement (the object).
- HAS_PROCESS: the subject entity undergoes or hosts the object process.
- CAUSES: the subject brings the object into existence (stronger than IS_AFFECTING — creation, not just influence).
- DURING: temporal containment (LOCATED_IN, but in time — subject occurs within the object time period).
- IS_PART_OF: the subject is a component/part of a larger, same-kind object entity.

# Predicate validity guardrails (from the MoBiKo guideline — do NOT force a wrong predicate; if a link fails its rule, SKIP that triplet)
- IS_AFFECTING is valid only when the SUBJECT is a PROCESS acting on an ENTITY or PROPERTY object. If the subject is an entity or a property, do not use IS_AFFECTING.
- LOCATED_IN is valid only when the SUBJECT is an ENTITY (BIOTIC/ABIOTIC/SPATIAL ENTITY) contained in a SPATIAL ENTITY or ABIOTIC ENTITY object. A process or property subject cannot be LOCATED_IN (use IS_AFFECTING for a process that occurs in / influences a place).
- HAS_PROPERTY is valid only when the SUBJECT is an ENTITY or PROCESS (not a CONCEPT), the OBJECT is a PROPERTY (not a CONCEPT/ENTITY/PROCESS), and the sentence states the subject possesses/exhibits/measures it.
- The other predicates (RELATED_TO, COMPARES_TO, HAS_PROCESS, CAUSES, DURING, IS_PART_OF) carry NO extra source/target restriction — apply each exactly as defined in the list above.
- Direction matters: subject is the source/agent, object is the target, as the sentence expresses it. Let the entities' types guide the predicate (a value/measurement object suggests HAS_PROPERTY; a place object suggests LOCATED_IN; a time period suggests DURING).
- The "predicate" is ALWAYS exactly one of the 9 labels. When no valid predicate fits a pair, simply do not emit that triplet.

# What to extract, and what to SKIP
- Extract only sentences carrying mountain-biodiversity content: the status, trends, or drivers (abiotic and anthropogenic) of mountain species/ecosystems, their conservation/management, and the ecosystem services / nature's contributions they provide.
- SKIP (emit no triplet from these): methods/analysis and sampling-protocol details (e.g. "analysis used the R package igraph"; "eighty 1 m quadrats were placed"); generic textbook truisms / universal claims (e.g. "Mountains host a large share of the world's species"); statements that only refer to previously published studies; hypotheses; and academic meta-commentary about the researchers or the paper (subjects/objects like "we", "the authors", "this study", "the results", "the dataset", "the analysis").
- Emit a relation ONLY when the sentence genuinely connects the two entities. Do NOT relate entities that merely co-occur or sit together in a list with no stated link.
- A single ordered (subject, object) pair MAY carry more than one relation — emit one triplet per relation, each with its own "relation" phrase and "predicate".
- Use the Prior Sentence Context ONLY to resolve pronouns/implicit references ("it", "this species", "there") in the Target Sentence. NEVER extract triplets from the context itself.
- If the Target Sentence is a section header/title (e.g. "Results", "2.1 Study Area") or expresses no valid relation, return an empty triplets list.

# Output Format
Return ONLY a single raw JSON object — no markdown fences, no preamble, no commentary — with this structure:
{
  "text": "<the Target Sentence, copied exactly>",
  "triplets": [
    { "subject": "<entity text, an exact substring of the sentence>", "subject_type": "<one of the 16 labels>",
      "relation": "<the relation phrase from the sentence>",
      "predicate": "<ONE of the 9 labels>",
      "object": "<entity text, an exact substring of the sentence>", "object_type": "<one of the 16 labels>" }
  ]
}
- Do NOT output character offsets — they are computed downstream from the entity text, so copy the text EXACTLY as it appears in the sentence.
- "relation" is a short phrase grounded in the sentence's wording (free text); "predicate" MUST be exactly one of: IS_AFFECTING, RELATED_TO, LOCATED_IN, COMPARES_TO, HAS_PROPERTY, HAS_PROCESS, CAUSES, DURING, IS_PART_OF.
- subject and object MUST be two DIFFERENT entities from the sentence.
- If the sentence expresses no valid relation, return: {"text": "<the Target Sentence>", "triplets": []}

# Example 1 (the SAME ordered pair carries two relations — a weaker influence and a stronger creation; both have a PROCESS subject acting on an ENTITY, as IS_AFFECTING/CAUSES require)
Target Sentence: "Sustained warming drives and ultimately triggers the formation of proglacial lakes."
Output:
{
  "text": "Sustained warming drives and ultimately triggers the formation of proglacial lakes.",
  "triplets": [
    {"subject": "Sustained warming", "subject_type": "ABIOTIC PROCESS",
     "relation": "drives", "predicate": "IS_AFFECTING",
     "object": "proglacial lakes", "object_type": "ABIOTIC ENTITY"},
    {"subject": "Sustained warming", "subject_type": "ABIOTIC PROCESS",
     "relation": "triggers the formation of", "predicate": "CAUSES",
     "object": "proglacial lakes", "object_type": "ABIOTIC ENTITY"}
  ]
}

# Example 2 (unfolding a conjunction; containment in a named place)
Target Sentence: "In the Tatra National Park, chamois and marmots depend on high-elevation meadows."
Output:
{
  "text": "In the Tatra National Park, chamois and marmots depend on high-elevation meadows.",
  "triplets": [
    {"subject": "chamois", "subject_type": "BIOTIC ENTITY",
     "relation": "depend on", "predicate": "RELATED_TO",
     "object": "high-elevation meadows", "object_type": "BIOTIC ENTITY"},
    {"subject": "marmots", "subject_type": "BIOTIC ENTITY",
     "relation": "depend on", "predicate": "RELATED_TO",
     "object": "high-elevation meadows", "object_type": "BIOTIC ENTITY"},
    {"subject": "chamois", "subject_type": "BIOTIC ENTITY",
     "relation": "in", "predicate": "LOCATED_IN",
     "object": "Tatra National Park", "object_type": "SPATIAL ENTITY"},
    {"subject": "marmots", "subject_type": "BIOTIC ENTITY",
     "relation": "in", "predicate": "LOCATED_IN",
     "object": "Tatra National Park", "object_type": "SPATIAL ENTITY"}
  ]
}

# Example 3 (a methods / meta-commentary sentence yields nothing)
Target Sentence: "We analysed the data using linear mixed models in R."
Output:
{"text": "We analysed the data using linear mixed models in R.", "triplets": []}

# Input Data
Prior Sentence Context: [INSERT CONTEXT HERE]
Target Sentence: [INSERT SENTENCE HERE]
'''
