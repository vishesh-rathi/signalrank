"""Central configuration for the Redrob ranker — the JD-as-config layer.

Everything that is specific to *this* job description (requirement probes, the
lexicons that encode the JD's "what they actually built" signal, feature
weights, and thresholds) lives here. Swapping this module retargets the ranker
to a different JD; the honeypot filter and behavioral multiplier are
JD-agnostic and read nothing from it beyond the thresholds below.

All lexicon terms are lowercase and are matched with WORD BOUNDARIES (see
``ranker.util.text_has``), never naive substring containment — otherwise
``"search engine"`` would spuriously match "re**search engine**er" and
``"search"`` would match "re**search**". Keep terms lowercase; do not add
leading/trailing punctuation.
"""

from datetime import date

# Fixed reference date for all recency math. The dataset is a static snapshot,
# so we never use ``date.today()`` — that would make scores non-reproducible
# (a Stage-3 requirement). This is the snapshot's "now".
DATA_AS_OF = date(2026, 6, 9)

# ---------------------------------------------------------------------------
# Free-text evidence lexicons
#
# The skills[] array is a decoy (uniform-frequency noise by design), so fit is
# scored on the career narrative (summary + role descriptions). STRONG phrases
# describe building the systems this JD wants; MED phrases describe generic
# ML-in-production (far more common, hence a weaker signal).
#
# STRONG terms are grouped into CONCEPTS so the evidence feature can grade
# *depth* by counting distinct concepts a narrative demonstrates, not just
# presence/absence: one passing mention of "ranking systems" is real but shallow
# evidence, four distinct built-systems concepts is a deep builder. Synonyms of
# the same idea (e.g. "ranking system"/"ranking models"/"re-rank") live in one
# bucket so they count once. STRONG_PHRASES is the flattened view the lexical
# matcher and the dev-set rubric consume.
# ---------------------------------------------------------------------------
STRONG_CONCEPTS = {
    "ranking": [
        "ranking system",
        "ranking systems",
        "ranking model",
        "ranking models",
        "learning to rank",
        "learning-to-rank",
        "candidate ranking",
        "re-rank",
        "re-ranking",
        "re-ranked",
        "rerank",
        "reranking",
        "reranked",
    ],
    "recommendation": [
        "recommendation system",
        "recommendation systems",
        "recommendation engine",
        "recommendation engines",
        "recommender",
        "recommenders",
        "recsys",
        "rec sys",
    ],
    "search": [
        "search engine",
        "search engines",
        "search system",
        "search systems",
        "semantic search",
    ],
    "retrieval": [
        "information retrieval",
        "retrieval system",
        "retrieval systems",
        "vector search",
        "embedding-based retrieval",
        "embeddings-based retrieval",
        "dense retrieval",
        "retrieval",
    ],
    # The JD's two infrastructure must-haves, in its own words: "embeddings-based
    # retrieval systems" and "vector databases or hybrid search infrastructure —
    # Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS"; it
    # also names "BM25" and "hybrid retrieval" as the stack being built. A
    # candidate narrating THIS vocabulary is demonstrating the exact capability
    # the JD calls non-negotiable, so it counts as its own depth concept.
    "hybrid search": [
        "hybrid retrieval",
        "hybrid search",
        "bm25",
        "dense vector",
        "sparse vector",
        "vector database",
        "vector databases",
        "vector index",
        "vector indexes",
        "faiss",
        "pinecone",
        "weaviate",
        "qdrant",
        "milvus",
        "opensearch",
        "elasticsearch",
    ],
    "relevance": ["relevance", "personalization", "personalisation"],
    "matching": ["matching system", "matching systems"],
    "evaluation": [
        "evaluation framework",
        "evaluation frameworks",
        "eval framework",
        "eval frameworks",
        "ndcg",
        "mrr",
        "map",
        "a/b test",
        "a/b testing",
        "ab test",
        "ab testing",
        "offline/online evaluation",
        "offline evaluation",
        "online evaluation",
    ],
}
STRONG_PHRASES = [phrase for terms in STRONG_CONCEPTS.values() for phrase in terms]
MED_PHRASES = [
    "machine learning model",
    "machine learning models",
    "ml model",
    "ml models",
    "deployed",
    "production model",
    "production models",
    "trained a model",
    "trained models",
    "deep learning",
    "neural network",
    "neural networks",
    "embeddings",
    "embedding",
    "transformer",
    "transformers",
    "fine-tuning",
    "fine-tuned",
    "nlp",
    "llm",
    "llms",
    "rag",
    "a/b test",
    "ab test",
    "click-through",
    "ctr",
]

# ---------------------------------------------------------------------------
# Domain gating
#
# Built-evidence alone is a trap: the candidate pool salts in Computer Vision
# and pure-research engineers who have strong ML prose but are explicitly
# rejected by the JD. POSITIVE terms keep the technical-fit gate open (1.0);
# NEGATIVE terms (or research without any production signal) clamp it to 0.2.
# ---------------------------------------------------------------------------
DOMAIN_POSITIVE = [
    "nlp",
    "natural language",
    "information retrieval",
    "retrieval",
    "ranking",
    "rankings",
    "recommendation",
    "recommendations",
    "recommender",
    "recommenders",
    "search",
    "semantic",
    "embedding",
    "embeddings",
    "llm",
    "llms",
]
DOMAIN_NEGATIVE = [
    "computer vision",
    "image classification",
    "object detection",
    "yolo",
    "speech recognition",
    "tts",
    "asr",
    "robotics",
    "autonomous",
    "lidar",
    "gans",
]
# "research" only counts against a candidate when no production signal accompanies it
# (the JD rejects research-only profiles, not applied researchers who ship).
RESEARCH_WORDS = [
    "research",
    "researcher",
    "researchers",
    "phd",
    "publication",
    "publications",
    "paper",
    "papers",
    "academic",
    "thesis",
]
PRODUCTION_WORDS = [
    "production",
    "deployed",
    "shipped",
    "users",
    "scale",
    "platform",
    "platforms",
    "live",
]

# Seniority is judged from years + the scope language in the narrative, not the
# job title (the JD's exact title barely exists in the pool).
SCOPE_WORDS = [
    "led",
    "owned",
    "designed",
    "architected",
    "at scale",
    "end-to-end",
    "production",
    "drove",
    "built and shipped",
]

# ---------------------------------------------------------------------------
# Location & company signals
# ---------------------------------------------------------------------------
# Cities the JD names explicitly (offices in Pune/Noida; "Hyderabad, Pune,
# Mumbai, Delhi NCR welcome to apply") score full location credit.
TARGET_CITIES = [
    "pune",
    "noida",
    "hyderabad",
    "mumbai",
    "delhi",
    "gurgaon",
    "gurugram",
    "ncr",
]
# Bangalore is India's largest tech hub but is NOT on the JD's list. Treat it as
# a strong secondary metro: high but not full credit, and full only with an
# explicit relocation intent — the JD's cities are Pune/Noida, so an unconditional
# 1.0 for Bangalore over-credits ~12% of the pool at the third-highest weight.
SECONDARY_CITIES = ["bangalore", "bengaluru"]
SECONDARY_CITY_SCORE = 0.8
# An entirely services/consulting career is a soft down-weight (not exclusion):
# genuine builders exist even inside these firms.
SERVICES_FIRMS = [
    "tcs",
    "tata consultancy",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl",
    "tech mahindra",
    "mindtree",
    "ltimindtree",
    "mphasis",
    "dxc",
    "larsen",
    "l&t infotech",
]
# Every product-tech industry code observed in the pool. "AI Services" is
# deliberately absent (services semantics -> neutral 0.6, and genuine builders
# there are rescued by prior product roles); Manufacturing / Conglomerate /
# Paper Products are non-tech and also stay neutral.
PRODUCT_INDUSTRIES = {
    "Software",
    "Fintech",
    "E-commerce",
    "Food Delivery",
    "SaaS",
    "AI/ML",
    "EdTech",
    "AdTech",
    "Insurance Tech",
    "Transportation",
    "HealthTech",
    "HealthTech AI",
    "Conversational AI",
    "Voice AI",
    "Gaming",
    "Internet",
    "Media",
    "Consumer Electronics",
}
SERVICES_INDUSTRIES = {"IT Services", "Consulting"}

# ---------------------------------------------------------------------------
# JD requirement probes — embedded by precompute.py and compared (cosine)
# against each candidate's narrative to produce the semantic-similarity signal.
# ---------------------------------------------------------------------------
JD_PROBES = [
    "Built and deployed embeddings-based retrieval systems for real users in production.",
    "Designed ranking and recommendation systems at scale at a product company.",
    "Experience with vector search and hybrid search infrastructure in production.",
    "Strong applied machine learning and NLP / information retrieval background.",
    "Designed evaluation frameworks for ranking systems (NDCG, MRR, MAP, A/B testing).",
    "Shipped end-to-end search or recommendation features to real users.",
    "LLM-based re-ranking and semantic search in a production system.",
    "Senior engineer who writes production Python and owns ML systems.",
]

# ---------------------------------------------------------------------------
# Technical blend: lexical evidence is precision, the pool-percentile semantic
# is recall. At the top of the pool the percentile saturates (~0.99 for every
# plausible builder — the raw cosines span only ~[0.60, 0.69]), so an
# equal-weight average halves the graded evidence gap exactly where NDCG@10 is
# decided. 4:1 keeps the semantic channel as the recall/tiebreak axis (it still
# floats plain-language builders the lexicon misses) without letting a
# saturated signal dilute the one feature that separates builder tiers.
# ---------------------------------------------------------------------------
TECH_SEMANTIC_SHARE = 0.2

# ---------------------------------------------------------------------------
# Fit weights. Hand-set from the JD's own emphasis and audited against
# full-pool behavior (eval/archetype_report.py), not auto-applied from the
# dev-set tuner — the dev set is builder-heavy and the tuner zeroes axes the
# JD names explicitly. tech dominates: once built-evidence actually separates
# builder tiers (graded summary-sourced concepts), the JD's core demand —
# "shipped at least one end-to-end ranking, search, or recommendation system"
# — must outvote every contextual signal; at 0.35 a location swing
# (0.6 vs 1.0) outweighed a full evidence band and let self-disclaiming
# generic profiles into the top-10 over measured builders. location is
# deliberately light: the JD calls it "flexible", welcomes four metros plus
# relocators, so it tie-breaks rather than ranks. seniority second (the 5-9y
# band), product non-zero (services-only careers are a JD rejection),
# stability modest (anti title-chaser), education minimal (the JD prizes
# shipped systems over pedigree). The A/B over the full pool: this vector vs
# the previous (tech .35 / location .17) moves the top-10 from 6 elite + 1
# generic to 7 elite + 0 generic and cuts top-50 generic intrusion 14 -> 8
# with honeypots still 0. tune.py renormalizes, so the sum need not be 1.0;
# it is here for readability.
# ---------------------------------------------------------------------------
WEIGHTS = {
    "tech": 0.45,
    "seniority": 0.22,
    "product": 0.10,
    "location": 0.12,
    "education": 0.03,
    "stability": 0.08,
}

# ---------------------------------------------------------------------------
# Title-chaser enforcement. The JD lists title-chasers under "Things we
# explicitly do NOT want": "optimizing for 'Senior' -> 'Staff' -> 'Principal'
# titles by switching companies every 1.5 years ... We need someone who plans to
# be here for 3+ years." ``stability_score`` (weight 0.08) grades retention
# smoothly across the whole pool, but at that weight it cannot keep a strong-fit
# chaser out of the top ranks on its own. So the *unambiguous* pattern — four or
# more roles averaging under 18 months (the JD's "every 1.5 years") —
# additionally scales fit down by a fixed factor. This is a down-weight, not an
# exclusion: parity with how the JD's other named rejections (CV/speech via the
# domain-title gate, consulting-only via product) are enforced multiplicatively,
# never a honeypot-style zero.
# ---------------------------------------------------------------------------
CHASER_MIN_ROLES = 4
CHASER_MAX_AVG_TENURE_MONTHS = 18
TENURE_CHASER_PENALTY = 0.75

# ---------------------------------------------------------------------------
# Behavioral multiplier: scales fit into [MULT_FLOOR, 1.0]. It can only
# down-weight — a great-on-paper but unavailable candidate sinks, but the
# modifier can never lift a weak fit above an available strong one.
# ---------------------------------------------------------------------------
MULT_WEIGHTS = {"availability": 0.30, "responsiveness": 0.30, "recency": 0.25, "credibility": 0.15}
MULT_FLOOR = 0.30

# Notice-period availability curve. The JD: "We'd love sub-30-day notice. We can
# buy out up to 30 days. 30+ day notice candidates are still in scope but the bar
# gets higher." A notice within the buy-out window therefore earns full notice
# credit; beyond it the factor decays *convexly* (squared) to zero at the
# dataset's 180-day ceiling, so a long notice stays in scope but the bar rises
# progressively — a 120-day notice is penalized far more than the previous linear
# decay (0.16 vs 0.33) while never being excluded outright.
NOTICE_BUYOUT_DAYS = 30
NOTICE_MAX_DAYS = 180

# ---------------------------------------------------------------------------
# Honeypot thresholds (deterministic consistency checks; ~80 forced to tier 0).
# Conservative — real honeypots show tenure deltas of +100 months or more.
# ---------------------------------------------------------------------------
TENURE_DELTA_MAX = 18  # months; |claimed duration - actual date span| above this is impossible
PHANTOM_EXPERT_MIN = 3  # count of expert/advanced skills claimed with 0 months used
CAREER_OVER_LIFE_SLACK = 24  # months of total tenure allowed beyond years_of_experience
# Stated experience may legitimately exceed the calendar span of the *listed*
# roles (early jobs omitted), so the slack is generous. The planted profiles in
# this class overshoot by 96+ months ("8 yrs at a 3-yr-old company"); the widest
# legitimate gap in the pool is <= 30 months, so 48 catches every plant with a
# ~4-year false-positive margin on either side of a clean bimodal gap.
CAREER_SPAN_SLACK = 48  # months of stated experience allowed beyond the listed career span
