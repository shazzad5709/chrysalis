SEED = 42
REGRESSION_THRESHOLD = -0.05
NEUTRAL_LABEL = 1
SCORE_TOLERANCE_AIRTIGHT = 0.05
SCORE_TOLERANCE_NEARFORMAL = 0.10

NEGATION_WORDS = [
    "not",
    "n't",
    "never",
    "no",
    "nobody",
    "nothing",
    "nowhere",
    "neither",
    "nor",
]

INTENSIFIER_POOL = [
    "very",
    "extremely",
    "incredibly",
    "remarkably",
    "exceptionally",
    "particularly",
    "truly",
    "genuinely",
]

ABBREVIATION_SAFELIST = {
    "titles": ["Mr.", "Mrs.", "Dr.", "Prof.", "Sr.", "Jr."],
    "geographic": ["St.", "Ave.", "Blvd."],
    "misc": ["etc.", "vs.", "approx."],
    "initials_pattern": r"^[A-Z]\.$",
}
