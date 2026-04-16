"""
LiftMind Brain - Core logic module.

This module contains the decoupled core logic used by all interfaces:
- Telegram bot
- REST API (IT dept app)
- Future clients

Architecture:
    Any Client -> brain.process_query() -> RAG + Claude -> Response
"""
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, List
from liftmind.claude_adapter import ask_claude, ask_claude_with_image, ask_claude_streaming
from liftmind.config import settings
from liftmind.rag import search_documents, search_documents_hybrid, search_images, format_context, AMBIGUOUS_DRIVE_MODELS, AMBIGUOUS_DOOR_MODELS
from liftmind.user_state import get_user_model, set_user_model, increment_query_count
from liftmind.context_cache import deep_dive_query
from liftmind.analytics import log_query
from liftmind.slang_interceptor import intercept_query_sync, _create_fallback_response
from liftmind.response_cache import get_cached_response, store_response

# Wiki article exact-match lookup
try:
    from liftmind.wiki_compiler import get_wiki_article
    WIKI_AVAILABLE = True
except ImportError:
    WIKI_AVAILABLE = False

# Semantic similarity cache
try:
    from liftmind.semantic_cache import get_semantic_cached_response, store_semantic_response
    SEMANTIC_CACHE_AVAILABLE = True
except ImportError:
    SEMANTIC_CACHE_AVAILABLE = False

# Direct PDF reader fallback
try:
    from liftmind.manual_reader import search_manuals_direct
    MANUAL_READER_AVAILABLE = True
except ImportError:
    MANUAL_READER_AVAILABLE = False

# Self-learning module
try:
    from liftmind.learning import learn_from_direct_read
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False

logger = logging.getLogger(__name__)

# Shared executor for background interceptor work
_interceptor_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="interceptor")


# ============================================================================
# LIFT MODEL CONFIGURATION
# ============================================================================

VALID_MODELS = {
    # Elfo family
    "Elfo", "Elfo 2", "E3", "Elfo Cabin",
    "Elfo Electronic", "Elfo Hydraulic controller", "Elfo Traction",
    # Supermec family
    "Supermec", "Supermec 2", "Supermec 3",
    # Freedom family
    "Freedom", "Freedom MAXI", "Freedom STEP",
    # Pollock
    "Pollock (P1)", "Pollock (Q1)",
    # Individual
    "Bari", "P4", "Tresa",
}

MODEL_ALIASES = {
    # Elfo family
    "elfo": "Elfo",
    "elfo traction": "Elfo Traction",
    "elfo t": "Elfo Traction",
    "elfo hydro": "Elfo Hydraulic controller",
    "elfo hydraulic": "Elfo Hydraulic controller",
    "elfo electronic": "Elfo Electronic",
    "elfo elec": "Elfo Electronic",
    "elfo cabin": "Elfo Cabin",
    "elfo 2": "Elfo 2",
    "elfo2": "Elfo 2",
    "e3": "E3",
    # Supermec family
    "supermec": "Supermec",
    "supermec 2": "Supermec 2",
    "supermec2": "Supermec 2",
    "sm2": "Supermec 2",
    "supermec 3": "Supermec 3",
    "supermec3": "Supermec 3",
    "sm3": "Supermec 3",
    # Freedom family
    "freedom": "Freedom",
    "freedom maxi": "Freedom MAXI",
    "freedom step": "Freedom STEP",
    # Pollock
    "pollock p1": "Pollock (P1)",
    "pollock q1": "Pollock (Q1)",
    "pollock": "Pollock (Q1)",  # Q1 is more common
    "p1": "Pollock (P1)",
    "q1": "Pollock (Q1)",
    # Individual
    "bari": "Bari",
    "p4": "P4",
    "tresa": "Tresa",
}


def resolve_model(model_input: str) -> Optional[str]:
    """
    Resolve a model input string to a canonical model name.

    Args:
        model_input: User-provided model name (may be fuzzy)

    Returns:
        Canonical model name or None if not found
    """
    if not model_input:
        return None

    normalized = model_input.lower().strip()

    # Check aliases first
    if normalized in MODEL_ALIASES:
        return MODEL_ALIASES[normalized]

    # Check exact match (case-insensitive)
    for valid_model in VALID_MODELS:
        if valid_model.lower() == normalized:
            return valid_model

    return None


# ============================================================================
# KEYWORD SUPPLEMENTATION
# ============================================================================

# Technical terms that should be extracted from user queries when the
# interceptor misses them. Single words checked via set membership.
_TECH_TERMS = {
    'door', 'lock', 'latch', 'interlock', 'motor', 'pump', 'controller', 'board',
    'sensor', 'switch', 'limit', 'safety', 'encoder', 'inverter', 'drive', 'relay',
    'contactor', 'brake', 'valve', 'cylinder', 'piston', 'ram', 'cable', 'rope',
    'leveling', 'levelling', 'floor', 'travel', 'speed', 'position',
    'error', 'fault', 'alarm', 'code', 'warning', 'hydraulic', 'traction',
    'mounting', 'calibrate', 'adjust', 'parameter', 'wiring', 'terminal',
    'voltage', 'current', 'power', 'supply', 'phase', 'overload', 'stuck',
    'blocked', 'slow', 'fast', 'closed', 'open', 'leakage', 'sinking',
    'buffer', 'magnet', 'polarity', 'maintenance', 'commissioning', 'inspection',
    'pcb', 'pit', 'governor', 'overspeed', 'overtravel', 'display', 'menu',
    'setting', 'config', 'curtain', 'light', 'diagnostic', 'diagnostics',
    'bistable', 'selector', 'sheave', 'counterweight', 'hoistway', 'nudging',
    'parking', 'safeties', 'plunger', 'manifold', 'isolator', 'fuse',
    'transformer', 'resistor', 'capacitor', 'diode', 'rectifier', 'thyristor',
    'sill', 'apron', 'handrail', 'balustrade', 'blinking', 'blink', 'blinks',
    'input', 'output', 'led', 'signal', 'component', 'failure',
}

# Multi-word technical phrases to extract as single terms.
_TECH_PHRASES = [
    'plc input', 'plc output', 'blinking code', 'error code', 'fault code',
    'safety circuit', 'door lock', 'door operator', 'limit switch',
    'maintenance mode', 'inspection mode', 'overload signal',
    'contactor blocked', 'component failure', 'slow down', 'slow-down',
]


def _supplement_keywords(original_query: str, interceptor_keywords: list) -> list:
    """
    Add technical terms from the user's original query that the interceptor missed.

    The interceptor sometimes drops important terms. This function extracts
    technical words/phrases from the raw query and appends any that aren't
    already covered by the interceptor's keywords.
    """
    existing_lower = ' '.join(interceptor_keywords).lower()
    query_lower = original_query.lower()
    additions = []

    # Extract multi-word technical phrases first
    for phrase in _TECH_PHRASES:
        if phrase in query_lower and phrase not in existing_lower:
            additions.append(phrase)

    # Extract single technical terms
    words = re.findall(r'\b[a-zA-Z0-9]+\b', query_lower)
    for word in words:
        if word in _TECH_TERMS and word not in existing_lower and word not in ' '.join(additions):
            additions.append(word)

    if additions:
        logger.info(f"Supplemented keywords with: {additions}")
        return interceptor_keywords + additions

    return interceptor_keywords


# ============================================================================
# QUERY PREPROCESSING
# ============================================================================

# Technical synonyms for query expansion
# Maps common terms to their technical equivalents for better search coverage
TECHNICAL_SYNONYMS = {
    # Symptoms and states
    'not responding': ['stuck', 'frozen', 'no response', 'not working', 'unresponsive'],
    'not working': ['stuck', 'frozen', 'no response', 'not responding', 'failed'],
    'stuck': ['jammed', 'blocked', 'frozen', 'not moving', 'seized'],
    'slow': ['sluggish', 'delayed', 'lagging', 'taking long'],

    # Errors and faults
    'error': ['fault', 'problem', 'issue', 'malfunction', 'failure', 'alarm'],
    'fault': ['error', 'problem', 'issue', 'malfunction', 'failure', 'alarm'],
    'problem': ['error', 'fault', 'issue', 'malfunction', 'failure'],
    'alarm': ['error', 'fault', 'warning', 'alert'],

    # Components
    'door': ['gate', 'landing door', 'car door', 'entrance'],
    'lock': ['latch', 'interlock', 'engage', 'secure', 'locking device'],
    'motor': ['drive', 'pump', 'unit', 'power unit'],
    'button': ['push button', 'call button', 'switch', 'cop button'],
    'controller': ['control board', 'pcb', 'board', 'control unit', 'processor'],
    'sensor': ['switch', 'detector', 'limit switch', 'proximity'],

    # Interface
    'menu': ['navigation', 'screen', 'display', 'settings', 'parameter', 'configuration'],
    'screen': ['display', 'panel', 'lcd', 'indicator'],
    'setting': ['parameter', 'configuration', 'option', 'value'],

    # Actions
    'adjust': ['calibrate', 'set', 'configure', 'tune', 'change'],
    'replace': ['change', 'swap', 'install new', 'substitute'],
    'check': ['verify', 'inspect', 'test', 'measure', 'confirm'],
    'reset': ['restart', 'reboot', 'clear', 'initialize'],

    # Lift-specific
    'leveling': ['levelling', 'floor level', 'stopping accuracy', 'positioning'],
    'overload': ['excess weight', 'too heavy', 'weight limit', 'capacity exceeded'],
    'emergency': ['e-stop', 'safety', 'alarm', 'rescue'],
}


def _expand_query_synonyms(query: str) -> str:
    """
    Expand query with technical synonyms for better search coverage.

    Takes common terms and adds related technical terms so full-text search
    can find matches even when exact words differ.

    Args:
        query: The original user query

    Returns:
        Expanded query with synonym terms appended
    """
    query_lower = query.lower()
    expansion_terms = set()

    for term, synonyms in TECHNICAL_SYNONYMS.items():
        # Check if the base term appears in the query
        if term in query_lower:
            # Add all synonyms (not the original term, it's already there)
            for syn in synonyms:
                # Don't add if synonym is also already in the query
                if syn not in query_lower:
                    expansion_terms.add(syn)

        # Also check if any synonym appears, and add the base term
        for syn in synonyms:
            if syn in query_lower and term not in query_lower:
                expansion_terms.add(term)
                break

    if expansion_terms:
        # Append expansion terms to original query
        return f"{query} {' '.join(expansion_terms)}"

    return query


def generate_synonym_queries(query: str, max_queries: int = 3) -> List[str]:
    """
    Generate multiple separate synonym queries for RRF fusion.

    Instead of appending all synonyms to a single query (which dilutes BM25 weight),
    this generates 2-3 separate queries, each focused on a specific synonym variant.
    Each query then gets full BM25 weight when run through RRF fusion.

    Args:
        query: The original user query
        max_queries: Maximum number of synonym queries to generate (default 3)

    Returns:
        List of query variants (original + up to max_queries-1 synonym variants)

    Example:
        "door stuck" -> [
            "door stuck",           # Original
            "gate jammed",          # Synonym variant 1
            "entrance blocked"      # Synonym variant 2
        ]
    """
    query_lower = query.lower()
    queries = [query]  # Always include original

    # Find terms in query that have synonyms
    matched_terms = []
    for term, synonyms in TECHNICAL_SYNONYMS.items():
        if term in query_lower:
            matched_terms.append((term, synonyms))
        else:
            # Check if any synonym is in the query
            for syn in synonyms:
                if syn in query_lower:
                    # Treat this as finding the base term
                    matched_terms.append((syn, [term] + [s for s in synonyms if s != syn]))
                    break

    if not matched_terms:
        return queries

    # Generate variant queries by replacing terms with synonyms
    # Limit to max_queries-1 variants (original is already in list)
    import itertools

    # Get up to 2 synonyms per matched term for combinatorics
    term_options = []
    for term, synonyms in matched_terms[:2]:  # Limit to 2 terms for combinatorics
        # Take top 2 most relevant synonyms
        term_options.append((term, synonyms[:2]))

    # Generate combinations
    variants_added = 0
    for term, synonyms in term_options:
        for syn in synonyms:
            if variants_added >= max_queries - 1:
                break
            # Create variant by replacing term with synonym
            variant = query_lower.replace(term, syn)
            if variant != query_lower and variant not in [q.lower() for q in queries]:
                queries.append(variant)
                variants_added += 1

        if variants_added >= max_queries - 1:
            break

    return queries


# Number word mappings for voice input normalization
NUMBER_WORDS = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
    'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
    'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
    'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
    'eighty': '80', 'ninety': '90', 'hundred': '100'
}

# Compound number patterns
COMPOUND_NUMBERS = [
    (r'twenty\s*one', '21'), (r'twenty\s*two', '22'), (r'twenty\s*three', '23'),
    (r'twenty\s*four', '24'), (r'twenty\s*five', '25'), (r'twenty\s*six', '26'),
    (r'twenty\s*seven', '27'), (r'twenty\s*eight', '28'), (r'twenty\s*nine', '29'),
    (r'thirty\s*one', '31'), (r'thirty\s*two', '32'), (r'thirty\s*three', '33'),
    (r'thirty\s*four', '34'), (r'thirty\s*five', '35'), (r'thirty\s*six', '36'),
    (r'thirty\s*seven', '37'), (r'thirty\s*eight', '38'), (r'thirty\s*nine', '39'),
    (r'forty\s*one', '41'), (r'forty\s*two', '42'), (r'forty\s*three', '43'),
    (r'forty\s*four', '44'), (r'forty\s*five', '45'), (r'forty\s*six', '46'),
    (r'forty\s*seven', '47'), (r'forty\s*eight', '48'), (r'forty\s*nine', '49'),
    (r'fifty\s*one', '51'), (r'fifty\s*two', '52'), (r'fifty\s*three', '53'),
]


def normalise_voice_input(query: str) -> str:
    """
    Normalize voice input to standard format.

    Handles:
    - "ee twenty three" -> "E23"
    - "fault seven" -> "Fault 7"
    - "error code forty two" -> "error code 42"
    - "point" -> "." for decimals

    Returns the normalized query.
    """
    result = query

    # Handle "ee" (spoken as letter E) at start of fault codes
    result = re.sub(r'\b(ee|e)\s+', r'E', result, flags=re.IGNORECASE)
    result = re.sub(r'\b(eff|ef)\s+', r'F', result, flags=re.IGNORECASE)

    # Convert compound numbers first (more specific)
    for pattern, digit in COMPOUND_NUMBERS:
        result = re.sub(pattern, digit, result, flags=re.IGNORECASE)

    # Convert single number words to digits
    for word, digit in NUMBER_WORDS.items():
        result = re.sub(rf'\b{word}\b', digit, result, flags=re.IGNORECASE)

    # Handle "point" for decimals (e.g., "twenty five point four" -> "25.4")
    result = re.sub(r'\s+point\s+', '.', result, flags=re.IGNORECASE)

    # Clean up extra spaces
    result = re.sub(r'\s+', ' ', result).strip()

    return result


# Fault code patterns (case-insensitive)
FAULT_CODE_PATTERNS = [
    r'^E\d{1,3}$',           # E23, E101
    r'^Er\d{1,3}$',          # Er07
    r'^F\d{1,3}$',           # F4, F12
    r'^Fault\s*\d{1,3}$',    # Fault 7, Fault12
    r'^AL\d{1,3}$',          # AL03
    r'^A\d{1,3}$',           # A23
    r'^error\s+\d{1,3}$',    # error 7
    r'^code\s+\d{1,3}$',     # code 12
    r'^alarm\s+\d{1,3}$',    # alarm 5
]

# Compiled patterns for efficiency
_FAULT_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in FAULT_CODE_PATTERNS]

# Query type detection patterns
PROCEDURE_KEYWORDS = [
    'how to', 'how do i', 'steps', 'procedure', 'set up', 'setup',
    'install', 'commission', 'adjust', 'replace', 'calibrate',
    'configure', 'program', 'reset', 'clear'
]
SPECIFICATION_KEYWORDS = [
    "what is the", "what's the", "spec", "torque", "voltage", "amp",
    "wire gauge", "clearance", "dimension", "size", "rating",
    "capacity", "weight", "speed"
]


def detect_fault_code(query: str) -> Optional[str]:
    """
    Detect if the query is a fault code.

    Returns the detected code or None.
    """
    query_stripped = query.strip()

    # Check against all patterns (whole-query match -- bare fault code queries)
    for pattern in _FAULT_PATTERNS_COMPILED:
        if pattern.match(query_stripped):
            return query_stripped.upper()

    # Check for fault codes embedded in short queries (compact style: E23, F4, Er07, AL03)
    if len(query_stripped) < 15:
        match = re.search(r'\b(E\d{1,3}|Er\d{1,3}|F\d{1,3}|AL\d{1,3})\b', query_stripped, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # For longer queries: also detect "ERROR N" / "ALARM N" style codes embedded in prose
    # (e.g. "The activation board is displaying an ERROR 4. What does it mean?")
    match = re.search(r'\b(ERROR\s+\d{1,3}|ALARM\s+\d{1,3}|EPCHOPPER\s+[EW]\d{1,2})\b', query_stripped, re.IGNORECASE)
    if match:
        # Normalise spacing: "ERROR  4" -> "ERROR 4"
        code = re.sub(r'\s+', ' ', match.group(1)).upper()
        return code

    return None


def detect_query_type(query: str) -> str:
    """
    Classify the query type.

    Returns: "fault_code", "procedure", "specification", or "general"
    """
    query_lower = query.lower()

    # Check for fault code
    if detect_fault_code(query):
        return "fault_code"

    # Check for procedure keywords
    for keyword in PROCEDURE_KEYWORDS:
        if keyword in query_lower:
            return "procedure"

    # Check for specification keywords
    for keyword in SPECIFICATION_KEYWORDS:
        if keyword in query_lower:
            return "specification"

    return "general"


def _lookup_wiki_bypass(
    identifier: str,
    entity_type: str,
    lift_models: Optional[List[str]],
) -> Optional[str]:
    """
    Exact-match lookup in wiki_articles for a known identifier.

    Returns compiled article content if found, else None.
    A None result means the caller should proceed with full hybrid search.
    """
    if not WIKI_AVAILABLE:
        return None
    try:
        return get_wiki_article(entity_type, identifier, lift_models)
    except Exception as e:
        logger.warning(f"Wiki bypass lookup failed ({entity_type}/{identifier}): {e}")
        return None


# Parameter code pattern: letter prefix + 2-3 digits (e.g. L65, P12, d03)
# Excludes fault-code prefixes (E, Er, F, AL, A) already handled by detect_fault_code()
_PARAM_CODE_RE = re.compile(r'^([LPDCBGHIJKMNOQRSTUVWXYZ]\d{2,3})$', re.IGNORECASE)

# Nidec/dotted-numeric parameter style: "Pr 0.024", "5.000", "06.010"
# Matches optional "Pr " prefix + dotted decimal (group/function.number format)
_NIDEC_PARAM_RE = re.compile(r'^(?:Pr\s+)?(\d{1,2}\.\d{3})$', re.IGNORECASE)

# Part/component code pattern for very short bare-identifier queries (e.g. TR1, PC4, MPCB)
_PART_CODE_RE = re.compile(r'^([A-Z]{1,5}[\-]?[0-9]{1,6})$', re.IGNORECASE)


def detect_model_from_query(query: str) -> Optional[str]:
    """
    Try to detect the lift model from the query text.

    Uses word boundary regex to avoid false matches.

    Returns:
        Canonical model name if detected, None otherwise
    """
    query_lower = query.lower()

    # Model detection patterns (order matters - check specific before general)
    model_patterns = [
        # Elfo family - specific first
        (r'\belfo\s+traction\b', "Elfo Traction"),
        (r'\belfo\s*t\b', "Elfo Traction"),
        (r'\belfo\s+hydraulic\b', "Elfo Hydraulic controller"),
        (r'\belfo\s+hydro\b', "Elfo Hydraulic controller"),
        (r'\belfo\s+electronic\b', "Elfo Electronic"),
        (r'\belfo\s+elec\b', "Elfo Electronic"),
        (r'\belfo\s+cabin\b', "Elfo Cabin"),
        (r'\belfo\s*2\b', "Elfo 2"),
        (r'\be3\b', "E3"),
        (r'\belfo\b', "Elfo"),  # Generic elfo last

        # Supermec family
        (r'\bsupermec\s*3\b', "Supermec 3"),
        (r'\bsm3\b', "Supermec 3"),
        (r'\bsupermec\s*2\b', "Supermec 2"),
        (r'\bsm2\b', "Supermec 2"),
        (r'\bsupermec\b', "Supermec"),

        # Freedom family
        (r'\bfreedom\s+maxi\b', "Freedom MAXI"),
        (r'\bfreedom\s+step\b', "Freedom STEP"),
        (r'\bfreedom\b', "Freedom"),

        # Pollock
        (r'\bpollock\s*\(?p1\)?\b', "Pollock (P1)"),
        (r'\bp1\s+lift\b', "Pollock (P1)"),
        (r'\bpollock\s*\(?q1\)?\b', "Pollock (Q1)"),
        (r'\bq1\s+lift\b', "Pollock (Q1)"),
        (r'\bq1\b', "Pollock (Q1)"),  # Q1 alone = Q1 (more common)
        (r'\bpollock\b', "Pollock (Q1)"),

        # Individual
        (r'\bbari\b', "Bari"),
        (r'\bp4\b', "P4"),
        (r'\btresa\b', "Tresa"),
    ]

    for pattern, model in model_patterns:
        if re.search(pattern, query_lower):
            return model

    return None


# ============================================================================
# DRIVE TYPE & DOOR TYPE DETECTION
# ============================================================================

# Patterns for drive type detection from query text
_DRIVE_TYPE_PATTERNS = [
    (r'\btraction\b', "traction"),
    (r'\bMRL\b', "traction"),
    (r'\bgearless\b', "traction"),
    (r'\bropes?\b.*\b(drive|motor|machine)\b', "traction"),
    (r'\bcounterweight\b', "traction"),
    (r'\bVSD\b', "traction"),
    (r'\bVFD\b', "traction"),
    (r'\binverter\s+drive\b', "traction"),
    (r'\bhydraulic\b', "hydraulic"),
    (r'\bhydro\b', "hydraulic"),
    (r'\bpump\b', "hydraulic"),
    (r'\b(ram|cylinder)\b.*\b(oil|hydraulic|lift)\b', "hydraulic"),
    (r'\boil\s+(unit|tank|level|leak)', "hydraulic"),
    (r'\bvalve\b.*\b(block|manifold|solenoid)\b', "hydraulic"),
    (r'\bpower\s+unit\b', "hydraulic"),
    (r'\bplatform\s+lift\b', "platform"),
    (r'\bscrew\s+drive\b', "platform"),
    (r'\bvertical\s+platform\b', "platform"),
]

_DOOR_TYPE_PATTERNS = [
    (r'\bswing\s+door\b', "swing"),
    (r'\bhinged\s+door\b', "swing"),
    (r'\bmanual\s+door\b', "swing"),
    (r'\bsliding\s+door\b', "sliding"),
    (r'\bautomatic\s+door\b', "sliding"),
    (r'\bpower\s+door\b', "sliding"),
    (r'\blanding\s+door\s+operator\b', "sliding"),
]

# Components that suggest drive-type-specific content
DRIVE_SPECIFIC_COMPONENTS = {
    "Hydraulics", "Motor", "Inverter", "Drive", "Power Unit", "VSD",
    "Pump", "Ram", "Cylinder", "Valve", "Counterweight", "Rope",
}

# Components that suggest door-type-specific content
DOOR_SPECIFIC_COMPONENTS = {
    "Door Operator", "Landing Door", "Car Door",
}

# Query intents that are typically drive/door-specific
DRIVE_SPECIFIC_INTENTS = {
    "symptom_troubleshooting", "procedure", "wiring", "commissioning",
}


def detect_drive_type_from_query(query: str) -> Optional[str]:
    """Detect drive type from query text using regex patterns.

    Returns 'traction', 'hydraulic', 'platform', or None.
    """
    query_lower = query.lower()
    for pattern, drive_type in _DRIVE_TYPE_PATTERNS:
        if re.search(pattern, query_lower):
            return drive_type
    return None


def detect_door_type_from_query(query: str) -> Optional[str]:
    """Detect door type from query text using regex patterns.

    Returns 'swing', 'sliding', or None.
    """
    query_lower = query.lower()
    for pattern, door_type in _DOOR_TYPE_PATTERNS:
        if re.search(pattern, query_lower):
            return door_type
    return None


def should_clarify_drive_type(model: str, query: str, interceptor_result: dict) -> bool:
    """Determine if we should ask the user to clarify drive type.

    Only asks when:
    1. Model supports multiple drive types (ambiguous)
    2. Neither query nor interceptor detected a drive type
    3. The query is about something drive-type-specific
    """
    if not model or model not in AMBIGUOUS_DRIVE_MODELS:
        return False

    # Already detected
    if interceptor_result.get('filters', {}).get('drive_type'):
        return False
    if detect_drive_type_from_query(query):
        return False

    # Check if the query is about drive-type-specific content
    component = interceptor_result.get('filters', {}).get('component')
    intent = interceptor_result.get('query_intent')

    if component and component in DRIVE_SPECIFIC_COMPONENTS:
        return True
    if intent in DRIVE_SPECIFIC_INTENTS:
        # Check if content might be universal (e.g., COP wiring is the same for all)
        query_lower = query.lower()
        universal_terms = ['cop', 'car operating panel', 'indicator', 'display', 'button']
        if any(term in query_lower for term in universal_terms):
            return False
        return True

    return False


def should_clarify_door_type(model: str, query: str, interceptor_result: dict) -> bool:
    """Determine if we should ask the user to clarify door type.

    Only asks when the query mentions doors but no specific door type.
    """
    if not model or model not in AMBIGUOUS_DOOR_MODELS:
        return False

    # Already detected
    if interceptor_result.get('filters', {}).get('door_type'):
        return False
    if detect_door_type_from_query(query):
        return False

    # Only ask if query is about doors
    component = interceptor_result.get('filters', {}).get('component')
    query_lower = query.lower()

    if component and component in DOOR_SPECIFIC_COMPONENTS:
        return True
    if any(term in query_lower for term in ['door', 'gate', 'entrance']):
        return True

    return False


def preprocess_query(query: str) -> dict:
    """
    Preprocess a query before RAG search.

    - Normalizes voice input (e.g., "ee twenty three" -> "E23")
    - Detects fault codes and enriches them for better RAG
    - Expands query with synonyms for better coverage
    - Generates multiple synonym queries for RRF fusion
    - Classifies query type

    Returns:
        {
            "enriched_query": str,  # Query to use for RAG search (legacy, single expanded query)
            "synonym_queries": list[str],  # Multiple queries for RRF fusion
            "query_type": str,      # fault_code, procedure, specification, general
            "detected_code": str|None,  # The fault code if detected
            "original_query": str   # Original query after normalization
        }
    """
    # Normalize voice input first
    normalized_query = normalise_voice_input(query)

    detected_code = detect_fault_code(normalized_query)
    query_type = detect_query_type(normalized_query)

    if detected_code:
        # Enrich fault code query for better RAG search
        enriched_query = f"fault code {detected_code} error cause fix troubleshooting"
        synonym_queries = [enriched_query]  # No synonyms for fault codes
    else:
        # Expand with synonyms for better search coverage
        enriched_query = _expand_query_synonyms(normalized_query)
        # Generate separate synonym queries for RRF (each gets full BM25 weight)
        synonym_queries = generate_synonym_queries(normalized_query, max_queries=3)

    return {
        "enriched_query": enriched_query,
        "synonym_queries": synonym_queries,
        "query_type": query_type,
        "detected_code": detected_code,
        "original_query": normalized_query
    }


def process_query(
    user_id: str,
    query: str,
    lift_model: str = None,
    image_path: str = None,
    previous_context: str = None,
    drive_type: str = None,
    door_type: str = None
) -> dict:
    """
    Core LiftMind logic - used by any interface.

    Args:
        user_id: User identifier (Telegram ID or external system ID)
        query: The user's question
        lift_model: Optional lift model filter for RAG
        image_path: Optional path to an image file
        drive_type: Optional drive type filter (traction/hydraulic/platform)
        door_type: Optional door type filter (swing/sliding)

    Returns:
        {
            "response": "Claude's answer...",
            "sources": ["manual.pdf p.22", ...],
            "model_used": "Elfo Traction",
            "response_time_ms": 1234,
            "rag_results_count": 5
        }
    """
    # Handle slash commands first (no RAG/Claude needed)
    if query.startswith('/'):
        cmd_result = handle_command(user_id, query)
        if cmd_result:
            return {
                "response": cmd_result["message"],
                "sources": [],
                "model_used": None,
                "model_auto_detected": False,
                "response_time_ms": 0,
                "rag_results_count": 0,
                "query_type": "command",
                "detected_code": None,
                "images": []
            }

    start_time = time.time()

    # Preprocess the query
    preprocessed = preprocess_query(query)
    enriched_query = preprocessed["enriched_query"]
    query_type = preprocessed["query_type"]
    detected_code = preprocessed["detected_code"]

    # Resolve the model name to canonical form
    resolved_model = resolve_model(lift_model) if lift_model else None
    model_auto_detected = False
    model_from_state = False

    # Auto-detect model from query text if not provided
    if not resolved_model:
        detected_model = detect_model_from_query(query)
        if detected_model:
            resolved_model = detected_model
            model_auto_detected = True
            logger.info(f"Auto-detected model '{resolved_model}' from query")

    # Fall back to user's previously selected model
    if not resolved_model:
        stored_model = get_user_model(user_id)
        if stored_model:
            resolved_model = stored_model
            model_from_state = True
            logger.info(f"Using stored model '{resolved_model}' for user {user_id}")

    # --- Step 9: Check response cache ---
    cached = get_cached_response(query, resolved_model)
    if cached and not image_path:
        cached["response_time_ms"] = int((time.time() - start_time) * 1000)
        cached["from_cache"] = True
        logger.info(f"Cache hit for user {user_id}: {query[:50]}... ({cached['response_time_ms']}ms)")
        return cached

    # --- Semantic similarity cache check ---
    if SEMANTIC_CACHE_AVAILABLE and not image_path:
        sem_cached = get_semantic_cached_response(query, resolved_model)
        if sem_cached:
            sem_cached["response_time_ms"] = int((time.time() - start_time) * 1000)
            return sem_cached

    # Determine RAG limit based on query type
    rag_limits = {
        "fault_code": 10,     # Need multiple results for error context
        "specification": 10,  # May need cross-references
        "procedure": 10,      # May span multiple pages
        "general": 12         # More context is better for accuracy
    }
    rag_limit = rag_limits.get(query_type, 8)

    # --- Step 1: Non-blocking interceptor ---
    # Get a deterministic fallback immediately (<1ms).
    # RAG search proceeds immediately using this -- the interceptor runs in parallel.
    interceptor_result = _create_fallback_response(query, resolved_model)

    # Fire the Haiku API call in a background thread regardless of key availability.
    # Do NOT block here -- RAG runs immediately below using the fallback.
    _interceptor_future: Optional[Future] = _interceptor_executor.submit(
        intercept_query_sync, query, resolved_model, previous_context
    )
    # Alias for compatibility with downstream checks
    cli_future = _interceptor_future

    # Use interceptor's model detection if we don't have one yet
    # Track full list of model variants for search (e.g. ["Elfo", "Elfo 2", "Elfo Traction"])
    search_models = [resolved_model] if resolved_model else None
    interceptor_models = interceptor_result.get('filters', {}).get('model')
    if not resolved_model and interceptor_models:
        if isinstance(interceptor_models, list):
            resolved_model = interceptor_models[0]  # Display/state uses first
            search_models = interceptor_models       # Search uses ALL variants
        else:
            resolved_model = interceptor_models
            search_models = [interceptor_models]
        model_auto_detected = True
        logger.info(f"Interceptor detected model: {resolved_model} (search_models: {search_models})")

    # Check for error code from interceptor
    interceptor_error = interceptor_result.get('filters', {}).get('error_code')
    if interceptor_error and not detected_code:
        detected_code = interceptor_error
        query_type = "fault_code"

    # --- Exact-match wiki bypass (fault codes, parameters, parts) ---
    wiki_context = None
    _bypass_models = [resolved_model] if resolved_model else None
    if WIKI_AVAILABLE:
        if detected_code:
            wiki_context = _lookup_wiki_bypass(detected_code, "error_code", _bypass_models)
            if wiki_context:
                logger.info(f"Wiki bypass hit for error_code/{detected_code}: {len(wiki_context)} chars")

        if not wiki_context:
            # Try parameter lookup for bare parameter codes (e.g. L65, P12)
            query_bare = query.strip()
            if len(query_bare) <= 8:
                m = _PARAM_CODE_RE.match(query_bare)
                if m:
                    param_id = m.group(1).upper()
                    wiki_context = _lookup_wiki_bypass(param_id, "parameter", _bypass_models)
                    if wiki_context:
                        logger.info(f"Wiki bypass hit for parameter/{param_id}: {len(wiki_context)} chars")

        if not wiki_context:
            # Try Nidec/dotted-numeric parameter codes (e.g. "Pr 0.024", "5.000", "06.010")
            query_bare = query.strip()
            if len(query_bare) <= 10:
                m = _NIDEC_PARAM_RE.match(query_bare)
                if m:
                    param_id = m.group(1)
                    wiki_context = _lookup_wiki_bypass(param_id, "parameter", _bypass_models)
                    if not wiki_context:
                        wiki_context = _lookup_wiki_bypass(param_id, "setting", _bypass_models)
                    if wiki_context:
                        logger.info(f"Wiki bypass hit for nidec-param/{param_id}: {len(wiki_context)} chars")

        if not wiki_context:
            # Try part/component lookup from interceptor-detected component
            component = interceptor_result.get('filters', {}).get('component') if interceptor_result else None
            if component:
                wiki_context = _lookup_wiki_bypass(component, "part", _bypass_models)
                if not wiki_context:
                    wiki_context = _lookup_wiki_bypass(component, "part_number", _bypass_models)
                if wiki_context:
                    logger.info(f"Wiki bypass hit for part/{component}: {len(wiki_context)} chars")

        if not wiki_context:
            # Try spec lookup for short bare identifiers that look like spec codes
            query_bare = query.strip()
            if len(query_bare) <= 12 and _PART_CODE_RE.match(query_bare):
                spec_id = query_bare.upper()
                wiki_context = _lookup_wiki_bypass(spec_id, "spec", _bypass_models)
                if not wiki_context:
                    wiki_context = _lookup_wiki_bypass(spec_id, "part_number", _bypass_models)
                if wiki_context:
                    logger.info(f"Wiki bypass hit for spec/{spec_id}: {len(wiki_context)} chars")

        if not wiki_context:
            # Try setting lookup for bare setting identifiers (e.g. "d01", "06.010")
            query_bare = query.strip()
            if len(query_bare) <= 10:
                setting_id = query_bare
                wiki_context = _lookup_wiki_bypass(setting_id, "setting", _bypass_models)
                if wiki_context:
                    logger.info(f"Wiki bypass hit for setting/{setting_id}: {len(wiki_context)} chars")

        if not wiki_context:
            # Try terminal lookup for bare terminal identifiers (e.g. "X1A", "XT10")
            query_bare = query.strip()
            if len(query_bare) <= 8:
                terminal_id = query_bare.upper()
                wiki_context = _lookup_wiki_bypass(terminal_id, "terminal", _bypass_models)
                if wiki_context:
                    logger.info(f"Wiki bypass hit for terminal/{terminal_id}: {len(wiki_context)} chars")

    # --- Drive/door type detection: interceptor result + regex fallback + user override ---
    if not interceptor_result.get('filters', {}).get('drive_type'):
        detected_drive = detect_drive_type_from_query(query)
        if detected_drive:
            interceptor_result['filters']['drive_type'] = detected_drive
            logger.info(f"Regex detected drive_type: {detected_drive}")

    if not interceptor_result.get('filters', {}).get('door_type'):
        detected_door = detect_door_type_from_query(query)
        if detected_door:
            interceptor_result['filters']['door_type'] = detected_door
            logger.info(f"Regex detected door_type: {detected_door}")

    # User-provided type overrides (from bot clarification buttons)
    if drive_type and not interceptor_result.get('filters', {}).get('drive_type'):
        interceptor_result['filters']['drive_type'] = drive_type
        logger.info(f"User-provided drive_type: {drive_type}")
    if door_type and not interceptor_result.get('filters', {}).get('door_type'):
        interceptor_result['filters']['door_type'] = door_type
        logger.info(f"User-provided door_type: {door_type}")

    # Merge exact_terms into keyword_queries (user's literal phrases first)
    exact_terms = interceptor_result.get('exact_terms', [])
    if exact_terms:
        kw = interceptor_result.get('keyword_queries', [])
        interceptor_result['keyword_queries'] = exact_terms + [k for k in kw if k not in exact_terms]

    # Supplement interceptor keywords with technical terms from original query
    # This catches terms the interceptor missed (e.g. "PLC input", "blinking")
    interceptor_result['keyword_queries'] = _supplement_keywords(
        query, interceptor_result.get('keyword_queries', [])
    )

    # Use interceptor's query_intent for better classification
    interceptor_intent = interceptor_result.get('query_intent')
    if interceptor_intent:
        intent_mapping = {
            'fault_code': 'fault_code',
            'symptom_troubleshooting': 'general',  # Needs semantic weight
            'procedure': 'procedure',
            'specification': 'specification',
            'wiring': 'specification',             # Exact match matters
            'commissioning': 'procedure',
            'general': 'general',
        }
        query_type = intent_mapping.get(interceptor_intent, query_type)
        # Write back so search_documents_hybrid can use it
        interceptor_result['query_type'] = query_type
        logger.info(f"Interceptor query_intent: {interceptor_intent} -> query_type: {query_type}")

    # Build search query - use interceptor's semantic query if available
    search_query = enriched_query
    if interceptor_result and interceptor_result.get('semantic_query'):
        search_query = interceptor_result['semantic_query']

    # If wiki bypass found an article, cap RAG results (save ~100ms)
    effective_rag_limit = 5 if wiki_context else rag_limit

    # RAG search - use hybrid search if interceptor available, fallback to legacy
    try:
        if interceptor_result:
            # Use full model list for search (all variants, not just first)
            if search_models:
                if 'filters' not in interceptor_result:
                    interceptor_result['filters'] = {}
                interceptor_result['filters']['model'] = search_models

            results = search_documents_hybrid(interceptor_result, limit=effective_rag_limit)
            logger.info(f"Hybrid search returned {len(results)} results")
        else:
            results = search_documents(search_query, lift_model=resolved_model, limit=effective_rag_limit)

        image_results = search_images(search_query, lift_model=resolved_model)

        # --- Step 1 continued: Check if CLI interceptor finished with better results ---
        if cli_future and cli_future.done():
            try:
                better_result = cli_future.result(timeout=0)
                if better_result and better_result.get('keyword_queries'):
                    logger.info("Upgraded interceptor result from background CLI")
                    interceptor_result = better_result
                    # Re-run RAG with better queries, always preserving user's model
                    if resolved_model:
                        if 'filters' not in interceptor_result:
                            interceptor_result['filters'] = {}
                        interceptor_result['filters']['model'] = [resolved_model]
                    results = search_documents_hybrid(interceptor_result, limit=rag_limit)
                    search_query = interceptor_result.get('semantic_query', search_query)
                    image_results = search_images(search_query, lift_model=resolved_model)
                    # Update model/code detection from better result (only if user didn't specify)
                    better_models = better_result.get('filters', {}).get('model')
                    if not resolved_model and better_models:
                        if isinstance(better_models, list):
                            resolved_model = better_models[0]
                            search_models = better_models
                        else:
                            resolved_model = better_models
                            search_models = [better_models]
                        model_auto_detected = True
                    better_code = better_result.get('filters', {}).get('error_code')
                    if better_code and not detected_code:
                        detected_code = better_code
                        query_type = "fault_code"
            except Exception:
                pass  # Stick with fallback results

        # --- Direct PDF fallback when DB has low confidence ---
        _used_direct_pdf = False
        if MANUAL_READER_AVAILABLE and len(results) < 2:
            logger.info(f"Low confidence from DB ({len(results)} results), triggering direct PDF read")
            try:
                manual_results = search_manuals_direct(
                    query, model_filter=resolved_model, max_results=6
                )
                if manual_results:
                    logger.info(f"Direct PDF reader found {len(manual_results)} results")
                    # Merge: DB results first, then direct PDF results
                    seen_keys = {(r.get('filename'), r.get('page_number')) for r in results}
                    for mr in manual_results:
                        key = (mr.get('filename'), mr.get('page_number'))
                        if key not in seen_keys:
                            results.append(mr)
                            seen_keys.add(key)
                    _used_direct_pdf = True
            except Exception as pdf_err:
                logger.error(f"Direct PDF read failed: {pdf_err}")

        # Extract verified fix IDs BEFORE format_context() pops them
        verified_fix_ids = []
        if results:
            vf = results[0].get('_verified_fixes', [])
            verified_fix_ids = [f['id'] for f in vf]

        context = format_context(
            results, image_results=image_results,
            query=query, lift_model=resolved_model
        )
        if wiki_context:
            context = f"## Wiki Article (pre-compiled reference)\n{wiki_context}\n\n---\n\n{context}"
    except Exception as e:
        logger.error(f"RAG search error for user {user_id}: {e}")
        results = []
        image_results = []
        context = ""
        _used_direct_pdf = False
        verified_fix_ids = []

    # Build sources list
    sources = []
    if results:
        seen = set()
        for r in results[:5]:
            source = f"{r['filename']} p.{r['page_number']}"
            if source not in seen:
                sources.append(source)
                seen.add(source)

    # Check for deep dive mode (full manual context)
    deep_dive = interceptor_result.get('deep_dive', False) if interceptor_result else False
    response = None

    if deep_dive and resolved_model and not image_path:
        logger.info(f"Deep dive mode activated for {resolved_model}")
        try:
            dd_response = deep_dive_query(query, resolved_model, ask_claude)
            if dd_response and not dd_response.startswith("Sorry,"):
                response = dd_response
                sources = [f"Full {resolved_model} Manual (Deep Dive Mode)"]
            else:
                logger.warning(f"Deep dive returned error, falling back to RAG")
        except Exception as e:
            logger.warning(f"Deep dive failed: {e}, falling back to RAG")

    # --- Step 6: Route to appropriate Claude model based on query type ---
    claude_model = settings.CLAUDE_MODEL_ROUTING.get(query_type)

    # Query Claude (normal mode or deep dive fallback)
    if response is None:
        try:
            if image_path:
                response = ask_claude_with_image(
                    query, image_path, context=context,
                    lift_model=resolved_model, rag_results_count=len(results),
                    model=claude_model
                )
            else:
                response = ask_claude(
                    query, context=context,
                    lift_model=resolved_model, rag_results_count=len(results),
                    model=claude_model
                )
        except Exception as e:
            logger.error(f"Claude query error for user {user_id}: {e}")
            response = "Sorry, I encountered an error processing your request. Please try again."

    response_time_ms = int((time.time() - start_time) * 1000)

    # Persist model selection and increment query count
    try:
        if resolved_model and not model_from_state:
            set_user_model(user_id, resolved_model)
        increment_query_count(user_id)
    except Exception as e:
        logger.error(f"Error persisting user state: {e}")

    logger.info(f"Processed query for user {user_id}: {query[:50]}... ({response_time_ms}ms)")

    # Build image references for the interface to send
    images = [
        {
            "filename": r["filename"],
            "path": r["full_path"],
            "description": r["description"]
        }
        for r in image_results
    ] if image_results else []

    # Extract fact IDs from RAG results for feedback tracking
    fact_ids = [r.get("id") for r in results if r.get("id") is not None]

    result = {
        "response": response,
        "sources": sources,
        "model_used": resolved_model,
        "model_auto_detected": model_auto_detected,
        "response_time_ms": response_time_ms,
        "rag_results_count": len(results),
        "query_type": query_type,
        "detected_code": detected_code,
        "images": images,
        "fact_ids": fact_ids,
        "verified_fix_ids": verified_fix_ids,
        "from_cache": False
    }

    # Log query for analytics (fire-and-forget)
    try:
        log_query(user_id, query, result)
    except Exception as e:
        logger.error(f"Analytics logging failed: {e}")

    # --- Self-learning: if direct PDF read found useful content, learn from it ---
    if _used_direct_pdf and LEARNING_AVAILABLE and response and not response.startswith("Sorry"):
        try:
            direct_results = [r for r in results if r.get("source") == "direct_pdf_read"]
            if direct_results:
                from concurrent.futures import ThreadPoolExecutor as _TPE
                _learn_executor = _TPE(max_workers=1, thread_name_prefix="learn")
                _learn_executor.submit(
                    learn_from_direct_read, query, direct_results, resolved_model
                )
                logger.info(f"Triggered self-learning from {len(direct_results)} direct PDF results")
        except Exception as learn_err:
            logger.error(f"Self-learning trigger failed: {learn_err}")

    # --- Step 9: Store in response cache ---
    if not image_path and response and not response.startswith("Sorry"):
        try:
            store_response(query, resolved_model, result)
        except Exception as e:
            logger.error(f"Cache store failed: {e}")
        if SEMANTIC_CACHE_AVAILABLE:
            store_semantic_response(query, resolved_model, result)

    # Store the CLI future reference for telegram_bot to check later
    result["_cli_future"] = cli_future

    return result


def process_query_streaming(
    user_id: str,
    query: str,
    lift_model: str = None,
    on_chunk=None,
    previous_context: str = None,
    drive_type: str = None,
    door_type: str = None
) -> dict:
    """
    Streaming version of process_query. Calls on_chunk(partial_text) as tokens arrive.

    Returns the same result dict as process_query().
    on_chunk is called from the same thread with cumulative text so far.
    Not used for image queries (no streaming for vision).
    """
    start_time = time.time()

    # Preprocess
    preprocessed = preprocess_query(query)
    enriched_query = preprocessed["enriched_query"]
    query_type = preprocessed["query_type"]
    detected_code = preprocessed["detected_code"]

    resolved_model = resolve_model(lift_model) if lift_model else None
    model_auto_detected = False
    model_from_state = False

    if not resolved_model:
        detected_model = detect_model_from_query(query)
        if detected_model:
            resolved_model = detected_model
            model_auto_detected = True

    if not resolved_model:
        stored_model = get_user_model(user_id)
        if stored_model:
            resolved_model = stored_model
            model_from_state = True

    # Check cache
    cached = get_cached_response(query, resolved_model)
    if cached:
        cached["response_time_ms"] = int((time.time() - start_time) * 1000)
        cached["from_cache"] = True
        if on_chunk:
            on_chunk(cached.get("response", ""))
        return cached

    # RAG limits
    rag_limits = {"fault_code": 10, "specification": 10, "procedure": 10, "general": 12}
    rag_limit = rag_limits.get(query_type, 8)

    # Fast-start interceptor
    interceptor_result = _create_fallback_response(query, resolved_model)
    cli_future: Optional[Future] = None

    if settings.ANTHROPIC_API_KEY:
        try:
            interceptor_result = intercept_query_sync(query, resolved_model, previous_context)
        except Exception:
            pass
    else:
        cli_future = _interceptor_executor.submit(intercept_query_sync, query, resolved_model, previous_context)
        try:
            cli_result = cli_future.result(timeout=18)
            if cli_result and cli_result.get('keyword_queries'):
                interceptor_result = cli_result
                cli_future = None
        except Exception:
            pass  # Use improved fallback keywords

    # Use interceptor results
    # Track full list of model variants for search
    search_models = [resolved_model] if resolved_model else None
    interceptor_models = interceptor_result.get('filters', {}).get('model')
    if not resolved_model and interceptor_models:
        if isinstance(interceptor_models, list):
            resolved_model = interceptor_models[0]  # Display/state uses first
            search_models = interceptor_models       # Search uses ALL variants
        else:
            resolved_model = interceptor_models
            search_models = [interceptor_models]
        model_auto_detected = True

    interceptor_error = interceptor_result.get('filters', {}).get('error_code')
    if interceptor_error and not detected_code:
        detected_code = interceptor_error
        query_type = "fault_code"

    # --- Exact-match wiki bypass (fault codes, parameters, parts, settings, terminals) ---
    wiki_context = None
    _bypass_models = [resolved_model] if resolved_model else None
    if WIKI_AVAILABLE:
        if detected_code:
            wiki_context = _lookup_wiki_bypass(detected_code, "error_code", _bypass_models)
            if wiki_context:
                logger.info(f"Wiki bypass hit (streaming) for error_code/{detected_code}: {len(wiki_context)} chars")

        if not wiki_context:
            query_bare = query.strip()
            if len(query_bare) <= 8:
                m = _PARAM_CODE_RE.match(query_bare)
                if m:
                    param_id = m.group(1).upper()
                    wiki_context = _lookup_wiki_bypass(param_id, "parameter", _bypass_models)
                    if wiki_context:
                        logger.info(f"Wiki bypass hit (streaming) for parameter/{param_id}: {len(wiki_context)} chars")

        if not wiki_context:
            query_bare = query.strip()
            if len(query_bare) <= 10:
                m = _NIDEC_PARAM_RE.match(query_bare)
                if m:
                    param_id = m.group(1)
                    wiki_context = _lookup_wiki_bypass(param_id, "parameter", _bypass_models)
                    if not wiki_context:
                        wiki_context = _lookup_wiki_bypass(param_id, "setting", _bypass_models)
                    if wiki_context:
                        logger.info(f"Wiki bypass hit (streaming) for nidec-param/{param_id}: {len(wiki_context)} chars")

        if not wiki_context:
            component = interceptor_result.get('filters', {}).get('component') if interceptor_result else None
            if component:
                wiki_context = _lookup_wiki_bypass(component, "part", _bypass_models)
                if not wiki_context:
                    wiki_context = _lookup_wiki_bypass(component, "part_number", _bypass_models)
                if wiki_context:
                    logger.info(f"Wiki bypass hit (streaming) for part/{component}: {len(wiki_context)} chars")

        if not wiki_context:
            query_bare = query.strip()
            if len(query_bare) <= 12 and _PART_CODE_RE.match(query_bare):
                spec_id = query_bare.upper()
                wiki_context = _lookup_wiki_bypass(spec_id, "spec", _bypass_models)
                if not wiki_context:
                    wiki_context = _lookup_wiki_bypass(spec_id, "part_number", _bypass_models)
                if wiki_context:
                    logger.info(f"Wiki bypass hit (streaming) for spec/{spec_id}: {len(wiki_context)} chars")

        if not wiki_context:
            query_bare = query.strip()
            if len(query_bare) <= 10:
                setting_id = query_bare
                wiki_context = _lookup_wiki_bypass(setting_id, "setting", _bypass_models)
                if wiki_context:
                    logger.info(f"Wiki bypass hit (streaming) for setting/{setting_id}: {len(wiki_context)} chars")

        if not wiki_context:
            query_bare = query.strip()
            if len(query_bare) <= 8:
                terminal_id = query_bare.upper()
                wiki_context = _lookup_wiki_bypass(terminal_id, "terminal", _bypass_models)
                if wiki_context:
                    logger.info(f"Wiki bypass hit (streaming) for terminal/{terminal_id}: {len(wiki_context)} chars")

    # --- Drive/door type detection: interceptor result + regex fallback + user override ---
    if not interceptor_result.get('filters', {}).get('drive_type'):
        detected_drive = detect_drive_type_from_query(query)
        if detected_drive:
            interceptor_result['filters']['drive_type'] = detected_drive
            logger.info(f"Regex detected drive_type: {detected_drive}")

    if not interceptor_result.get('filters', {}).get('door_type'):
        detected_door = detect_door_type_from_query(query)
        if detected_door:
            interceptor_result['filters']['door_type'] = detected_door
            logger.info(f"Regex detected door_type: {detected_door}")

    # User-provided type overrides (from bot clarification buttons)
    if drive_type and not interceptor_result.get('filters', {}).get('drive_type'):
        interceptor_result['filters']['drive_type'] = drive_type
        logger.info(f"User-provided drive_type: {drive_type}")
    if door_type and not interceptor_result.get('filters', {}).get('door_type'):
        interceptor_result['filters']['door_type'] = door_type
        logger.info(f"User-provided door_type: {door_type}")

    # Merge exact_terms into keyword_queries (user's literal phrases first)
    exact_terms = interceptor_result.get('exact_terms', [])
    if exact_terms:
        kw = interceptor_result.get('keyword_queries', [])
        interceptor_result['keyword_queries'] = exact_terms + [k for k in kw if k not in exact_terms]

    # Supplement interceptor keywords with technical terms from original query
    interceptor_result['keyword_queries'] = _supplement_keywords(
        query, interceptor_result.get('keyword_queries', [])
    )

    # Use interceptor's query_intent for better classification
    interceptor_intent = interceptor_result.get('query_intent')
    if interceptor_intent:
        intent_mapping = {
            'fault_code': 'fault_code',
            'symptom_troubleshooting': 'general',
            'procedure': 'procedure',
            'specification': 'specification',
            'wiring': 'specification',
            'commissioning': 'procedure',
            'general': 'general',
        }
        query_type = intent_mapping.get(interceptor_intent, query_type)
        interceptor_result['query_type'] = query_type
        logger.info(f"Interceptor query_intent: {interceptor_intent} -> query_type: {query_type}")

    search_query = enriched_query
    if interceptor_result and interceptor_result.get('semantic_query'):
        search_query = interceptor_result['semantic_query']

    # If wiki bypass found an article, cap RAG results (save latency)
    effective_rag_limit = 5 if wiki_context else rag_limit

    # RAG search
    try:
        if interceptor_result:
            if search_models:
                if 'filters' not in interceptor_result:
                    interceptor_result['filters'] = {}
                interceptor_result['filters']['model'] = search_models
            results = search_documents_hybrid(interceptor_result, limit=effective_rag_limit)
        else:
            results = search_documents(search_query, lift_model=resolved_model, limit=effective_rag_limit)

        image_results = search_images(search_query, lift_model=resolved_model)

        # Check CLI future
        if cli_future and cli_future.done():
            try:
                better_result = cli_future.result(timeout=0)
                if better_result and better_result.get('keyword_queries'):
                    interceptor_result = better_result
                    if search_models:
                        if 'filters' not in interceptor_result:
                            interceptor_result['filters'] = {}
                        interceptor_result['filters']['model'] = search_models
                    results = search_documents_hybrid(interceptor_result, limit=rag_limit)
                    search_query = interceptor_result.get('semantic_query', search_query)
                    image_results = search_images(search_query, lift_model=resolved_model)
                    # Update model detection from better result
                    better_models = better_result.get('filters', {}).get('model')
                    if not resolved_model and better_models:
                        if isinstance(better_models, list):
                            resolved_model = better_models[0]
                            search_models = better_models
                        else:
                            resolved_model = better_models
                            search_models = [better_models]
                        model_auto_detected = True
            except Exception:
                pass

        # --- Direct PDF fallback when DB has low confidence ---
        _used_direct_pdf = False
        if MANUAL_READER_AVAILABLE and len(results) < 2:
            logger.info(f"Low confidence from DB ({len(results)} results), triggering direct PDF read (streaming)")
            try:
                manual_results = search_manuals_direct(
                    query, model_filter=resolved_model, max_results=6
                )
                if manual_results:
                    logger.info(f"Direct PDF reader found {len(manual_results)} results (streaming)")
                    seen_keys = {(r.get('filename'), r.get('page_number')) for r in results}
                    for mr in manual_results:
                        key = (mr.get('filename'), mr.get('page_number'))
                        if key not in seen_keys:
                            results.append(mr)
                            seen_keys.add(key)
                    _used_direct_pdf = True
            except Exception as pdf_err:
                logger.error(f"Direct PDF read failed (streaming): {pdf_err}")

        # Extract verified fix IDs BEFORE format_context() pops them
        verified_fix_ids = []
        if results:
            vf = results[0].get('_verified_fixes', [])
            verified_fix_ids = [f['id'] for f in vf]

        context = format_context(results, image_results=image_results, query=query, lift_model=resolved_model)
        if wiki_context:
            context = f"## Wiki Article (pre-compiled reference)\n{wiki_context}\n\n---\n\n{context}"
    except Exception as e:
        logger.error(f"RAG search error for user {user_id}: {e}")
        results = []
        image_results = []
        context = ""
        _used_direct_pdf = False
        verified_fix_ids = []

    sources = []
    if results:
        seen = set()
        for r in results[:5]:
            source = f"{r['filename']} p.{r['page_number']}"
            if source not in seen:
                sources.append(source)
                seen.add(source)

    # Model routing
    claude_model = settings.CLAUDE_MODEL_ROUTING.get(query_type)

    # Stream Claude response
    response = ""
    try:
        for chunk in ask_claude_streaming(
            query, context=context,
            lift_model=resolved_model, rag_results_count=len(results),
            model=claude_model
        ):
            response = chunk
            if on_chunk:
                on_chunk(chunk)
    except Exception as e:
        logger.error(f"Streaming Claude error for user {user_id}: {e}")
        if not response:
            response = "Sorry, I encountered an error processing your request. Please try again."

    response_time_ms = int((time.time() - start_time) * 1000)

    try:
        if resolved_model and not model_from_state:
            set_user_model(user_id, resolved_model)
        increment_query_count(user_id)
    except Exception as e:
        logger.error(f"Error persisting user state: {e}")

    logger.info(f"Processed streaming query for user {user_id}: {query[:50]}... ({response_time_ms}ms)")

    images = [
        {"filename": r["filename"], "path": r["full_path"], "description": r["description"]}
        for r in image_results
    ] if image_results else []

    fact_ids = [r.get("id") for r in results if r.get("id") is not None]

    result = {
        "response": response,
        "sources": sources,
        "model_used": resolved_model,
        "model_auto_detected": model_auto_detected,
        "response_time_ms": response_time_ms,
        "rag_results_count": len(results),
        "query_type": query_type,
        "detected_code": detected_code,
        "images": images,
        "fact_ids": fact_ids,
        "verified_fix_ids": verified_fix_ids,
        "from_cache": False
    }

    try:
        log_query(user_id, query, result)
    except Exception:
        pass

    # --- Self-learning: if direct PDF read found useful content, learn from it ---
    if _used_direct_pdf and LEARNING_AVAILABLE and response and not response.startswith("Sorry"):
        try:
            direct_results = [r for r in results if r.get("source") == "direct_pdf_read"]
            if direct_results:
                from concurrent.futures import ThreadPoolExecutor as _TPE
                _learn_executor = _TPE(max_workers=1, thread_name_prefix="learn")
                _learn_executor.submit(
                    learn_from_direct_read, query, direct_results, resolved_model
                )
                logger.info(f"Triggered self-learning from {len(direct_results)} direct PDF results (streaming)")
        except Exception as learn_err:
            logger.error(f"Self-learning trigger failed (streaming): {learn_err}")

    if response and not response.startswith("Sorry"):
        try:
            store_response(query, resolved_model, result)
        except Exception:
            pass

    return result


def format_response_with_sources(result: dict, include_sources: bool = True) -> str:
    """
    Format the process_query result into a user-friendly message.

    Only shows sources if Claude actually cited them in the response.

    Args:
        result: The dict returned by process_query()
        include_sources: Whether to append source references

    Returns:
        Formatted string suitable for sending to user
    """
    response = result.get("response", "")

    if not include_sources:
        return response

    # Only show sources if Claude actually cited documents in the response
    # Only look for actual citation format [Source: ...]
    import re
    has_citations = bool(re.search(r'\[Source:\s*\w+.*?\]', response))

    if not has_citations:
        return response

    sources = result.get("sources", [])
    if sources:
        # Add sources section
        sources_text = "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources[:5])
        response += sources_text

    return response


def get_available_models() -> list:
    """
    Get list of available lift models.

    Returns:
        Sorted list of valid model names
    """
    return sorted(VALID_MODELS)


def handle_command(user_id: str, command: str) -> Optional[dict]:
    """
    Handle slash commands.

    Supported commands:
    - /model [name] - Set the current lift model
    - /model - Show current model and available options
    - /help - Show usage summary

    Args:
        user_id: User identifier
        command: The full command string (e.g., "/model Elfo Traction")

    Returns:
        {"type": "command_response", "message": str} or None if not a recognized command
    """
    if not command.startswith('/'):
        return None

    parts = command.split(None, 1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if cmd == '/model':
        if args:
            # Set model
            resolved = resolve_model(args)
            if resolved:
                set_user_model(user_id, resolved)
                return {
                    "type": "command_response",
                    "message": f"Model set to: **{resolved}**"
                }
            else:
                return {
                    "type": "command_response",
                    "message": f"Unknown model: '{args}'\n\nAvailable models:\n" +
                              "\n".join(f"\u2022 {m}" for m in sorted(VALID_MODELS))
                }
        else:
            # Show current model
            current = get_user_model(user_id)
            models_list = "\n".join(f"\u2022 {m}" for m in sorted(VALID_MODELS))
            if current:
                return {
                    "type": "command_response",
                    "message": f"Current model: **{current}**\n\nAvailable models:\n{models_list}"
                }
            else:
                return {
                    "type": "command_response",
                    "message": f"No model selected.\n\nSet one with: `/model [name]`\n\nAvailable models:\n{models_list}"
                }

    elif cmd == '/help':
        return {
            "type": "command_response",
            "message": """**LiftMind Commands**

`/model [name]` - Set your lift model
`/model` - Show current model and options
`/help` - Show this help

**Tips:**
- Ask fault codes directly: `E23`
- Include model in your question: "Elfo Traction commissioning steps"
- Send photos for wiring checks"""
        }

    # Unknown command - let it pass through as a query
    return None


def detect_model_family_mention(query: str) -> Optional[dict]:
    """
    Detect if user mentioned a model family that needs clarification.

    Returns dict with family info if clarification needed, None otherwise.

    Example: "working on an elfo" -> returns Elfo family options
             "working on elfo traction" -> returns None (specific model)
    """
    query_lower = query.lower()

    # Model families with their trigger words and specific models
    families = {
        "Elfo": {
            "triggers": ["elfo", "e3"],
            "specific": ["elfo traction", "elfo 2", "elfo cabin", "elfo electronic",
                        "elfo hydraulic", "elfo hydro", "e3"],
            "models": ["Elfo", "Elfo 2", "E3", "Elfo Cabin",
                      "Elfo Electronic", "Elfo Hydraulic controller", "Elfo Traction"]
        },
        "Supermec": {
            "triggers": ["supermec", "sm2", "sm3"],
            "specific": ["supermec 2", "supermec 3", "supermec2", "supermec3", "sm2", "sm3"],
            "models": ["Supermec", "Supermec 2", "Supermec 3"]
        },
        "Freedom": {
            "triggers": ["freedom"],
            "specific": ["freedom maxi", "freedom step"],
            "models": ["Freedom", "Freedom MAXI", "Freedom STEP"]
        },
        "Pollock": {
            "triggers": ["pollock", "p1", "q1"],
            "specific": ["pollock p1", "pollock q1", "p1", "q1"],
            "models": ["Pollock (P1)", "Pollock (Q1)"]
        }
    }

    for family_name, family_data in families.items():
        # Check if any trigger word is in the query
        has_trigger = any(trigger in query_lower for trigger in family_data["triggers"])
        if not has_trigger:
            continue

        # Check if a specific model is already mentioned
        has_specific = any(specific in query_lower for specific in family_data["specific"])
        if has_specific:
            return None  # Specific model mentioned, no clarification needed

        # Family mentioned but not specific model - needs clarification
        return {
            "family": family_name,
            "models": family_data["models"],
            "message": f"Which {family_name} model are you working on?"
        }

    return None


def get_model_selector_data() -> dict:
    """
    Get structured model data for building UI selectors.

    Returns a grouped structure suitable for Telegram inline keyboards,
    web UIs, or any other interface.

    Returns:
        {
            "groups": [
                {"name": "Elfo", "models": [...]},
                ...
            ],
            "all_models": [sorted list],
            "aliases": MODEL_ALIASES
        }
    """
    groups = [
        {
            "name": "Elfo",
            "models": ["Elfo", "Elfo 2", "E3", "Elfo Cabin",
                      "Elfo Electronic", "Elfo Hydraulic controller", "Elfo Traction"]
        },
        {
            "name": "Supermec",
            "models": ["Supermec", "Supermec 2", "Supermec 3"]
        },
        {
            "name": "Freedom",
            "models": ["Freedom", "Freedom MAXI", "Freedom STEP"]
        },
        {
            "name": "Pollock",
            "models": ["Pollock (P1)", "Pollock (Q1)"]
        },
        {
            "name": "Other",
            "models": ["Bari", "P4", "Tresa"]
        }
    ]

    return {
        "groups": groups,
        "all_models": sorted(VALID_MODELS),
        "aliases": MODEL_ALIASES
    }
