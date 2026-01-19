from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple
import re
import hashlib
import json

import numpy as np
import inflect
import spacy
import requests
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ENDPOINT_URL, SHAPES_GRAPH_URI, VALIDATION_REPORT_URI

"""
Summary Service Module

This module provides natural language generation (NLG) functionality for summarizing
SHACL validation results. It transforms raw validation data into human-readable
natural language summaries using template-based generation, rule-based logic, and
optional LLM (Large Language Model) enhancement.

The module implements a comprehensive NLG pipeline that:
1. Analyzes violation patterns (distributions, clusters, outliers)
2. Processes labels using NLP techniques (lemmatization, semantic categorization)
3. Selects appropriate templates based on data patterns
4. Generates natural language summaries at different detail levels (high/low)
5. Optionally uses LLM for flexible, domain-specific category detection

Key functions:

Label Processing & Classification:
- clean_label: Convert IRIs/camelCase to readable labels
- analyze_label_semantics: Classify labels into semantic categories (entity, temporal, spatial, etc.)
- dominant_category: Find dominant category using keyword-based classification
- dominant_category_llm: Find dominant category using LLM (optional, adds latency)

Home View Summaries:
- summarize_nodeshape_with_templates: Summarize violations per node shape histogram
- summarize_path_with_templates: Summarize violations per path histogram
- summarize_focusnode_with_templates: Summarize violations per focus node histogram
- summarize_constraint_with_templates: Summarize violations per constraint component histogram
# (removed) home_paths_top: Top violated paths summary (endpoint removed)

Shapes View Summaries:
- summarize_shape_constraint_distribution_templates: Analyze violation-to-constraint ratios
- summarize_shapes_correlation_templates: Analyze correlation between constraints and violations
- summarize_shapes_diversity_intensity_templates: Analyze violation diversity (entropy) vs intensity

Utility Functions:
- adaptive_thresholds: Calculate adaptive thresholds based on data size
- _example_terms: Extract representative examples for categories
- _percentage: Calculate percentage with proper formatting
- _cache_key: Generate cache keys for LLM responses

NLP Components:
- Uses spaCy for lemmatization and POS tagging
- Keyword-based semantic classification (7 categories: entity, temporal, spatial, identifier, attribute, generic)
- Optional LLM-based classification for flexible, domain-specific categories

Configuration:
- ENDPOINT_URL: SPARQL endpoint URL (default: http://localhost:8890/sparql)
- SHAPES_GRAPH_URI: URI for the shapes graph (default: http://ex.org/ShapesGraph)
- VALIDATION_REPORT_URI: URI for validation report (default: http://ex.org/ValidationReport)

LLM Support:
- Optional OpenAI integration for enhanced category detection
- Caching mechanism to avoid redundant API calls
- Falls back to keyword-based classification if LLM unavailable
- Configurable via use_llm parameter and OPENAI_API_KEY environment variable
"""

# Try to import OpenAI, if not available, set _openai_available to False
try:
    from openai import OpenAI
    _openai_available = True
    print("✓ OpenAI library imported successfully", file=sys.stderr)
except ImportError as e:
    _openai_available = False

# LLM response cache (simple in-memory cache)
# Key: hash of labels + context, Value: category name
_llm_cache: Dict[str, str] = {}
_MAX_CACHE_SIZE = 1000  # Limit cache size to prevent memory issues


# Lightweight NLP

_p = inflect.engine()

try:
    # We only need tagging + lemma
    _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])
except Exception:
    _nlp = spacy.blank("en")



# Label cleaning + semantic classification 
# ============================================================

_IRI_LAST_SEG_RE = re.compile(r"[#/](?!.*[/#])")

PERSON_KEYWORDS = {"person", "employee", "author", "user", "customer", "patient", "citizen"}
ORG_KEYWORDS = {"organization", "organisation", "company", "firm", "vendor", "provider", "institution"}
TIME_KEYWORDS = {"date", "time", "year", "month", "day", "timestamp", "period"}
LOC_KEYWORDS = {"address", "city", "country", "place", "location", "region", "street"}
ID_KEYWORDS = {"id", "identifier", "uuid", "code", "number", "no"}
NUM_KEYWORDS = {"amount", "price", "score", "rating", "age", "value", "count", "quantity"}
TEXT_KEYWORDS = {"name", "title", "label", "description", "comment", "text", "note"}

CATEGORY_KEYWORDS = {
    "entity": PERSON_KEYWORDS | ORG_KEYWORDS,
    "temporal": TIME_KEYWORDS,
    "spatial": LOC_KEYWORDS,
    "identifier": ID_KEYWORDS,
    "attribute": NUM_KEYWORDS | TEXT_KEYWORDS,
}


def _last_segment(s: str) -> str:
    if "://" in s or "#" in s or "/" in s:
        return _IRI_LAST_SEG_RE.split(s)[-1]
    if ":" in s:
        return s.split(":", 1)[-1]
    return s


def _doc_for_label(raw: str):
    """Process label through spaCy."""
    """http://ex.org/PersonShape" → "PersonShape"""
    base = _last_segment(raw).replace("_", " ").replace("-", " ")
    """postal_code" → "postal code"
    "order-id" → "order id"""
    base = re.sub(r"(?<=[a-z0-9])([A-Z])", r" \1", base)
    return _nlp(base)   


def analyze_label_semantics(raw: str) -> str:
    """
    Classify a label using keyword matching with lemmatization.
    Uses spaCy for better word matching (handles plurals, verb forms, etc.)
    """
    doc = _doc_for_label(raw)
    # Use lemmas for better matching (handles "employees" -> "employee", "dated" -> "date")
    lemmas = {t.lemma_.lower() for t in doc if t.is_alpha}
    # Also check original text for compound words
    texts = {t.text.lower() for t in doc if t.is_alpha}
    bag = lemmas | texts

    for category, keywords in CATEGORY_KEYWORDS.items():
        if bag & keywords:
            return category
    return "generic"


def category_display_word(category: str, plural: bool = False) -> str:
    """
    Map semantic category to a human phrase.
    """
    mapping = {
        "entity": "entity record",
        "temporal": "time-based field",
        "spatial": "location field",
        "identifier": "identifier field",
        "attribute": "data attribute",
        "generic": "data object",
    }
    base = mapping.get(category, "data object")
    if plural:
        return _p.plural(base)
    return base
    



def clean_label(raw: str) -> str:
    """
    Turn an IRI / QName / camelCase into a nicer label.

      http://ex.org/birthDate → "Birth date"
      ex:worksFor            → "Works for"
    """
    doc = _doc_for_label(raw)
    toks = []
    for t in doc:
        if not t.text.strip():
            continue
        # Use POS tags for better capitalization (proper nouns and nouns get capitalized)
        if t.pos_ in ("PROPN", "NOUN"):
            toks.append(t.text.capitalize())
        else:
            toks.append(t.text.lower())
    out = re.sub(r"\s+", " ", " ".join(toks)).strip()
    return out or raw


def format_name_for_text(name: str) -> str:
    """
    Format a name/label for insertion into text with quotes and capitalization.
    
    Examples:
      "type" -> "'Type'"
      "Person Shape" -> "'Person Shape'"
      "in constraint component" -> "'In Constraint Component'"
    """
    if not name:
        return ""
    # Capitalize first letter of each word
    words = name.split()
    capitalized_words = [word.capitalize() for word in words]
    capitalized_name = " ".join(capitalized_words)
    # Add quotes
    return f"'{capitalized_name}'"


def clean_pairs(pairs: Iterable[tuple[str, int]]) -> List[tuple[str, int]]:
    return [(clean_label(n), int(c)) for (n, c) in pairs]


def _create_openai_client(api_key: str, timeout: float) -> 'OpenAI':
    """
    Helper function to create OpenAI client with proper timeout configuration.
    Handles both httpx and fallback methods.
    """
    try:
        import httpx
        print(f"Using httpx for HTTP client", file=sys.stderr)
        timeout_obj = httpx.Timeout(timeout, connect=10.0)
        http_client = httpx.Client(timeout=timeout_obj)
        client = OpenAI(api_key=api_key, http_client=http_client, timeout=timeout)
        print(f"✓ OpenAI client created with httpx", file=sys.stderr)
        return client
    except (ImportError, TypeError) as e:
        print(f"httpx not available or error: {e}, using fallback method", file=sys.stderr)
        import os as os_module
        proxy_backup = {}
        for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
            if var in os_module.environ:
                proxy_backup[var] = os_module.environ[var]
                del os_module.environ[var]
        
        try:
            old_key = os_module.environ.get('OPENAI_API_KEY')
            os_module.environ['OPENAI_API_KEY'] = api_key
            client = OpenAI(timeout=timeout)
            return client
        finally:
            for var, value in proxy_backup.items():
                os_module.environ[var] = value
            if old_key:
                os_module.environ['OPENAI_API_KEY'] = old_key
            elif 'OPENAI_API_KEY' in os_module.environ:
                del os_module.environ['OPENAI_API_KEY']


def _make_llm_category_call(
    client: 'OpenAI',
    prompt: str,
    system_message: str,
    model: str,
    timeout: float,
    max_tokens: int = 15
) -> Optional[str]:
    """
    Helper function to make LLM API call and extract category.
    """
    print(f"Making LLM API call (timeout={timeout}s, model={model})...", file=sys.stderr)
    print(f"Prompt length: {len(prompt)} characters", file=sys.stderr)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=max_tokens,
            timeout=timeout
        )
        print(f"✓ LLM API call completed successfully", file=sys.stderr)
        return response.choices[0].message.content.strip().lower()
    except Exception as api_error:
        print(f"✗ API call raised exception: {type(api_error).__name__}: {api_error}", file=sys.stderr)
        raise


def _build_context_string(shapes_graph_uri: Optional[str], validation_report_uri: Optional[str]) -> str:
    """Helper to build context string from URIs."""
    context_parts = []
    if shapes_graph_uri:
        context_parts.append(f"Shapes Graph URI: {shapes_graph_uri}")
    if validation_report_uri:
        context_parts.append(f"Validation Report URI: {validation_report_uri}")
    return "\n".join(context_parts) if context_parts else "No additional context available."


def dominant_category(labels: Iterable[str]) -> str:
    """
    Return the *true* dominant category (including 'generic').
    Uses keyword-based classification.

    """
    counts: Dict[str, int] = {}
    for lab in labels:
        cat = analyze_label_semantics(lab)
        counts[cat] = counts.get(cat, 0) + 1
    if not counts:
        return "generic"
    # Strict argmax (generic may win)
    return max(counts.items(), key=lambda kv: kv[1])[0]


def dominant_category_llm(
    labels: Iterable[str],
    shapes_graph_uri: Optional[str] = None,
    validation_report_uri: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    timeout: float = 60.0,
    use_cache: bool = True
) -> Optional[str]:
    """
    Use LLM to determine the dominant category from labels.
    Returns a flexible category name (not limited to predefined categories), or None if LLM is unavailable.
    
    Note: This function makes an external API call and adds ~1-3 seconds of latency.
    Use only when flexible, context-aware category detection is needed.
    Results are cached to avoid redundant API calls.
    
    Args:
        labels: Iterable of labels (node shape names, paths, etc.)
        shapes_graph_uri: URI of the shapes graph (for context)
        validation_report_uri: URI of the validation report (for context)
        api_key: OpenAI API key (if None, tries environment variable)
        model: Model to use (default: gpt-4o - state-of-the-art and fast)
        timeout: Timeout in seconds
        use_cache: If True, cache results to avoid redundant API calls
    
    Returns:
        Category name as a string (flexible, domain-specific) or None if unavailable
    """
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"dominant_category_llm() called", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    
    if not _openai_available:
        print("✗ OpenAI library not available", file=sys.stderr)
        return None
    
    # Get API key from environment or parameter
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("✗ No API key found (neither parameter nor env var)", file=sys.stderr)
        return None
    
    print(f"✓ API key found (length: {len(api_key)})", file=sys.stderr)
    
    labels_list = list(labels)
    if not labels_list:
        return None
    
    # Check cache first
    if use_cache:
        cache_key = _cache_key(labels_list, shapes_graph_uri, validation_report_uri)
        if cache_key in _llm_cache:
            return _llm_cache[cache_key]
    
    try:
        client = _create_openai_client(api_key, timeout)
        
        # Clean labels and prepare prompt
        cleaned_labels = [clean_label(label) for label in labels_list]
        sample_labels = cleaned_labels[:30]
        if not sample_labels:
            return None
        
        context_str = _build_context_string(shapes_graph_uri, validation_report_uri)
        prompt = f"""You are analyzing labels from a SHACL validation report. SHACL (Shapes Constraint Language) is used to validate RDF data.

Context:
{context_str}

Labels (these represent data types, properties, or entities):
{', '.join(sample_labels)}

Based on these labels, determine the dominant semantic category or domain type.
Provide a concise category name (1-3 words) that best describes what these labels represent.
Examples:
- "person records" or "employee data"
- "product catalog"
- "geographic locations"
- "financial transactions"
- "scientific publications"
- "medical records"
- "organizational structures"
- "temporal data"
- "identifier fields"
- "spatial data"

Respond with ONLY the category name (1-3 words), nothing else. Use lowercase."""

        print(f"Sample labels: {sample_labels[:5]}...", file=sys.stderr)
        category = _make_llm_category_call(
            client, prompt,
            "You are a semantic categorization assistant. Respond with only a category name (1-3 words).",
            model, timeout, max_tokens=15
        )
        
        # Cache the result
        if use_cache and category:
            # Simple cache eviction: remove oldest entries if cache is too large
            if len(_llm_cache) >= _MAX_CACHE_SIZE:
                # Remove 20% of oldest entries (simple FIFO approximation)
                keys_to_remove = list(_llm_cache.keys())[:_MAX_CACHE_SIZE // 5]
                for key in keys_to_remove:
                    del _llm_cache[key]
            _llm_cache[cache_key] = category
        
        return category
            
    except Exception as e:
        # On any error, return None to fallback to keyword-based approach
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"LLM category detection FAILED: {error_type}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"Error message: {error_msg}", file=sys.stderr)
        print(f"API key provided: {'Yes' if api_key else 'No'}", file=sys.stderr)
        if api_key:
            print(f"API key preview: {api_key[:10]}...{api_key[-4:]}", file=sys.stderr)
        else:
            env_key = os.getenv("OPENAI_API_KEY", "")
            if env_key:
                print(f"Environment variable OPENAI_API_KEY found: {env_key[:10]}...{env_key[-4:]}", file=sys.stderr)
            else:
                print("No API key found in environment variable OPENAI_API_KEY", file=sys.stderr)
        print(f"Labels count: {len(labels_list)}", file=sys.stderr)
        print(f"Timeout setting: {timeout}s", file=sys.stderr)
        print(f"Model: {model}", file=sys.stderr)
        
        # Common error diagnostics
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            print("\n→ TIMEOUT ERROR: The API call took too long", file=sys.stderr)
            print("  - Try increasing the timeout value", file=sys.stderr)
            print("  - Check your network connection", file=sys.stderr)
        elif "api key" in error_msg.lower() or "authentication" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
            print("\n→ AUTHENTICATION ERROR: API key issue", file=sys.stderr)
            print("  - Verify your API key is correct", file=sys.stderr)
            print("  - Check if you have credits/quota", file=sys.stderr)
        elif "rate limit" in error_msg.lower() or "429" in error_msg:
            print("\n→ RATE LIMIT ERROR: Too many requests", file=sys.stderr)
            print("  - Wait a few minutes and try again", file=sys.stderr)
        
        import traceback
        print(f"\nFull traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
        return None


def dominant_category_llm_constraint(
    labels: Iterable[str],
    shapes_graph_uri: Optional[str] = None,
    validation_report_uri: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    timeout: float = 60.0
) -> Optional[str]:
    """
    Use LLM to determine the dominant constraint category from SHACL constraint component labels.
    Specialized version for constraint components that asks for specific constraint types.
    
    Returns categories like "cardinality constraints", "value constraints", "type constraints", etc.
    """
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"dominant_category_llm_constraint() called", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    
    if not _openai_available:
        print("✗ OpenAI library not available", file=sys.stderr)
        return None
    
    # Get API key from environment or parameter
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("✗ No API key found (neither parameter nor env var)", file=sys.stderr)
        return None
    
    print(f"✓ API key found (length: {len(api_key)})", file=sys.stderr)
    
    labels_list = list(labels)
    if not labels_list:
        return None
    
    print(f"✓ Processing {len(labels_list)} constraint component labels", file=sys.stderr)
    
    try:
        client = _create_openai_client(api_key, timeout)
        
        # Clean labels and prepare specialized prompt
        cleaned_labels = [clean_label(label) for label in labels_list]
        sample_labels = cleaned_labels[:30]
        if not sample_labels:
            return None
        
        context_str = _build_context_string(shapes_graph_uri, validation_report_uri)
        prompt = f"""You are analyzing SHACL constraint component names from a validation report. These are specific types of validation rules.

Context:
{context_str}

Constraint Component Names (these are SHACL validation rule types):
{', '.join(sample_labels)}

Based on these constraint component names, categorize them into a specific constraint type category.
Provide a concise category name (2-4 words) that describes the type of constraints.

Valid categories include:
- "cardinality constraints" (for minCount, maxCount, qualifiedMinCount, qualifiedMaxCount)
- "value constraints" (for in, hasValue, equals, disjoint)
- "type constraints" (for datatype, nodeKind, class)
- "range constraints" (for minLength, maxLength, minInclusive, maxInclusive, minExclusive, maxExclusive)
- "pattern constraints" (for pattern)
- "language constraints" (for languageIn, uniqueLang)
- "logical constraints" (for not, and, or, xone)
- "structural constraints" (for closed, node)

Respond with ONLY the category name (2-4 words), nothing else. Use lowercase."""
        
        category = _make_llm_category_call(
            client, prompt,
            "You are a SHACL constraint categorization assistant. Respond with only a constraint category name (2-4 words).",
            model, timeout, max_tokens=20
        )
        
        return category
            
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"LLM constraint categorization FAILED: {error_type}", file=sys.stderr)
        print(f"Error: {error_msg}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
        return None


def _example_terms(labels: Iterable[str], category: str, k: int = 2) -> List[str]:
    """
    Use spaCy to surface a couple of representative labels for the given category.

    If the category is 'generic' or we don't find matches,
    we just show a couple of cleaned labels (but only if the caller wants them).
    """
    labels = list(labels)
    examples: List[str] = []
    seen_cleaned = set()  # Track cleaned labels to avoid duplicates
    if category != "generic":
        for lab in labels:
            cleaned = clean_label(lab)
            if cleaned not in seen_cleaned and analyze_label_semantics(lab) == category:
                examples.append(cleaned)
                seen_cleaned.add(cleaned)
            if len(examples) >= k:
                break
    if not examples:
        # Deduplicate cleaned labels while preserving order
        seen_cleaned = set()
        for l in labels:
            cleaned = clean_label(l)
            if cleaned not in seen_cleaned:
                examples.append(cleaned)
                seen_cleaned.add(cleaned)
            if len(examples) >= k:
                break
    return examples


def _percentage(part: int, whole: int, level: str = "high") -> float:
    """
    Calculate percentage with rounding based on level.
    High level: round to 0 decimal places (94.4% -> 94%)
    Low level: round to 1 decimal place (94.4%)
    """
    if whole <= 0:
        return 0.0
    percentage = 100.0 * part / whole
    if level == "high":
        return round(percentage, 0)  # Round to 0 decimals for simplicity
    return round(percentage, 1)  # Round to 1 decimal for low level


def _format_percentage(perc: float, level: str = "high") -> str:
    """
    Format percentage as string based on level.
    High level: format as integer (94.0 -> "94")
    Low level: format with 1 decimal (94.4 -> "94.4")
    """
    if level == "high":
        return str(int(perc))  # Format as integer for high level
    return f"{perc:.1f}"  # Format with 1 decimal for low level


def _top_items(labels: List[str], counts: List[int], top_k: int = 3) -> List[Tuple[str, int]]:
    """Return top items by count with cleaned labels."""
    if not labels or not counts:
        return []
    size = min(len(labels), len(counts))
    items = [(clean_label(labels[i]), int(counts[i])) for i in range(size)]
    items.sort(key=lambda item: (-item[1], item[0]))
    return items[:top_k]


def _detail_insights(
    counts: List[int],
    labels: List[str],
    singular_label: str,
    plural_label: str,
) -> str:
    """Extra insights for low-level summaries."""
    if not counts:
        return ""
    median_val = int(np.percentile(counts, 50))
    p90_val = int(np.percentile(counts, 90))
    stats_sentence = (
        f"Median violations per {singular_label}: {median_val}; "
        f"90th percentile: {p90_val}."
    )
    top_items = _top_items(labels, counts, top_k=3)
    if top_items:
        top_list = ", ".join(
            f"{format_name_for_text(name)} ({count})" for name, count in top_items
        )
        top_sentence = f"Top {len(top_items)} {plural_label} by violations: {top_list}."
    else:
        top_sentence = ""
    return " ".join([stats_sentence, top_sentence]).strip()


def _join_nonempty(*parts: str) -> str:
    return " ".join([part for part in parts if part])


def _percentile_value(values: List[float], percentile: float) -> float:
    """Return percentile value or 0.0 for empty lists."""
    if not values:
        return 0.0
    return float(np.percentile(values, percentile))


def adaptive_thresholds(data_size: int) -> Tuple[float, float]:
    """
    Calculate adaptive thresholds based on data size.
    
    Args:
        data_size: Number of data points (shapes, paths, etc.)
    
    Returns:
        Tuple of (outlier_multiplier, cluster_threshold)
        - outlier_multiplier: Multiplier for detecting outliers (higher = stricter)
        - cluster_threshold: Minimum share for dominant cluster (higher = stricter)
    """
    if data_size < 5:
        # Small dataset: stricter thresholds (need higher concentration to be significant)
        outlier_multiplier = 2.0
        cluster_threshold = 0.5
    elif data_size > 50:
        # Large dataset: more lenient thresholds (patterns emerge with more data)
        outlier_multiplier = 1.3
        cluster_threshold = 0.25
    else:
        # Medium dataset: balanced defaults
        outlier_multiplier = 1.5
        cluster_threshold = 0.35
    
    return outlier_multiplier, cluster_threshold
    




def _is_wide_spread(violated_counts: List[int], min_threshold_ratio: float = 0.3) -> bool:
    """
    Determine if violation counts show a wide spread using statistical measures.
    
    Uses coefficient of variation (CV) and relative range to assess spread.
    A wide spread means there's meaningful variation relative to the average.
    
    Args:
        violated_counts: List of violation counts (must be non-empty, non-negative)
        min_threshold_ratio: Minimum coefficient of variation or relative range to consider "wide"
                            (default: 0.3 = 30% variation)
    
    Returns:
        True if the spread is considered "wide", False otherwise
    
    Examples:
        [1, 2, 3, 4, 5] -> True (high relative variation)
        [1000, 1001, 1002, 1003, 1004] -> False (low relative variation)
        [1, 1, 1, 1, 100] -> True (high relative variation due to outlier)
    """
    if not violated_counts or len(violated_counts) < 2:
        return False
    
    # Convert to numpy array for easier calculations
    counts_array = np.array(violated_counts, dtype=float)
    
    # Calculate statistics
    mean_val = np.mean(counts_array)
    std_val = np.std(counts_array)
    min_val = np.min(counts_array)
    max_val = np.max(counts_array)
    range_val = max_val - min_val
    
    # If all values are the same, no spread
    if range_val == 0:
        return False
    
    # If mean is very small (< 1), use absolute range threshold
    # For small means, even small absolute differences are significant
    if mean_val < 1.0:
        # Consider wide spread if range is at least 2 (meaningful difference for small numbers)
        return range_val >= 2.0
    
    # Calculate coefficient of variation (CV = std / mean)
    # CV measures relative variability independent of scale
    coefficient_of_variation = std_val / mean_val if mean_val > 0 else 0.0
    
    # Calculate relative range (range / mean)
    # This measures how much the range spans relative to the average
    relative_range = range_val / mean_val if mean_val > 0 else 0.0
    
    # Consider wide spread if either CV or relative range exceeds threshold
    # This catches both cases: high variability (CV) and large relative differences (relative_range)
    is_wide = coefficient_of_variation >= min_threshold_ratio or relative_range >= min_threshold_ratio
    
    return is_wide


def _cache_key(labels: List[str], shapes_graph_uri: Optional[str], validation_report_uri: Optional[str]) -> str:
    """Generate cache key from labels and context."""
    # Sort labels for consistent hashing
    sorted_labels = sorted([clean_label(l) for l in labels])
    context = f"{shapes_graph_uri or ''}|{validation_report_uri or ''}"
    content = json.dumps(sorted_labels[:30], sort_keys=True) + context
    return hashlib.md5(content.encode()).hexdigest()



# ============================================================
# HOME – Node Shape histogram
# ============================================================

def summarize_nodeshape_with_templates(
    homepage_service,
    shapes_graph_uri: str,
    report_uri: str,
    level: str = "high",
    use_llm: bool = False,
    include_category: bool = True,
    llm_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o",
) -> str:
    """
    'Violations per Node Shape' histogram.

    Each bar = how many node shapes fall into a violation interval.
    
    Args:
        use_llm: If True, use LLM to determine dominant category (adds ~1-3s latency). 
                 If False (default), use fast keyword-based classification (no overhead).
        include_category: If True, include category/domain information in summary. If False, omit it.
        llm_api_key: OpenAI API key (if None, uses environment variable)
        llm_model: Model to use (default: gpt-4o - state-of-the-art)
    """
    shape_data = homepage_service.get_violations_per_node_shape(
        shapes_graph_uri=shapes_graph_uri,
        validation_report_uri=report_uri,
    )
    counts = [int(item["NumViolations"]) for item in shape_data] if shape_data else []
    labels_ns = [item["NodeShapeName"] for item in shape_data] if shape_data else []
    detail_insights = ""
    if level != "high":
        detail_insights = _detail_insights(counts, labels_ns, "node shape", "node shapes")

    total_shapes = len(counts)
    violated_counts = [c for c in counts if c > 0]
    violated_shapes = len(violated_counts)

    # Determine category only if include_category is True
    use_semantic = False
    entity_word_plural = "data objects"
    examples = []
    
    if include_category:
        if use_llm:
            # Use LLM to determine dominant category (flexible, not limited to 7 categories)
            dom_category = dominant_category_llm(
                labels=labels_ns,
                shapes_graph_uri=shapes_graph_uri,
                validation_report_uri=report_uri,
                api_key=llm_api_key,
                model=llm_model,
            )
            if dom_category:
                entity_word_plural = dom_category
                examples = [clean_label(l) for l in list(labels_ns)[:2]]
                use_semantic = True
            else:
                # Fallback to keyword-based if LLM fails
                # Log warning if LLM was requested but failed
                if use_llm:
                    print(f"Warning: LLM category detection failed or returned None. Falling back to keyword-based classification.", file=sys.stderr)
                    if not llm_api_key and not os.getenv("OPENAI_API_KEY"):
                        print(f"  → No API key provided (neither llm_api_key parameter nor OPENAI_API_KEY env var)", file=sys.stderr)
                    elif llm_api_key:
                        print(f"  → API key provided via parameter (starts with: {llm_api_key[:7]}...)", file=sys.stderr)
                    elif os.getenv("OPENAI_API_KEY"):
                        print(f"  → API key from environment variable (starts with: {os.getenv('OPENAI_API_KEY', '')[:7]}...)", file=sys.stderr)
                dom_category = dominant_category(labels_ns)
                use_semantic = dom_category != "generic"
                if use_semantic:
                    entity_word_plural = category_display_word(dom_category, plural=True)
                    examples = _example_terms(labels_ns, dom_category, k=2)
        else:
            # Use keyword-based classification
            dom_category = dominant_category(labels_ns)
            use_semantic = dom_category != "generic"
            if use_semantic:
                entity_word_plural = category_display_word(dom_category, plural=True)
                examples = _example_terms(labels_ns, dom_category, k=2)

    if level == "high":
        intro = (
            "Node shape refers to a group of data objects for which certain rules apply. "
            "This chart groups these rule sets by how many violations they have."
        )
        if include_category and use_semantic and examples:
            classifier_type = "LLM analysis" if use_llm else "NLP classifier"
            intro += (
                f" Based on the shape names, the {classifier_type} suggests that most shapes describe "
                f"{entity_word_plural}, for example {', '.join(examples)}."
            )
    else:
        intro = (
            "Each bar aggregates SHACL NodeShapes by their total number of validation results. "
            "The bar height is the number of NodeShapes whose violation count falls into that interval."
        )
        if include_category and use_semantic and examples:
            intro += (
                f" From their labels, many of these shapes appear to describe {entity_word_plural}, "
                f"such as {', '.join(examples)}."
            )
    
    if total_shapes == 0:
        return _join_nonempty(intro, "No node shapes are present in the shapes graph.", detail_insights)

    if violated_shapes == 0:
        return _join_nonempty(intro, "No node shapes are violated.", detail_insights)

    dist = homepage_service.distribution_of_violations_per_shape(
        shapes_graph_uri=shapes_graph_uri,
        validation_report_uri=report_uri,
    )
    labels = dist.get("labels", [])
    freqs = dist.get("datasets", [{}])[0].get("data", [])

    if not labels or not freqs:
        if violated_counts:
            min_v, max_v = min(violated_counts), max(violated_counts)
            if _is_wide_spread(violated_counts):
                spread_sentence = (
                    f"They show a wide spread of violation counts, ranging from {min_v} to {max_v} violations. "
                    "Some shapes trigger only very few issues, while others suffer from much larger rule problems."
                )
            else:
                spread_sentence = (
                    f"Violation counts range from {min_v} to {max_v} violations. "
                    "The counts are relatively consistent across shapes, suggesting similar data quality levels."
                )
            return _join_nonempty(intro, spread_sentence, detail_insights)
        return _join_nonempty(intro, detail_insights)

    total_bins = sum(freqs) or 1
    max_freq = max(freqs)
    # Find all indices with the maximum frequency
    dom_indices = [i for i, freq in enumerate(freqs) if freq == max_freq]
    dom_labels = [labels[i] for i in dom_indices]
    dom_share = max_freq / total_bins

    # Extract ranges from labels
    dom_ranges = []
    dom_highs = []
    for dom_label in dom_labels:
        try:
            low_s, high_s = re.split(r"[-–]", dom_label)
            dom_range = f"{int(low_s)}-{int(high_s)}"
            dom_high = int(high_s)
            dom_ranges.append(dom_range)
            dom_highs.append(dom_high)
        except Exception:
            dom_ranges.append(dom_label)
            dom_highs.append(max(violated_counts) if violated_counts else 0)
    
    # Use the highest range value for threshold calculations
    dom_high = max(dom_highs) if dom_highs else 0
    # Format ranges for display (handles any number of items)
    if len(dom_ranges) == 1:
        dom_range_str = dom_ranges[0]
    elif len(dom_ranges) == 2:
        dom_range_str = f"{dom_ranges[0]} and {dom_ranges[1]}"
    else:
        # For 3+ items: "X, Y, Z, and A" or "X, Y, Z, A, and B" etc.
        dom_range_str = f"{', '.join(dom_ranges[:-1])}, and {dom_ranges[-1]}"

    vmin, vmax = (min(violated_counts), max(violated_counts)) if violated_counts else (0, 0)

    if counts:
        max_val = max(counts)
        # Find all indices with the maximum count
        max_indices = [i for i, count in enumerate(counts) if count == max_val]
        most_violated_names = [clean_label(labels_ns[i]) for i in max_indices]
        share_max = max_val / (sum(counts) or 1)
        # Format names for display (handles any number of items)
        if len(most_violated_names) == 1:
            most_violated_name = most_violated_names[0]
        elif len(most_violated_names) == 2:
            most_violated_name = f"{most_violated_names[0]} and {most_violated_names[1]}"
        else:
            # For 3+ items: "X, Y, Z, and A" or "X, Y, Z, A, and B" etc.
            most_violated_name = f"{', '.join(most_violated_names[:-1])}, and {most_violated_names[-1]}"
        max_indices_count = len(max_indices)
    else:
        most_violated_name, share_max = "", 0.0
        max_indices_count = 0,
    

    # Use adaptive thresholds based on data size
    outlier_multiplier, cluster_threshold = adaptive_thresholds(total_shapes)
    outlier_threshold = dom_high * outlier_multiplier if dom_high > 0 else 0
    has_outliers = vmax > outlier_threshold and outlier_threshold > 0
    no_dominant_sentence = ""
    if counts and not (share_max > cluster_threshold and vmax >= dom_high):
        no_dominant_sentence = "No single node shape dominates the overall violations."

    if dom_share < cluster_threshold:
        if _is_wide_spread(violated_counts):
            body = (
                f"They show a wide spread of violation counts, ranging from {vmin} to {vmax} violations. "
                "This means some shapes trigger only very few issues, while others suffer from much larger rule problems. "
                "The differences suggest varying data quality across shape types."
            )
        else:
            body = (
                f"Violation counts range from {vmin} to {vmax} violations. "
                "The counts are relatively consistent across shapes, suggesting similar data quality levels, "
                "though the distribution is not concentrated in a single range."
            )
        return _join_nonempty(intro, body, no_dominant_sentence, detail_insights)

    # Use adaptive threshold for share_max check
    if share_max > cluster_threshold and vmax >= dom_high:
        perc_out = _percentage(max_val, sum(counts) or 1, level)
        if len(dom_ranges) == 1:
            range_text = f"the range of {dom_range_str}"
        else:
            range_text = f"the ranges of {dom_range_str}"
        formatted_name = format_name_for_text(most_violated_name)
        body = (
            f"While most of them fall into {range_text} violations, "
            f"the shape{'s' if max_indices_count > 1 else ''} {formatted_name} {'are' if max_indices_count > 1 else 'is'} responsible for about {_format_percentage(perc_out, level)}% of all issues and should be reviewed first."
        )
        return _join_nonempty(intro, body, no_dominant_sentence, detail_insights)

    if not has_outliers:
        if len(dom_ranges) == 1:
            range_text = f"the range of {dom_range_str}"
        else:
            range_text = f"the ranges of {dom_range_str}"
        body = (
            f"Most of them fall into {range_text} violations, "
            "meaning that many data entities of this type likely share similar rule issues. "
            "This indicates that most of the problems are concentrated and probably relate to a recurring pattern in these shapes."
        )
        return _join_nonempty(intro, body, no_dominant_sentence, detail_insights)

    if len(dom_ranges) == 1:
        range_text = f"the range of {dom_range_str}"
    else:
        range_text = f"the ranges of {dom_range_str}"
    body = (
        f"While most of them fall into {range_text} violations, "
        f"a few shapes show unusually high counts with up to {vmax} violations. "
        "These outliers likely point to specific modeling or data-entry issues related to those shapes and should be reviewed first."
    )
    return _join_nonempty(intro, body, no_dominant_sentence, detail_insights)


def home_nodeshape_hist(
    homepage_service,
    shapes_graph_uri: str,
    report_uri: str,
    level: str,
    use_llm: bool = False,
    include_category: bool = True,
    llm_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o",
) -> str:
    return summarize_nodeshape_with_templates(
        homepage_service=homepage_service,
        shapes_graph_uri=shapes_graph_uri,
        report_uri=report_uri,
        level=level,
        use_llm=use_llm,
        include_category=include_category,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
    )


# ============================================================
# HOME – Path histogram
# ============================================================

def summarize_path_with_templates(
    homepage_service,
    report_uri: str,
    level: str = "high",
    use_llm: bool = False,
    include_category: bool = True,
    shapes_graph_uri: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o",
) -> str:
    """
    'Violations per Path' histogram.

    Each bar = how many paths fall into a violation interval.
    
    Args:
        use_llm: If True, use LLM to determine dominant category (adds ~1-3s latency).
                 If False (default), use fast keyword-based classification (no overhead).
        include_category: If True, include category information in summary. If False, omit it.
        shapes_graph_uri: URI of shapes graph (for LLM context)
        llm_api_key: OpenAI API key (if None, uses environment variable)
        llm_model: Model to use (default: gpt-4o)
    """
    path_data = homepage_service.get_violations_per_path(validation_report_uri=report_uri)
    counts = [int(item["NumViolations"]) for item in path_data] if path_data else []
    labels_path = [item["PathName"] for item in path_data] if path_data else []
    detail_insights = ""
    if level != "high":
        detail_insights = _detail_insights(counts, labels_path, "path", "paths")

    # Get total number of paths from shapes graph (not just violated ones)
    try:
        shapes_uri = shapes_graph_uri or SHAPES_GRAPH_URI
        total_paths = homepage_service.get_number_of_paths_in_shapes_graph(graph_uri=shapes_uri)
    except Exception:
        # Fallback: if we can't get total, use count from report (but this will show 100%)
        total_paths = len(counts)
    
    violated_counts = [c for c in counts if c > 0]
    violated_paths = len(violated_counts)
    
    # Use adaptive thresholds based on data size
    outlier_multiplier, cluster_threshold = adaptive_thresholds(total_paths)

    # Determine category if include_category is True
    use_semantic = False
    category_description = ""
    
    if include_category and labels_path:
        if use_llm:
            dom_category = dominant_category_llm(
                labels=labels_path,
                shapes_graph_uri=shapes_graph_uri,
                validation_report_uri=report_uri,
                api_key=llm_api_key,
                model=llm_model,
            )
            if dom_category:
                category_description = f" These paths primarily relate to {dom_category}."
                use_semantic = True

    if level == "high":
        intro = (
            "Path refers to the specific data field or property path where errors occur. "
            "This chart groups these paths by how many violations they have."
        )
        if use_semantic:
            intro += category_description
    else:
        intro = (
            "Each bar aggregates paths by their total number of validation results. "
            "The bar height is the number of paths whose violation count falls into that interval."
        )
        if use_semantic:
            intro += category_description

    if total_paths == 0:
        return _join_nonempty(intro, "No paths are present in the validation report.", detail_insights)

    if violated_paths == 0:
        return _join_nonempty(intro, "No paths are violated.", detail_insights)
    else:
        perc = _percentage(violated_paths, total_paths, level)
        perc_sentence = f"{_format_percentage(perc, level)}% of paths are violated."

    dist = homepage_service.distribution_of_violations_per_path(validation_report_uri=report_uri)
    labels = dist.get("labels", [])
    freqs = dist.get("datasets", [{}])[0].get("data", [])

    if not labels or not freqs:
        if violated_counts:
            min_v, max_v = min(violated_counts), max(violated_counts)
            if _is_wide_spread(violated_counts):
                spread_sentence = (
                    f"They show a wide spread of violation counts, ranging from {min_v} to {max_v} violations. "
                    "This means some paths trigger only very few issues, while others cause much larger rule problems. "
                    "Suggesting that the rule issues are broad and affect many properties of the data, rather than being isolated."
                )
            else:
                spread_sentence = (
                    f"Violation counts range from {min_v} to {max_v} violations. "
                    "The counts are relatively consistent across paths, suggesting similar data quality levels."
                )
            return _join_nonempty(intro, perc_sentence, spread_sentence, detail_insights)
        return _join_nonempty(intro, perc_sentence, detail_insights)

    total_bins = sum(freqs) or 1
    max_freq = max(freqs)
    # Find all indices with the maximum frequency
    dom_indices = [i for i, freq in enumerate(freqs) if freq == max_freq]
    dom_labels = [labels[i] for i in dom_indices]
    dom_share = max_freq / total_bins

    # Extract ranges from labels
    dom_ranges = []
    dom_highs = []
    for dom_label in dom_labels:
        try:
            low_s, high_s = re.split(r"[-–]", dom_label)
            dom_range = f"{int(low_s)}-{int(high_s)}"
            dom_high = int(high_s)
            dom_ranges.append(dom_range)
            dom_highs.append(dom_high)
        except Exception:
            dom_ranges.append(dom_label)
            dom_highs.append(max(violated_counts) if violated_counts else 0)
    
    # Use the highest range value for threshold calculations
    dom_high = max(dom_highs) if dom_highs else 0
    # Format ranges for display (handles any number of items)
    if len(dom_ranges) == 1:
        dom_range_str = dom_ranges[0]
    elif len(dom_ranges) == 2:
        dom_range_str = f"{dom_ranges[0]} and {dom_ranges[1]}"
    else:
        # For 3+ items: "X, Y, Z, and A" or "X, Y, Z, A, and B" etc.
        dom_range_str = f"{', '.join(dom_ranges[:-1])}, and {dom_ranges[-1]}"

    vmin, vmax = (min(violated_counts), max(violated_counts)) if violated_counts else (0, 0)

    if counts:
        max_val = max(counts)
        # Find all indices with the maximum count
        max_indices = [i for i, count in enumerate(counts) if count == max_val]
        most_violated_names = [clean_label(labels_path[i]) for i in max_indices]
        share_max = max_val / (sum(counts) or 1)
        # Format names for display (handles any number of items)
        if len(most_violated_names) == 1:
            most_violated_name = most_violated_names[0]
        elif len(most_violated_names) == 2:
            most_violated_name = f"{most_violated_names[0]} and {most_violated_names[1]}"
        else:
            # For 3+ items: "X, Y, Z, and A" or "X, Y, Z, A, and B" etc.
            most_violated_name = f"{', '.join(most_violated_names[:-1])}, and {most_violated_names[-1]}"
        max_indices_count = len(max_indices)
    else:
        most_violated_name, share_max = "", 0.0
        max_indices_count = 0

    outlier_threshold = dom_high * outlier_multiplier if dom_high > 0 else 0
    has_outliers = vmax > outlier_threshold and outlier_threshold > 0

    if dom_share < cluster_threshold:
        if _is_wide_spread(violated_counts):
            body = (
                f"They show a wide spread of violation counts, ranging from {vmin} to {vmax} violations. "
                "This means some paths trigger only very few issues, while others cause much larger rule problems. "
                "Suggesting that the rule issues are broad and affect many properties of the data, rather than being isolated."
            )
        else:
            body = (
                f"Violation counts range from {vmin} to {vmax} violations. "
                "The counts are relatively consistent across paths, suggesting similar data quality levels, "
                "though the distribution is not concentrated in a single range."
            )
        return _join_nonempty(intro, body, detail_insights)

    if share_max > cluster_threshold and vmax >= dom_high:
        perc_out = _percentage(max_val, sum(counts) or 1, level)
        if len(dom_ranges) == 1:
            range_text = f"the range of {dom_range_str}"
        else:
            range_text = f"the ranges of {dom_range_str}"
        formatted_name = format_name_for_text(most_violated_name)
        body = (
            f"While most of them fall into {range_text} violations, "
            f"the path{'s' if max_indices_count > 1 else ''} {formatted_name} {'are' if max_indices_count > 1 else 'is'} responsible for about {_format_percentage(perc_out, level)}% of all issues and should be reviewed first."
        )
        return _join_nonempty(intro, body, detail_insights)

    if not has_outliers:
        if len(dom_ranges) == 1:
            range_text = f"the range of {dom_range_str}"
        else:
            range_text = f"the ranges of {dom_range_str}"
        body = (
            f"The violations are distributed across multiple paths. The most common values lie in {range_text}. "
            "This indicates that most of the problems are concentrated and probably relate to a recurring pattern in some node shapes regarding these paths."
        )
        return _join_nonempty(intro, body, detail_insights)

    if len(dom_ranges) == 1:
        range_text = f"the range of {dom_range_str}"
    else:
        range_text = f"the ranges of {dom_range_str}"
    body = (
        f"The violations are distributed across multiple paths. While the most common values lie in {range_text}, "
        f"a few paths show unusually high counts with up to {vmax} violations. "
        "This indicates widespread nonconformity across the data to those property constraints, so these paths should be reviewed first."
    )
    return _join_nonempty(intro, perc_sentence, body, detail_insights)


def home_path_hist(
    homepage_service,
    report_uri: str,
    level: str,
    use_llm: bool = False,
    include_category: bool = True,
    shapes_graph_uri: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o",
) -> str:
    return summarize_path_with_templates(
        homepage_service=homepage_service,
        report_uri=report_uri,
        level=level,
        use_llm=use_llm,
        include_category=include_category,
        shapes_graph_uri=shapes_graph_uri,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
    )


# ============================================================
# HOME – Focus node histogram
# ============================================================

def summarize_focusnode_with_templates(
    homepage_service,
    report_uri: str,
    level: str = "high",
    use_llm: bool = False,
    include_category: bool = True,
    shapes_graph_uri: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o",
) -> str:
    """
    'Violations per Focus Node' histogram.

    Each bar = how many focus nodes fall into a violation interval.
    
    Args:
        use_llm: If True, use LLM to determine dominant category (adds ~1-3s latency).
                 If False (default), use fast keyword-based classification (no overhead).
        include_category: If True, include category/domain information in summary. If False, omit it.
        shapes_graph_uri: URI of shapes graph (for LLM context)
        llm_api_key: OpenAI API key (if None, uses environment variable)
        llm_model: Model to use (default: gpt-4o)
    """
    focus_data = homepage_service.get_violations_per_focus_node(validation_report_uri=report_uri)
    counts = [int(item["NumViolations"]) for item in focus_data] if focus_data else []
    labels_fn = [item["FocusNodeName"] for item in focus_data] if focus_data else []
    detail_insights = ""
    if level != "high":
        detail_insights = _detail_insights(counts, labels_fn, "focus node", "focus nodes")

    total_nodes = len(counts)
    violated_counts = [c for c in counts if c > 0]
    violated_nodes = len(violated_counts)
    
    # Use adaptive thresholds based on data size
    outlier_multiplier, cluster_threshold = adaptive_thresholds(total_nodes)

    # Determine category only if include_category is True
    use_semantic = False
    entity_word_plural = "data objects"
    examples = []
    
    if include_category and labels_fn:
        if use_llm:
            # Use LLM to determine dominant category (flexible, not limited to 7 categories)
            dom_category = dominant_category_llm(
                labels=labels_fn,
                shapes_graph_uri=shapes_graph_uri,
                validation_report_uri=report_uri,
                api_key=llm_api_key,
                model=llm_model,
            )
            if dom_category:
                entity_word_plural = dom_category
                # Get unique examples to avoid duplicates (deduplicate after cleaning since different URIs may clean to same name)
                cleaned_labels = []
                seen_cleaned = set()
                for l in labels_fn:
                    cleaned = clean_label(l)
                    if cleaned not in seen_cleaned:
                        cleaned_labels.append(cleaned)
                        seen_cleaned.add(cleaned)
                        if len(cleaned_labels) >= 2:
                            break
                examples = cleaned_labels
                use_semantic = True
            else:
                # Fallback to keyword-based if LLM fails
                dom_category = dominant_category(labels_fn)
                use_semantic = dom_category != "generic"
                if use_semantic:
                    entity_word_plural = category_display_word(dom_category, plural=True)
                    examples = _example_terms(labels_fn, dom_category, k=2)
        else:
            # Use keyword-based classification
            dom_category = dominant_category(labels_fn)
            use_semantic = dom_category != "generic"
            if use_semantic:
                entity_word_plural = category_display_word(dom_category, plural=True)
                examples = _example_terms(labels_fn, dom_category, k=2)

    if level == "high":
        intro = (
            "Focus node refers to a single verified object. "
            "This chart groups these objects by how many violations they have."
        )
        if include_category and use_semantic and examples:
            classifier_type = "LLM analysis" if use_llm else "NLP classifier"
            intro += (
                f" Based on the focus node names, the {classifier_type} suggests that most nodes describe "
                f"{entity_word_plural}, for example {', '.join(examples)}."
            )
    else:
        intro = (
            "Each bar aggregates focus nodes by their total number of validation results. "
            "The bar height is the number of focus nodes whose violation count falls into that interval."
        )
        if include_category and use_semantic and examples:
            intro += (
                f" From their labels, many of these nodes appear to describe {entity_word_plural}, "
                f"such as {', '.join(examples)}."
            )

    if total_nodes == 0:
        return _join_nonempty(intro, "No focus nodes are present in the validation report.", detail_insights)

    if violated_nodes == 0:
        return _join_nonempty(intro, "No focus nodes are violated.", detail_insights)

    dist = homepage_service.distribution_of_violations_per_focus_node(validation_report_uri=report_uri)
    labels = dist.get("labels", [])
    freqs = dist.get("datasets", [{}])[0].get("data", [])

    if not labels or not freqs:
        if violated_counts:
            min_v, max_v = min(violated_counts), max(violated_counts)
            if _is_wide_spread(violated_counts):
                spread_sentence = (
                    f"They differ widely in how many violations they have. "
                    f"Some objects trigger almost no rule issues, while others have a very large number of up to {max_v}. "
                    "This variance may point to inconsistent data entries or different data sources of mixed quality."
                )
            else:
                spread_sentence = (
                    f"Violation counts range from {min_v} to {max_v}. "
                    "The counts are relatively consistent across focus nodes, suggesting similar data quality levels."
                )
            return _join_nonempty(intro, spread_sentence, detail_insights)
        return _join_nonempty(intro, detail_insights)

    total_bins = sum(freqs) or 1
    max_freq = max(freqs)
    # Find all indices with the maximum frequency
    dom_indices = [i for i, freq in enumerate(freqs) if freq == max_freq]
    dom_labels = [labels[i] for i in dom_indices]
    dom_share = max_freq / total_bins

    # Extract ranges from labels
    dom_ranges = []
    dom_highs = []
    for dom_label in dom_labels:
        try:
            low_s, high_s = re.split(r"[-–]", dom_label)
            dom_range = f"{int(low_s)}-{int(high_s)}"
            dom_high = int(high_s)
            dom_ranges.append(dom_range)
            dom_highs.append(dom_high)
        except Exception:
            dom_ranges.append(dom_label)
            dom_highs.append(max(violated_counts) if violated_counts else 0)
    
    # Use the highest range value for threshold calculations
    dom_high = max(dom_highs) if dom_highs else 0
    # Format ranges for display (handles any number of items)
    if len(dom_ranges) == 1:
        dom_range_str = dom_ranges[0]
    elif len(dom_ranges) == 2:
        dom_range_str = f"{dom_ranges[0]} and {dom_ranges[1]}"
    else:
        # For 3+ items: "X, Y, Z, and A" or "X, Y, Z, A, and B" etc.
        dom_range_str = f"{', '.join(dom_ranges[:-1])}, and {dom_ranges[-1]}"

    vmin, vmax = (min(violated_counts), max(violated_counts)) if violated_counts else (0, 0)

    if counts:
        max_val = max(counts)
        # Find all indices with the maximum count
        max_indices = [i for i, count in enumerate(counts) if count == max_val]
        most_violated_names = [clean_label(labels_fn[i]) for i in max_indices]
        share_max = max_val / (sum(counts) or 1)
        # Format names for display (handles any number of items)
        if len(most_violated_names) == 1:
            most_violated_name = most_violated_names[0]
        elif len(most_violated_names) == 2:
            most_violated_name = f"{most_violated_names[0]} and {most_violated_names[1]}"
        else:
            # For 3+ items: "X, Y, Z, and A" or "X, Y, Z, A, and B" etc.
            most_violated_name = f"{', '.join(most_violated_names[:-1])}, and {most_violated_names[-1]}"
        max_indices_count = len(max_indices)
    else:
        most_violated_name, share_max = "", 0.0
        max_indices_count = 0

    outlier_threshold = dom_high * outlier_multiplier if dom_high > 0 else 0
    has_outliers = vmax > outlier_threshold and outlier_threshold > 0

    if dom_share < cluster_threshold:
        if _is_wide_spread(violated_counts):
            body = (
                f"They differ widely in how many violations they have. "
                f"Some objects trigger almost no rule issues, while others have a very large number of up to {vmax}. "
                "This variance may point to inconsistent data entries or different data sources of mixed quality."
            )
        else:
            body = (
                f"Violation counts range from {vmin} to {vmax}. "
                "The counts are relatively consistent across focus nodes, suggesting similar data quality levels, "
                "though the distribution is not concentrated in a single range."
            )
        return _join_nonempty(intro, body, detail_insights)

    if share_max > cluster_threshold and vmax >= dom_high:
        perc_out = _percentage(max_val, sum(counts) or 1, level)
        if len(dom_ranges) == 1:
            range_text = f"a range of {dom_range_str}"
        else:
            range_text = f"ranges of {dom_range_str}"
        formatted_name = format_name_for_text(most_violated_name)
        body = (
            f"Although most of them fall into {range_text} violations, "
            f"the node{'s' if max_indices_count > 1 else ''} {formatted_name} {'are' if max_indices_count > 1 else 'is'} responsible for most issues out of all nodes and should be reviewed first."
        )
        return _join_nonempty(intro, body, detail_insights)

    if not has_outliers:
        if len(dom_ranges) == 1:
            range_text = f"a range of {dom_range_str}"
        else:
            range_text = f"ranges of {dom_range_str}"
        body = (
            f"Most of them fall into {range_text} violations, "
            "meaning that the majority of individual records have a similar number of rule issues. "
            "This suggests that the problems are systematic and likely affect many entries in the same way."
        )
        return _join_nonempty(intro, body, detail_insights)

    if len(dom_ranges) == 1:
        range_text = f"a range of {dom_range_str}"
    else:
        range_text = f"ranges of {dom_range_str}"
    body = (
        f"Although most of them fall into {range_text} violations, "
        f"a few nodes show unusually high counts of up to {vmax} violations. "
        "These outliers likely need targeted review, as they contain most of the problematic data."
    )
    return _join_nonempty(intro, body, detail_insights)


def home_focusnode_hist(
    homepage_service,
    report_uri: str,
    level: str,
    use_llm: bool = False,
    include_category: bool = True,
    shapes_graph_uri: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o",
) -> str:
    return summarize_focusnode_with_templates(
        homepage_service=homepage_service,
        report_uri=report_uri,
        level=level,
        use_llm=use_llm,
        include_category=include_category,
        shapes_graph_uri=shapes_graph_uri,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
    )


# ============================================================
# HOME – Constraint component histogram
# ============================================================

def summarize_constraint_with_templates(
    homepage_service,
    report_uri: str,
    level: str = "high",
    use_llm: bool = False,
    include_category: bool = True,
    shapes_graph_uri: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o",
) -> str:
    """
    'Violations per Constraint Component' histogram.

    Each bar = how many constraint components fall into a violation interval.
    
    Args:
        use_llm: If True, use LLM to determine dominant category (adds ~1-3s latency).
                 If False (default), use fast keyword-based classification (no overhead).
        include_category: If True, include category/domain information in summary. If False, omit it.
        shapes_graph_uri: URI of shapes graph (for LLM context)
        llm_api_key: OpenAI API key (if None, uses environment variable)
        llm_model: Model to use (default: gpt-4o)
    """
    # Get raw constraint component data
    query = f"""
    PREFIX sh: <http://www.w3.org/ns/shacl#>

    SELECT ?constraintComponent (COUNT(?violation) AS ?violationCount)
    WHERE {{
      GRAPH <{report_uri}> {{
        ?violation sh:sourceConstraintComponent ?constraintComponent .
      }}
    }}
    GROUP BY ?constraintComponent
    """
    
    try:
        response = requests.get(
            ENDPOINT_URL,
            params={"query": query, "format": "json"},
        )
        response.raise_for_status()
        results = response.json()["results"]["bindings"]
        
        constraint_data = [
            {
                "ConstraintComponentName": result["constraintComponent"]["value"],
                "NumViolations": int(result["violationCount"]["value"])
            }
            for result in results
        ]
    except Exception:
        constraint_data = []

    counts = [int(item["NumViolations"]) for item in constraint_data] if constraint_data else []
    labels_cc = [item["ConstraintComponentName"] for item in constraint_data] if constraint_data else []
    detail_insights = ""
    if level != "high":
        detail_insights = _detail_insights(
            counts, labels_cc, "constraint component", "constraint components"
        )

    # Get total number of constraint components from shapes graph (not just violated ones)
    try:
        shapes_uri = shapes_graph_uri or SHAPES_GRAPH_URI
        total_constraints = homepage_service.get_distinct_constraints_count_in_shapes(shapes_graph_uri=shapes_uri)
    except Exception:
        # Fallback: if we can't get total, use count from report (but this will show 100%)
        total_constraints = len(counts)
    
    violated_counts = [c for c in counts if c > 0]
    violated_constraints = len(violated_counts)
    
    # Use adaptive thresholds based on data size
    outlier_multiplier, cluster_threshold = adaptive_thresholds(total_constraints)

    # Determine category only if include_category is True
    use_semantic = False
    category_description = ""
    
    if include_category and labels_cc:
        if use_llm:
            # Use specialized function for constraint components
            dom_category = dominant_category_llm_constraint(
                labels=labels_cc,
                shapes_graph_uri=shapes_graph_uri,
                validation_report_uri=report_uri,
                api_key=llm_api_key,
                model=llm_model,
            )
            if dom_category:
                category_description = f" These constraint types primarily relate to {dom_category}."
                use_semantic = True
        else:
            # For constraint components, keyword-based classification may not be as useful
            # but we can still try
            dom_category = dominant_category(labels_cc)
            use_semantic = dom_category != "generic"
            if use_semantic:
                category_description = f" These constraint types primarily relate to {category_display_word(dom_category, plural=True)}."

    if level == "high":
        intro = (
            "Constraint component refers to the type of rule that was violated. "
            "This chart groups these rule types by how many violations they have."
        )
        if use_semantic:
            intro += category_description
    else:
        intro = (
            "Each bar aggregates constraint components by their total number of validation results. "
            "The bar height is the number of constraint components whose violation count falls into that interval."
        )
        if use_semantic:
            intro += category_description

    if total_constraints == 0:
        return _join_nonempty(
            intro,
            "No constraint components are present in the validation report.",
            detail_insights,
        )

    if violated_constraints == 0:
        return _join_nonempty(
            intro,
            "No constraint components are violated.",
            detail_insights,
        )
    else:
        perc = _percentage(violated_constraints, total_constraints, level)
        perc_sentence = f"{_format_percentage(perc, level)}% of constraint components are violated."

    dist = homepage_service.get_distribution_of_violations_per_constraint_component(
        validation_report_uri=report_uri
    )
    labels = dist.get("labels", [])
    freqs = dist.get("datasets", [{}])[0].get("data", [])

    if not labels or not freqs:
        if violated_counts:
            min_v, max_v = min(violated_counts), max(violated_counts)
            if _is_wide_spread(violated_counts):
                spread_sentence = (
                    f"They differ widely in how often they are violated. "
                    f"Some rules cause almost no issues, while others are violated up to {max_v} times. "
                    "This variance may point to inconsistent data entries or different data sources of mixed quality."
                )
            else:
                spread_sentence = (
                    f"Violation counts range from {min_v} to {max_v}. "
                    "The counts are relatively consistent across constraint components, suggesting similar violation rates."
                )
            return _join_nonempty(intro, perc_sentence, spread_sentence, detail_insights)
        return _join_nonempty(intro, perc_sentence, detail_insights)

    total_bins = sum(freqs) or 1
    max_freq = max(freqs)
    # Find all indices with the maximum frequency
    dom_indices = [i for i, freq in enumerate(freqs) if freq == max_freq]
    dom_labels = [labels[i] for i in dom_indices]
    dom_share = max_freq / total_bins

    # Extract ranges from labels
    dom_ranges = []
    dom_highs = []
    for dom_label in dom_labels:
        try:
            low_s, high_s = re.split(r"[-–]", dom_label)
            dom_range = f"{int(low_s)}-{int(high_s)}"
            dom_high = int(high_s)
            dom_ranges.append(dom_range)
            dom_highs.append(dom_high)
        except Exception:
            dom_ranges.append(dom_label)
            dom_highs.append(max(violated_counts) if violated_counts else 0)
    
    # Use the highest range value for threshold calculations
    dom_high = max(dom_highs) if dom_highs else 0
    # Format ranges for display (handles any number of items)
    if len(dom_ranges) == 1:
        dom_range_str = dom_ranges[0]
    elif len(dom_ranges) == 2:
        dom_range_str = f"{dom_ranges[0]} and {dom_ranges[1]}"
    else:
        # For 3+ items: "X, Y, Z, and A" or "X, Y, Z, A, and B" etc.
        dom_range_str = f"{', '.join(dom_ranges[:-1])}, and {dom_ranges[-1]}"

    vmin, vmax = (min(violated_counts), max(violated_counts)) if violated_counts else (0, 0)

    if counts:
        max_val = max(counts)
        # Find all indices with the maximum count
        max_indices = [i for i, count in enumerate(counts) if count == max_val]
        most_violated_names = [clean_label(labels_cc[i]) for i in max_indices]
        share_max = max_val / (sum(counts) or 1)
        # Format names for display (handles any number of items)
        if len(most_violated_names) == 1:
            most_violated_name = most_violated_names[0]
        elif len(most_violated_names) == 2:
            most_violated_name = f"{most_violated_names[0]} and {most_violated_names[1]}"
        else:
            # For 3+ items: "X, Y, Z, and A" or "X, Y, Z, A, and B" etc.
            most_violated_name = f"{', '.join(most_violated_names[:-1])}, and {most_violated_names[-1]}"
        max_indices_count = len(max_indices)
    else:
        most_violated_name, share_max = "", 0.0
        max_indices_count = 0

    outlier_threshold = dom_high * outlier_multiplier if dom_high > 0 else 0
    has_outliers = vmax > outlier_threshold and outlier_threshold > 0

    if dom_share < cluster_threshold:
        if _is_wide_spread(violated_counts):
            body = (
                f"They differ widely in how often they are violated. "
                f"Some rules cause almost no issues, while others are violated up to {vmax} times. "
                "This variance may point to inconsistent data entries or different data sources of mixed quality."
            )
        else:
            body = (
                f"Violation counts range from {vmin} to {vmax}. "
                "The counts are relatively consistent across constraint components, suggesting similar violation rates, "
                "though the distribution is not concentrated in a single range."
            )
        return _join_nonempty(intro, body, detail_insights)

    if share_max > cluster_threshold and vmax >= dom_high:
        perc_out = _percentage(max_val, sum(counts) or 1, level)
        if len(dom_ranges) == 1:
            range_text = f"range of {dom_range_str}"
        else:
            range_text = f"ranges of {dom_range_str}"
        formatted_name = format_name_for_text(most_violated_name)
        body = (
            f"Most rule types contribute to the violations with counts in {range_text}. "
            f"However {_format_percentage(perc_out, level)}% of all violations stem from the constraint component{'s' if max_indices_count > 1 else ''} {formatted_name}, "
            "meaning that this requirement is frequently not met. "
            "This indicates a systematic issue, such as missing values or incorrect formats."
        )
        return _join_nonempty(intro, body, detail_insights)

    if not has_outliers:
        if len(dom_ranges) == 1:
            range_text = f"the range of {dom_range_str}"
        else:
            range_text = f"the ranges of {dom_range_str}"
        body = (
            f"The different rule types are mostly violated at similar levels in {range_text}. "
            "No single rule stands out, which suggests that the data issues are broad and not limited to any particular rule."
        )
        return _join_nonempty(intro, body, detail_insights)

    if len(dom_ranges) == 1:
        range_text = f"range of {dom_range_str}"
    else:
        range_text = f"ranges of {dom_range_str}"
    body = (
        f"While several rule types contribute to the violations with counts in {range_text}, "
        f"some are violated up to {vmax} times. "
        "This indicates a systematic issue, such as missing values and incorrect formats. These outliers should be reviewed first."
    )
    return _join_nonempty(intro, perc_sentence, body, detail_insights)


def home_constraint_hist(
    homepage_service,
    report_uri: str,
    level: str,
    use_llm: bool = False,
    include_category: bool = True,
    shapes_graph_uri: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o",
) -> str:
    return summarize_constraint_with_templates(
        homepage_service=homepage_service,
        report_uri=report_uri,
        level=level,
        use_llm=use_llm,
        include_category=include_category,
        shapes_graph_uri=shapes_graph_uri,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
    )


# ============================================================
# HOME – Path Toplist (removed)
# ============================================================

# ============================================================
# SHAPES VIEW – Distribution per constraint inside a NodeShape
# ============================================================

def summarize_shape_constraint_distribution_templates(
    shapes_overview_service,
    node_shape: str,
    report_uri: str,
    level: str = "high",
    use_llm: bool = False,
    include_category: bool = True,
    shapes_graph_uri: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o",
) -> str:
    """
    Per-constraint distribution inside a NodeShape.
    Analyzes the distribution of violations per constraint ratio across all node shapes.
    
    Args:
        use_llm: If True, use LLM to determine dominant category (adds ~1-3s latency).
                 If False (default), use fast keyword-based classification (no overhead).
        include_category: If True, include category/domain information in summary. If False, omit it.
        shapes_graph_uri: URI of shapes graph (for LLM context and to get node shape names)
        llm_api_key: OpenAI API key (if None, uses environment variable)
        llm_model: Model to use (default: gpt-4o)
    """
    try:
        dist = shapes_overview_service.get_distribution_of_violations_per_constraint(
            shapes_graph_uri=SHAPES_GRAPH_URI,
            validation_report_uri=report_uri,
        )
    except Exception:
        if level == "high":
            return (
                "This chart groups node shapes by the ratio of violations to constraints. "
                "Unable to retrieve distribution data."
            )
        return (
            "Each bar corresponds to a bin of violation-to-constraint ratios for NodeShapes. "
            "No data available."
        )
    
    labels = dist.get("labels", [])
    freqs = dist.get("datasets", [{}])[0].get("data", [])
    
    # Get node shape names for category detection
    node_shape_names = []
    if include_category:
        try:
            shapes_uri = shapes_graph_uri or SHAPES_GRAPH_URI
            query = f"""
            PREFIX sh: <http://www.w3.org/ns/shacl#>
            SELECT DISTINCT ?nodeShape
            FROM <{shapes_uri}>
            WHERE {{
                ?nodeShape a sh:NodeShape .
            }}
            """
            response = requests.get(ENDPOINT_URL, params={"query": query, "format": "json"})
            response.raise_for_status()
            results = response.json()["results"]["bindings"]
            node_shape_names = [result["nodeShape"]["value"] for result in results]
        except Exception:
            node_shape_names = []
    
    # Determine category only if include_category is True
    use_semantic = False
    entity_word_plural = "data objects"
    examples = []
    
    if include_category and node_shape_names:
        if use_llm:
            dom_category = dominant_category_llm(
                labels=node_shape_names,
                shapes_graph_uri=shapes_graph_uri or SHAPES_GRAPH_URI,
                validation_report_uri=report_uri,
                api_key=llm_api_key,
                model=llm_model,
            )
            if dom_category:
                entity_word_plural = dom_category
                examples = [clean_label(l) for l in list(node_shape_names)[:2]]
                use_semantic = True
            else:
                dom_category = dominant_category(node_shape_names)
                use_semantic = dom_category != "generic"
                if use_semantic:
                    entity_word_plural = category_display_word(dom_category, plural=True)
                    examples = _example_terms(node_shape_names, dom_category, k=2)
        else:
            dom_category = dominant_category(node_shape_names)
            use_semantic = dom_category != "generic"
            if use_semantic:
                entity_word_plural = category_display_word(dom_category, plural=True)
                examples = _example_terms(node_shape_names, dom_category, k=2)
    
    if level == "high":
        intro = (
            "This chart groups node shapes by the ratio of violations to constraints. "
            "The ratio indicates how many violations occur per constraint on average."
        )
        if include_category and use_semantic and examples:
            classifier_type = "LLM analysis" if use_llm else "NLP classifier"
            intro += (
                f" Based on the shape names, the {classifier_type} suggests that most shapes describe "
                f"{entity_word_plural}, for example {', '.join(examples)}."
            )
    else:
        intro = (
            "Each bar corresponds to a bin of violation-to-constraint ratios for NodeShapes. "
            "The bar height represents the number of node shapes whose ratio falls into that interval."
        )
        if include_category and use_semantic and examples:
            intro += (
                f" From their labels, many of these shapes appear to describe {entity_word_plural}, "
                f"such as {', '.join(examples)}."
            )
    
    if not labels or not freqs:
        return intro + " No distribution data available."
    
    total_shapes = sum(freqs)
    if total_shapes == 0:
        return intro + " No node shapes found in the distribution."

    # Detect zero-violation shapes to avoid masking by wide first bin
    zero_note = ""
    try:
        from functions import homepage_service

        total_node_shapes = homepage_service.get_number_of_node_shapes(SHAPES_GRAPH_URI)
        shapes_with_violations = homepage_service.get_number_of_node_shapes_with_violations(
            SHAPES_GRAPH_URI, report_uri
        )
        zero_count = max(total_node_shapes - shapes_with_violations, 0)
        if zero_count > 0:
            zero_pct = _format_percentage(_percentage(zero_count, total_shapes, level), level)
            zero_note = (
                f"Note: {zero_count} node shapes ({zero_pct}%) have zero violations, "
                "so the lowest bin mixes zero-violation shapes with low-but-nonzero ratios."
            )
    except Exception:
        zero_note = ""
    
    num_bins = len(freqs)  # Should always be 10 (or num_bins from distribution function)
    max_freq = max(freqs)
    # Find all indices with the maximum frequency
    dom_indices = [i for i, freq in enumerate(freqs) if freq == max_freq]
    dom_labels = [labels[i] for i in dom_indices]
    dom_share = max_freq / total_shapes if total_shapes > 0 else 0
    
    # Extract ranges from labels
    dom_ranges = []
    dom_highs = []
    for dom_label in dom_labels:
        try:
            low_s, high_s = re.split(r"[-–]", dom_label)
            dom_range = f"{float(low_s):.2f}-{float(high_s):.2f}"
            dom_high = float(high_s)
            dom_ranges.append(dom_range)
            dom_highs.append(dom_high)
        except Exception:
            dom_ranges.append(dom_label)
            dom_highs.append(0.0)
    
    # Use the highest range value for threshold calculations
    dom_high = max(dom_highs) if dom_highs else 0.0
    # Format ranges for display (handles any number of items)
    if len(dom_ranges) == 1:
        dom_range_str = dom_ranges[0]
    elif len(dom_ranges) == 2:
        dom_range_str = f"{dom_ranges[0]} and {dom_ranges[1]}"
    else:
        # For 3+ items: "X, Y, Z, and A" or "X, Y, Z, A, and B" etc.
        dom_range_str = f"{', '.join(dom_ranges[:-1])}, and {dom_ranges[-1]}"
    
    # Analyze distribution pattern
    non_zero_freqs = [f for f in freqs if f > 0]
    if not non_zero_freqs:
        return intro + " All node shapes have zero violations per constraint."
    
    # Calculate spread
    active_bins = len(non_zero_freqs)
    
    detail_insights = ""
    if level != "high":
        active_bins = len([freq for freq in freqs if freq > 0])
        high_bins = freqs[-2:] if len(freqs) >= 2 else freqs
        high_share = sum(high_bins) / total_shapes if total_shapes > 0 else 0.0
        dom_share_pct = _format_percentage(
            _percentage(int(max_freq), total_shapes, level), level
        )
        detail_insights = (
            f"Active ratio bins: {active_bins} of {num_bins}. "
            f"Highest-ratio bins cover {_format_percentage(_percentage(int(sum(high_bins)), total_shapes, level), level)}% of shapes."
        )

    if dom_share < 0.35:
        if len(dom_ranges) == 1:
            range_text = f"the most common range is {dom_range_str}"
        else:
            range_text = f"the most common ranges are {dom_range_str}"
        body = (
            f"The distribution shows a wide spread across {active_bins} different ratio ranges. "
            f"While {range_text} violations per constraint, "
            "the data is spread across many different ratio levels. "
            "This suggests varying levels of constraint effectiveness across different node shapes."
        )
        return _join_nonempty(intro, zero_note, body, detail_insights)
    
    if dom_share >= 0.7:
        if len(dom_ranges) == 1:
            range_text = f"the range of {dom_range_str}"
        else:
            range_text = f"the ranges of {dom_range_str}"
        body = (
            f"Most node shapes ({_format_percentage(_percentage(int(max_freq), total_shapes, level), level)}%) "
            f"fall into {range_text} violations per constraint. "
            "This indicates a consistent pattern where most shapes have similar violation-to-constraint ratios, "
            "suggesting uniform constraint effectiveness across the shapes graph."
        )
        return _join_nonempty(intro, zero_note, body, detail_insights)
    
    # Moderate concentration
    if len(dom_ranges) == 1:
        range_text = f"the range of {dom_range_str}"
    else:
        range_text = f"the ranges of {dom_range_str}"
    body = (
        f"A significant portion ({_format_percentage(_percentage(int(max_freq), total_shapes, level), level)}%) of node shapes "
        f"fall into {range_text} violations per constraint. "
        "While there is some concentration, the distribution also shows variation, "
        "indicating that some shapes have better constraint compliance than others."
    )
    return _join_nonempty(intro, zero_note, body, detail_insights)


def shapes_distribution_per_constraint(
    shapes_overview_service,
    node_shape: str,
    report_uri: str,
    level: str,
    use_llm: bool = False,
    include_category: bool = True,
    shapes_graph_uri: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o",
) -> str:
    return summarize_shape_constraint_distribution_templates(
        shapes_overview_service=shapes_overview_service,
        node_shape=node_shape,
        report_uri=report_uri,
        level=level,
        use_llm=use_llm,
        include_category=include_category,
        shapes_graph_uri=shapes_graph_uri,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
    )


# ============================================================
# SHAPES VIEW – Correlation: constraints vs violations
# ============================================================

def summarize_shapes_correlation_templates(
    shapes_overview_service,
    report_uri: str,
    level: str = "high",
    use_llm: bool = False,
    include_category: bool = True,
    shapes_graph_uri: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o",
) -> str:
    """
    Correlation summary between constraints and violations.
    Analyzes scatter plot data to identify patterns and relationships.
    
    Args:
        use_llm: If True, use LLM to determine dominant category (adds ~1-3s latency).
                 If False (default), use fast keyword-based classification (no overhead).
        include_category: If True, include category/domain information in summary. If False, omit it.
        shapes_graph_uri: URI of shapes graph (for LLM context and to get node shape names)
        llm_api_key: OpenAI API key (if None, uses environment variable)
        llm_model: Model to use (default: gpt-4o)
    """
    try:
        correlation_data = shapes_overview_service.get_correlation_of_constraints_and_violations(
            shapes_graph_uri=SHAPES_GRAPH_URI,
            validation_report_uri=report_uri,
        )
    except Exception:
        if level == "high":
            return (
                "This scatter plot shows the relationship between the number of constraints "
                "and violations per constraint for each node shape. Unable to retrieve data."
            )
        return (
            "Each point represents a NodeShape with constraints on x-axis and violations per constraint on y-axis. "
            "No data available."
        )
    
    if not correlation_data:
        if level == "high":
            return (
                "This scatter plot shows the relationship between the number of constraints "
                "and violations per constraint for each node shape. No data available."
            )
        return (
            "Each point represents a NodeShape with constraints on x-axis and violations per constraint on y-axis. "
            "No node shapes found."
        )
    
    # Get node shape names for category detection
    node_shape_names = []
    if include_category:
        try:
            shapes_uri = shapes_graph_uri or SHAPES_GRAPH_URI
            query = f"""
            PREFIX sh: <http://www.w3.org/ns/shacl#>
            SELECT DISTINCT ?nodeShape
            FROM <{shapes_uri}>
            WHERE {{
                ?nodeShape a sh:NodeShape .
            }}
            """
            response = requests.get(ENDPOINT_URL, params={"query": query, "format": "json"})
            response.raise_for_status()
            results = response.json()["results"]["bindings"]
            node_shape_names = [result["nodeShape"]["value"] for result in results]
        except Exception:
            node_shape_names = []
    
    # Determine category only if include_category is True
    use_semantic = False
    entity_word_plural = "data objects"
    examples = []
    
    if include_category and node_shape_names:
        if use_llm:
            dom_category = dominant_category_llm(
                labels=node_shape_names,
                shapes_graph_uri=shapes_graph_uri or SHAPES_GRAPH_URI,
                validation_report_uri=report_uri,
                api_key=llm_api_key,
                model=llm_model,
            )
            if dom_category:
                entity_word_plural = dom_category
                examples = [clean_label(l) for l in list(node_shape_names)[:2]]
                use_semantic = True
            else:
                dom_category = dominant_category(node_shape_names)
                use_semantic = dom_category != "generic"
                if use_semantic:
                    entity_word_plural = category_display_word(dom_category, plural=True)
                    examples = _example_terms(node_shape_names, dom_category, k=2)
        else:
            dom_category = dominant_category(node_shape_names)
            use_semantic = dom_category != "generic"
            if use_semantic:
                entity_word_plural = category_display_word(dom_category, plural=True)
                examples = _example_terms(node_shape_names, dom_category, k=2)
    
    # Extract data
    constraints = [item.get("num_constraints", 0) for item in correlation_data]
    violations = [item.get("num_violations", 0) for item in correlation_data]
    ratios = []
    for item in correlation_data:
        num_constraints = item.get("num_constraints", 0)
        num_violations = item.get("num_violations", 0)
        if num_constraints > 0:
            ratios.append(num_violations / num_constraints)
        else:
            ratios.append(0.0)
    
    if not constraints or not violations:
        return "No valid correlation data available."
    
    # Calculate statistics
    total_shapes = len(correlation_data)
    avg_constraints = np.mean(constraints) if constraints else 0
    avg_ratio = np.mean(ratios) if ratios else 0
    detail_insights = ""
    if level != "high":
        med_constraints = _percentile_value([float(c) for c in constraints], 50)
        p90_constraints = _percentile_value([float(c) for c in constraints], 90)
        med_ratio = _percentile_value(ratios, 50)
        p90_ratio = _percentile_value(ratios, 90)
        detail_insights = (
            f"Median constraints per shape: {med_constraints:.1f}; "
            f"90th percentile: {p90_constraints:.1f}. "
            f"Median violations per constraint: {med_ratio:.2f}; "
            f"90th percentile: {p90_ratio:.2f}."
        )
    
    # Calculate correlation coefficient
    if len(constraints) > 1 and np.std(constraints) > 0 and np.std(ratios) > 0:
        correlation = np.corrcoef(constraints, ratios)[0, 1]
    else:
        correlation = 0.0
    
    # Identify quadrants using range midpoints (matching frontend logic)
    # Frontend uses: xMid = (xMin + xMax) / 2, yMid = (yMin + yMax) / 2
    x_min = min(constraints) if constraints else 0
    x_max = max(constraints) if constraints else 0
    y_min = min(ratios) if ratios else 0
    y_max = max(ratios) if ratios else 0
    
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    
    high_constraints_indices = [i for i, c in enumerate(constraints) if c >= x_mid]
    low_constraints_indices = [i for i, c in enumerate(constraints) if c < x_mid]
    high_ratios_indices = [i for i, r in enumerate(ratios) if r >= y_mid]
    low_ratios_indices = [i for i, r in enumerate(ratios) if r < y_mid]
    
    # Quadrant analysis - count shapes in each quadrant
    q1 = len(set(high_constraints_indices) & set(high_ratios_indices))  # High constraints, High ratio
    q2 = len(set(low_constraints_indices) & set(high_ratios_indices))   # Low constraints, High ratio
    q3 = len(set(low_constraints_indices) & set(low_ratios_indices))    # Low constraints, Low ratio
    q4 = len(set(high_constraints_indices) & set(low_ratios_indices))   # High constraints, Low ratio
    
    if level == "high":
        intro = (
            "This scatter plot shows the relationship between the number of constraints "
            "and violations per constraint for each node shape. "
            "Each point represents one node shape."
        )
        if include_category and use_semantic and examples:
            classifier_type = "LLM analysis" if use_llm else "NLP classifier"
            intro += (
                f" Based on the shape names, the {classifier_type} suggests that most shapes describe "
                f"{entity_word_plural}, for example {', '.join(examples)}."
            )
    else:
        intro = (
            "Each point represents a NodeShape with number of constraints on the x-axis "
            "and violations per constraint (ratio) on the y-axis. "
            f"The plot contains {total_shapes} node shapes."
        )
        if include_category and use_semantic and examples:
            intro += (
                f" From their labels, many of these shapes appear to describe {entity_word_plural}, "
                f"such as {', '.join(examples)}."
            )
    
    # Interpret correlation
    abs_corr = abs(correlation)
    if abs_corr < 0.3:
        corr_interpretation = "weak"
        corr_strength = "little to no"
    elif abs_corr < 0.7:
        corr_interpretation = "moderate"
        corr_strength = "some"
    else:
        corr_interpretation = "strong"
        corr_strength = "strong"
    
    if correlation > 0:
        corr_direction = "positive"
        corr_meaning = "shapes with more constraints tend to have higher violation ratios"
    elif correlation < 0:
        corr_direction = "negative"
        corr_meaning = "shapes with more constraints tend to have lower violation ratios"
    else:
        corr_direction = "no"
        corr_meaning = "no clear relationship between constraint count and violation ratio"
    
    # Quadrant insights
    dominant_quadrant = max([(q1, "high constraints, high violation ratio"),
                            (q2, "low constraints, high violation ratio"),
                            (q3, "low constraints, low violation ratio"),
                            (q4, "high constraints, low violation ratio")],
                           key=lambda x: x[0])
    
    body = (
        f"The analysis shows a {corr_interpretation} {corr_direction} correlation "
        f"(r={correlation:.2f}), indicating {corr_strength} relationship where {corr_meaning}. "
        f"On average, node shapes have {avg_constraints:.1f} constraints with an average ratio of {avg_ratio:.2f} violations per constraint. "
        f"Most shapes ({dominant_quadrant[0]} out of {total_shapes}, {_format_percentage(_percentage(dominant_quadrant[0], total_shapes, level), level)}%) "
        f"fall into the quadrant with {dominant_quadrant[1]}, suggesting that "
    )
    
    if dominant_quadrant[1] == "high constraints, high violation ratio":
        body += (
            "shapes with many constraints are experiencing proportionally more violations, "
            "which may indicate that complex shapes are harder to satisfy."
        )
    elif dominant_quadrant[1] == "low constraints, high violation ratio":
        body += (
            "shapes with fewer constraints are experiencing disproportionately high violations, "
            "suggesting that even simple constraints are frequently violated."
        )
    elif dominant_quadrant[1] == "low constraints, low violation ratio":
        body += (
            "shapes with fewer constraints tend to have better compliance, "
            "indicating that simpler shapes are easier to satisfy."
        )
    else:  # high constraints, low violation ratio
        body += (
            "shapes with many constraints are maintaining good compliance, "
            "suggesting effective constraint design despite complexity."
        )
    
    return _join_nonempty(intro, body, detail_insights)


def shapes_correlation_constraints_vs_violations(
    shapes_overview_service,
    report_uri: str,
    level: str,
    use_llm: bool = False,
    include_category: bool = True,
    shapes_graph_uri: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o",
) -> str:
    return summarize_shapes_correlation_templates(
        shapes_overview_service=shapes_overview_service,
        report_uri=report_uri,
        level=level,
        use_llm=use_llm,
        include_category=include_category,
        shapes_graph_uri=shapes_graph_uri,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
    )


# ============================================================
# SHAPES VIEW – Diversity and Intensity
# ============================================================

def summarize_shapes_diversity_intensity_templates(
    shapes_overview_service,
    report_uri: str,
    level: str = "high",
    use_llm: bool = False,
    include_category: bool = True,
    shapes_graph_uri: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o",
) -> str:
    """
    Diversity and Intensity summary.
    Analyzes scatter plot of violation entropy (diversity) vs violations per constraint (intensity).
    Uses Shannon entropy to measure diversity of constraint violations.
    
    Args:
        use_llm: If True, use LLM to determine dominant category (adds ~1-3s latency).
                 If False (default), use fast keyword-based classification (no overhead).
        include_category: If True, include category/domain information in summary. If False, omit it.
        shapes_graph_uri: URI of shapes graph (for LLM context and to get node shape names)
        llm_api_key: OpenAI API key (if None, uses environment variable)
        llm_model: Model to use (default: gpt-4o)
    """
    try:
        correlation_data = shapes_overview_service.get_correlation_of_constraints_and_violations(
            shapes_graph_uri=SHAPES_GRAPH_URI,
            validation_report_uri=report_uri,
        )
    except Exception:
        if level == "high":
            return (
                "This scatter plot shows the relationship between violation diversity (entropy) "
                "and violation intensity (violations per constraint) for each node shape. "
                "Unable to retrieve data."
            )
        return (
            "Each point represents a NodeShape with violation entropy on x-axis and violations per constraint on y-axis. "
            "No data available."
        )
    
    if not correlation_data:
        if level == "high":
            return (
                "This scatter plot shows the relationship between violation diversity (entropy) "
                "and violation intensity (violations per constraint) for each node shape. "
                "No data available."
            )
        return (
            "Each point represents a NodeShape with violation entropy on x-axis and violations per constraint on y-axis. "
            "No node shapes found."
        )
    
    # Get node shape names for category detection
    node_shape_names = []
    if include_category:
        try:
            shapes_uri = shapes_graph_uri or SHAPES_GRAPH_URI
            query = f"""
            PREFIX sh: <http://www.w3.org/ns/shacl#>
            SELECT DISTINCT ?nodeShape
            FROM <{shapes_uri}>
            WHERE {{
                ?nodeShape a sh:NodeShape .
            }}
            """
            response = requests.get(ENDPOINT_URL, params={"query": query, "format": "json"})
            response.raise_for_status()
            results = response.json()["results"]["bindings"]
            node_shape_names = [result["nodeShape"]["value"] for result in results]
        except Exception:
            node_shape_names = []
    
    # Determine category only if include_category is True
    use_semantic = False
    entity_word_plural = "data objects"
    examples = []
    
    if include_category and node_shape_names:
        if use_llm:
            dom_category = dominant_category_llm(
                labels=node_shape_names,
                shapes_graph_uri=shapes_graph_uri or SHAPES_GRAPH_URI,
                validation_report_uri=report_uri,
                api_key=llm_api_key,
                model=llm_model,
            )
            if dom_category:
                entity_word_plural = dom_category
                examples = [clean_label(l) for l in list(node_shape_names)[:2]]
                use_semantic = True
            else:
                dom_category = dominant_category(node_shape_names)
                use_semantic = dom_category != "generic"
                if use_semantic:
                    entity_word_plural = category_display_word(dom_category, plural=True)
                    examples = _example_terms(node_shape_names, dom_category, k=2)
        else:
            dom_category = dominant_category(node_shape_names)
            use_semantic = dom_category != "generic"
            if use_semantic:
                entity_word_plural = category_display_word(dom_category, plural=True)
                examples = _example_terms(node_shape_names, dom_category, k=2)
    
    # Extract data
    entropies = [item.get("violation_entropy", 0.0) for item in correlation_data]
    constraints = [item.get("num_constraints", 0) for item in correlation_data]
    violations = [item.get("num_violations", 0) for item in correlation_data]
    ratios = []
    for item in correlation_data:
        num_constraints = item.get("num_constraints", 0)
        num_violations = item.get("num_violations", 0)
        if num_constraints > 0:
            ratios.append(num_violations / num_constraints)
        else:
            ratios.append(0.0)
    
    if not entropies or not ratios:
        return "No valid diversity and intensity data available."
    
    total_shapes = len(correlation_data)
    
    # Calculate statistics
    avg_entropy = np.mean(entropies) if entropies else 0
    avg_ratio = np.mean(ratios) if ratios else 0
    max_entropy = max(entropies) if entropies else 0
    min_entropy = min(entropies) if entropies else 0
    detail_insights = ""
    if level != "high":
        median_entropy = _percentile_value(entropies, 50)
        p90_entropy = _percentile_value(entropies, 90)
        median_ratio = _percentile_value(ratios, 50)
        p90_ratio = _percentile_value(ratios, 90)
        detail_insights = (
            f"Median entropy: {median_entropy:.2f}; 90th percentile: {p90_entropy:.2f}. "
            f"Median violations per constraint: {median_ratio:.2f}; "
            f"90th percentile: {p90_ratio:.2f}."
        )
    
    # Calculate correlation
    if len(entropies) > 1 and np.std(entropies) > 0 and np.std(ratios) > 0:
        correlation = np.corrcoef(entropies, ratios)[0, 1]
    else:
        correlation = 0.0
    
    # Identify quadrants using range midpoints (matching frontend logic)
    # Frontend uses: xMid = (xMin + xMax) / 2, yMid = (yMin + yMax) / 2
    x_min = min(entropies) if entropies else 0
    x_max = max(entropies) if entropies else 0
    y_min = min(ratios) if ratios else 0
    y_max = max(ratios) if ratios else 0
    
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    
    high_entropy_indices = [i for i, e in enumerate(entropies) if e >= x_mid]
    low_entropy_indices = [i for i, e in enumerate(entropies) if e < x_mid]
    high_ratio_indices = [i for i, r in enumerate(ratios) if r >= y_mid]
    low_ratio_indices = [i for i, r in enumerate(ratios) if r < y_mid]
    
    # Quadrant analysis - count shapes in each quadrant
    q1 = len(set(high_entropy_indices) & set(high_ratio_indices))  # High diversity, High intensity
    q2 = len(set(low_entropy_indices) & set(high_ratio_indices))    # Low diversity, High intensity
    q3 = len(set(low_entropy_indices) & set(low_ratio_indices))    # Low diversity, Low intensity
    q4 = len(set(high_entropy_indices) & set(low_ratio_indices))   # High diversity, Low intensity
    
    # Entropy interpretation
    # Shannon entropy ranges: 0 (no diversity, all violations from one constraint type)
    # to log2(n) where n is number of constraint types (maximum diversity)
    # High entropy = diverse violations across many constraint types
    # Low entropy = violations concentrated in few constraint types
    
    if level == "high":
        intro = (
            "This scatter plot shows the relationship between violation diversity and violation intensity. "
            "Diversity (x-axis) is measured using Shannon entropy, which indicates how evenly violations "
            "are distributed across different constraint types. Higher entropy means violations are spread "
            "across many constraint types, while lower entropy means violations are concentrated in few types. "
            "Intensity (y-axis) shows violations per constraint, indicating how severe the violation problem is. "
            "Each point represents one node shape."
        )
        if include_category and use_semantic and examples:
            classifier_type = "LLM analysis" if use_llm else "NLP classifier"
            intro += (
                f" Based on the shape names, the {classifier_type} suggests that most shapes describe "
                f"{entity_word_plural}, for example {', '.join(examples)}."
            )
    else:
        intro = (
            "Each point represents a NodeShape with violation entropy (Shannon entropy of constraint component distribution) "
            f"on the x-axis and violations per constraint on the y-axis. The plot contains {total_shapes} node shapes. "
            "Entropy ranges from 0 (all violations from one constraint type) to higher values (violations spread across many types)."
        )
        if include_category and use_semantic and examples:
            intro += (
                f" From their labels, many of these shapes appear to describe {entity_word_plural}, "
                f"such as {', '.join(examples)}."
            )
    
    # Interpret entropy levels
    if avg_entropy < 1.0:
        entropy_level = "low"
        entropy_meaning = "violations are concentrated in a few constraint types"
    elif avg_entropy < 2.5:
        entropy_level = "moderate"
        entropy_meaning = "violations show moderate diversity across constraint types"
    else:
        entropy_level = "high"
        entropy_meaning = "violations are spread across many different constraint types"
    
    # Interpret correlation
    abs_corr = abs(correlation)
    if abs_corr < 0.3:
        corr_interpretation = "weak"
    elif abs_corr < 0.7:
        corr_interpretation = "moderate"
    else:
        corr_interpretation = "strong"
    
    if correlation > 0:
        corr_meaning = "shapes with more diverse violations tend to have higher violation intensity"
    elif correlation < 0:
        corr_meaning = "shapes with more diverse violations tend to have lower violation intensity"
    else:
        corr_meaning = "no clear relationship between violation diversity and intensity"
    
    # Quadrant insights - find the quadrant with the most shapes
    # Debug: print quadrant counts to verify
    quadrant_counts = [
        (q1, "high diversity, high intensity"),
                            (q2, "low diversity, high intensity"),
                            (q3, "low diversity, low intensity"),
        (q4, "high diversity, low intensity")
    ]
    dominant_quadrant = max(quadrant_counts, key=lambda x: x[0])
    
    # Verify all quadrants sum to total_shapes (for debugging)
    total_in_quadrants = q1 + q2 + q3 + q4
    if total_in_quadrants != total_shapes:
        # This shouldn't happen, but if it does, log it
        import sys
        print(f"Warning: Quadrant counts don't sum to total shapes. Total: {total_shapes}, Sum: {total_in_quadrants}", file=sys.stderr)
    
    body = (
        f"The analysis reveals {entropy_level} average diversity (entropy={avg_entropy:.2f}), "
        f"indicating that {entropy_meaning}. "
        f"The average violation intensity is {avg_ratio:.2f} violations per constraint. "
        f"There is a {corr_interpretation} correlation (r={correlation:.2f}) between diversity and intensity, "
        f"suggesting that {corr_meaning}. "
        f"Most shapes ({dominant_quadrant[0]} out of {total_shapes}, {_format_percentage(_percentage(dominant_quadrant[0], total_shapes, level), level)}%) "
        f"fall into the quadrant with {dominant_quadrant[1]}, which indicates that "
    )
    
    if dominant_quadrant[1] == "high diversity, high intensity":
        body += (
            "many shapes experience both diverse violations across multiple constraint types "
            "and high violation rates. This suggests systemic data quality issues affecting "
            "multiple aspects of the constraints simultaneously."
        )
    elif dominant_quadrant[1] == "low diversity, high intensity":
        body += (
            "many shapes have violations concentrated in specific constraint types but with high intensity. "
            "This pattern suggests targeted issues with particular constraint types that are frequently violated, "
            "which could indicate systematic problems with specific validation rules."
        )
    elif dominant_quadrant[1] == "low diversity, low intensity":
        body += (
            "many shapes have both low diversity and low intensity of violations. "
            "This is the most favorable pattern, indicating that when violations occur, "
            "they are limited to few constraint types and occur at low rates."
        )
    else:  # high diversity, low intensity
        body += (
            "many shapes show diverse violations across many constraint types but at low intensity. "
            "This suggests that while violations affect many different constraint types, "
            "the overall violation rate per constraint is manageable."
        )
    
    # Add entropy-specific insights
    if level == "high":
        median_entropy = np.median(entropies) if entropies else 0
        entropy_insight = (
            f" The entropy range spans from {min_entropy:.2f} to {max_entropy:.2f}, "
            f"with a median of {median_entropy:.2f}. "
        )
        if max_entropy - min_entropy > 2.0:
            entropy_insight += (
                "The wide range indicates significant variation in how violations are distributed "
                "across constraint types among different shapes."
            )
        else:
            entropy_insight += (
                "The narrow range suggests consistent patterns in violation distribution across shapes."
            )
        body += entropy_insight
    
    return _join_nonempty(intro, body, detail_insights)


def shapes_diversity_intensity(
    shapes_overview_service,
    report_uri: str,
    level: str,
    use_llm: bool = False,
    include_category: bool = True,
    shapes_graph_uri: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o",
) -> str:
    return summarize_shapes_diversity_intensity_templates(
        shapes_overview_service=shapes_overview_service,
        report_uri=report_uri,
        level=level,
        use_llm=use_llm,
        include_category=include_category,
        shapes_graph_uri=shapes_graph_uri,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
    )


# ============================================================
# SHAPES VIEW – Heatmap summary (placeholder)
# ============================================================

def shapes_heatmap_summary(
    shapes_overview_service,
    report_uri: str,
    level: str,
) -> str:
    """
    Heatmap summary placeholder.
    TODO: Implement heatmap analysis if needed.
    """
    if level == "high":
        return (
            "This heatmap visualization shows the relationship between different aspects of shapes and violations. "
            "Implementation pending."
        )
    return (
        "This heatmap shows a matrix representation of shape and violation relationships. "
        "Implementation pending."
    )


# ============================================================
# SHAPES VIEW – Property contribution summary (placeholder)
# ============================================================

def shapes_property_contribution(
    shapes_overview_service,
    node_shape: str,
    report_uri: str,
    level: str,
    top_k: int = 3,
) -> str:
    """
    Property contribution summary placeholder.
    TODO: Implement property contribution analysis if needed.
    """
    if level == "high":
        return (
            f"This view shows which property shapes contribute most to violations for the node shape {clean_label(node_shape)}. "
            "Implementation pending."
        )
    return (
        f"This analysis identifies the top {top_k} property shapes contributing to violations "
        f"for NodeShape {clean_label(node_shape)}. Implementation pending."
    )

