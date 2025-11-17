from __future__ import annotations
from typing import Dict, Iterable, List
import re

import numpy as np
import inflect
import spacy
from jinja2 import Template

# ============================================================
# Lightweight NLP
# ============================================================

_p = inflect.engine()

try:
    # We only need tagging + lemma, keep it light
    _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])
except Exception:
    _nlp = spacy.blank("en")


# ============================================================
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


def _last_segment(s: str) -> str:
    if "://" in s or "#" in s or "/" in s:
        return _IRI_LAST_SEG_RE.split(s)[-1]
    if ":" in s:
        return s.split(":", 1)[-1]
    return s


def _doc_for_label(raw: str):
    base = _last_segment(raw).replace("_", " ").replace("-", " ")
    base = re.sub(r"(?<=[a-z0-9])([A-Z])", r" \1", base)
    return _nlp(base)


def analyze_label_semantics(raw: str) -> str:
    """
    Classify a label 
    """
    doc = _doc_for_label(raw)
    lemmas = {t.lemma_.lower() for t in doc if t.is_alpha}
    texts = {t.text.lower() for t in doc if t.is_alpha}
    bag = lemmas | texts

    if bag & PERSON_KEYWORDS:
        return "person"
    if bag & ORG_KEYWORDS:
        return "org"
    if bag & TIME_KEYWORDS:
        return "time"
    if bag & LOC_KEYWORDS:
        return "location"
    if bag & ID_KEYWORDS:
        return "identifier"
    if bag & NUM_KEYWORDS:
        return "numeric"
    if bag & TEXT_KEYWORDS:
        return "text"
    return "generic"


def category_display_word(category: str, plural: bool = False) -> str:
    """
    Map semantic category to a human phrase.
    """
    mapping = {
        "person": "person record",
        "org": "organization record",
        "time": "date field",
        "location": "address or location",
        "identifier": "identifier field",
        "numeric": "numeric field",
        "text": "text field",
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
        if t.pos_ in ("PROPN", "NOUN"):
            toks.append(t.text.capitalize())
        else:
            toks.append(t.text.lower())
    out = re.sub(r"\s+", " ", " ".join(toks)).strip()
    return out or raw


def clean_pairs(pairs: Iterable[tuple[str, int]]) -> List[tuple[str, int]]:
    return [(clean_label(n), int(c)) for (n, c) in pairs]


def dominant_category(labels: Iterable[str]) -> str:
    """
    Return the *true* dominant category (including 'generic').

    """
    counts: Dict[str, int] = {}
    for lab in labels:
        cat = analyze_label_semantics(lab)
        counts[cat] = counts.get(cat, 0) + 1
    if not counts:
        return "generic"
    # Strict argmax (generic may win)
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _example_terms(labels: Iterable[str], category: str, k: int = 2) -> List[str]:
    """
    Use spaCy to surface a couple of representative labels for the given category.

    If the category is 'generic' or we don't find matches,
    we just show a couple of cleaned labels (but only if the caller wants them).
    """
    labels = list(labels)
    examples: List[str] = []
    if category != "generic":
        for lab in labels:
            if analyze_label_semantics(lab) == category:
                examples.append(clean_label(lab))
            if len(examples) >= k:
                break
    if not examples:
        examples = [clean_label(l) for l in labels[:k]]
    return examples


def _percentage(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return round(100.0 * part / whole, 1)



# ============================================================
# HOME – Node Shape histogram
# ============================================================

def summarize_nodeshape_with_templates(
    homepage_service,
    shapes_graph_uri: str,
    report_uri: str,
    level: str = "high",
) -> str:
    """
    'Violations per Node Shape' histogram.

    Each bar = how many node shapes fall into a violation interval.
    """
    shape_data = homepage_service.get_violations_per_node_shape(
        shapes_graph_uri=shapes_graph_uri,
        validation_report_uri=report_uri,
    )
    counts = [int(item["NumViolations"]) for item in shape_data] if shape_data else []
    labels_ns = [item["NodeShapeName"] for item in shape_data] if shape_data else []

    total_shapes = len(counts)
    violated_counts = [c for c in counts if c > 0]
    violated_shapes = len(violated_counts)

    dom_category = dominant_category(labels_ns)
    use_semantic = dom_category != "generic"
    if use_semantic:
        entity_word_plural = category_display_word(dom_category, plural=True)
        examples = _example_terms(labels_ns, dom_category, k=2)
    else:
        entity_word_plural = "data objects"
        examples = []

    if level == "high":
        intro = (
            "Node shape refers to a group of data objects for which certain rules apply. "
            "This chart groups these rule sets by how many violations they have."
        )
        if use_semantic and examples:
            intro += (
                f" Based on the shape names, the NLP classifier suggests that most shapes describe "
                f"{entity_word_plural}, for example {', '.join(examples)}."
            )
    else:
        intro = (
            "Each bar aggregates SHACL NodeShapes by their total number of validation results. "
            "The bar height is the number of NodeShapes whose violation count falls into that interval."
        )
        if use_semantic and examples:
            intro += (
                f" From their labels, many of these shapes appear to describe {entity_word_plural}, "
                f"such as {', '.join(examples)}."
            )

    if total_shapes == 0:
        return intro + " No node shapes are present in the shapes graph."

    if violated_shapes == 0:
        return intro + " No node shapes are violated."
    else:
        perc = _percentage(violated_shapes, total_shapes)
        perc_sentence = f"{perc}% of node shapes are violated."

    dist = homepage_service.distribution_of_violations_per_shape(
        shapes_graph_uri=shapes_graph_uri,
        validation_report_uri=report_uri,
    )
    labels = dist.get("labels", [])
    freqs = dist.get("datasets", [{}])[0].get("data", [])

    if not labels or not freqs:
        if violated_counts:
            min_v, max_v = min(violated_counts), max(violated_counts)
            spread_sentence = (
                f"They show a wide spread of violation counts, ranging from {min_v} to {max_v} violations. "
                "Some shapes trigger only very few issues, while others suffer from much larger rule problems."
            )
            return " ".join([intro, perc_sentence, spread_sentence])
        return " ".join([intro, perc_sentence])

    total_bins = sum(freqs) or 1
    dom_idx = int(np.argmax(freqs))
    dom_label = labels[dom_idx]
    dom_share = freqs[dom_idx] / total_bins

    try:
        low_s, high_s = re.split(r"[-–]", dom_label)
        dom_range = f"{int(low_s)}–{int(high_s)}"
        dom_high = int(high_s)
    except Exception:
        dom_range = dom_label
        dom_high = max(violated_counts) if violated_counts else 0

    vmin, vmax = (min(violated_counts), max(violated_counts)) if violated_counts else (0, 0)

    if counts:
        max_val = max(counts)
        idx_max = counts.index(max_val)
        most_violated_name = clean_label(labels_ns[idx_max])
        share_max = max_val / (sum(counts) or 1)
    else:
        most_violated_name, share_max = "", 0.0

    outlier_threshold = dom_high * 1.5 if dom_high > 0 else 0
    has_outliers = vmax > outlier_threshold and outlier_threshold > 0

    if dom_share < 0.35:
        body = (
            f"They show a wide spread of violation counts, ranging from {vmin} to {vmax} violations. "
            "This means some shapes trigger only very few issues, while others suffer from much larger rule problems. "
            "The differences suggest varying data quality across shape types."
        )
        return " ".join([intro, perc_sentence, body])

    if share_max > 0.35 and vmax >= dom_high:
        perc_out = _percentage(max_val, sum(counts) or 1)
        body = (
            f"While most of them fall into the range of {dom_range} violations, "
            f"the shape {most_violated_name} is responsible for about {perc_out}% of all issues and should be reviewed first."
        )
        return " ".join([intro, perc_sentence, body])

    if not has_outliers:
        body = (
            f"Most of them fall into the range of {dom_range} violations, "
            "meaning that many data entities of this type likely share similar rule issues. "
            "This indicates that most of the problems are concentrated and probably relate to a recurring pattern in these shapes."
        )
        return " ".join([intro, perc_sentence, body])

    body = (
        f"While most of them fall into the range of {dom_range} violations, "
        f"a few shapes show unusually high counts with up to {vmax} violations. "
        "These outliers likely point to specific modeling or data-entry issues related to those shapes and should be reviewed first."
    )
    return " ".join([intro, perc_sentence, body])


def home_nodeshape_hist(
    homepage_service,
    shapes_graph_uri: str,
    report_uri: str,
    level: str,
) -> str:
    return summarize_nodeshape_with_templates(
        homepage_service=homepage_service,
        shapes_graph_uri=shapes_graph_uri,
        report_uri=report_uri,
        level=level,
    )


# ============================================================
# HOME – Path histogram
# ============================================================

def summarize_path_hist_with_templates(
    homepage_service,
    report_uri: str,
    level: str = "high",
) -> str:
    """
    Summary for path histogram.
    TODO: get real path violation counts, bin them, detect heavy hitters, etc.
    """
    if level == "high":
        explanation = (
            "This chart groups data fields (paths) by how many violations they cause."
        )
    else:
        explanation = (
            "Each bar corresponds to a bin of violation counts for sh:resultPath values."
        )
    return explanation + " Actual stats will be added later (once implemented)."


def home_path_hist(
    homepage_service,
    report_uri: str,
    level: str,
) -> str:
    return summarize_path_hist_with_templates(
        homepage_service=homepage_service,
        report_uri=report_uri,
        level=level,
    )


# ============================================================
# HOME – Focus node histogram
# ============================================================

def summarize_focusnode_hist_with_templates(
    homepage_service,
    report_uri: str,
    level: str = "high",
) -> str:
    """
    Summary for focus node histogram.
    TODO: need to calculate bins properly + maybe highlight extreme nodes.
    """
    if level == "high":
        explanation = (
            "This chart groups records (focus nodes) by how many rule violations they have."
        )
    else:
        explanation = (
            "Each bar shows how many focus nodes fall into a given range of violation counts."
        )
    return explanation + " More detailed logic still missing (TODO)."


def home_focusnode_hist(
    homepage_service,
    report_uri: str,
    level: str,
) -> str:
    return summarize_focusnode_hist_with_templates(
        homepage_service=homepage_service,
        report_uri=report_uri,
        level=level,
    )


# ============================================================
# HOME – Constraint component histogram
# ============================================================

def summarize_constraint_hist_with_templates(
    homepage_service,
    report_uri: str,
    level: str = "high",
) -> str:
    """
    Summary for constraint component histogram.
    TODO: figure out which constraints have most violations + sort them.
    """
    if level == "high":
        explanation = (
            "This chart groups rule types (constraint components) by how often they are violated."
        )
    else:
        explanation = (
            "Each bar aggregates sh:sourceConstraintComponent values by their violation counts."
        )
    return explanation + " Stats logic will be added later."


def home_constraint_hist(
    homepage_service,
    report_uri: str,
    level: str,
) -> str:
    return summarize_constraint_hist_with_templates(
        homepage_service=homepage_service,
        report_uri=report_uri,
        level=level,
    )


# ============================================================
# HOME – Path Toplist
# ============================================================

def home_paths_top(
    homepage_service,
    report_uri: str,
    level: str,
    top_k: int = 3,
) -> str:
    """
    Top violated paths.
    TODO: actually fetch paths, sort by violation counts, compute percentages, etc.
    """
    if level == "high":
        explanation = (
            "This view lists the data fields that are responsible for most of the violations."
        )
    else:
        explanation = (
            "This top list shows the sh:resultPath values with the highest violation counts."
        )
    return explanation + f" TODO: implement real top {top_k} logic."


# ============================================================
# SHAPES VIEW – Distribution per constraint inside a NodeShape
# ============================================================

def summarize_shape_constraint_distribution_templates(
    shapes_overview_service,
    node_shape: str,
    report_uri: str,
    level: str = "high",
) -> str:
    """
    Per-constraint distribution inside a NodeShape.
    TODO: bucket property shapes by violation count, highlight spikes, etc.
    """
    if level == "high":
        explanation = (
            "This chart groups the rules inside the selected rule set by their violation counts."
        )
    else:
        explanation = (
            "Each bar corresponds to a bin of violation counts for PropertyShapes within the NodeShape."
        )
    return explanation + " (TODO: real distribution logic goes here)."


def shapes_distribution_per_constraint(
    shapes_overview_service,
    node_shape: str,
    report_uri: str,
    level: str,
) -> str:
    return summarize_shape_constraint_distribution_templates(
        shapes_overview_service=shapes_overview_service,
        node_shape=node_shape,
        report_uri=report_uri,
        level=level,
    )


# ============================================================
# SHAPES VIEW – Correlation: constraints vs violations
# ============================================================

def summarize_shapes_correlation_templates(
    shapes_overview_service,
    report_uri: str,
    level: str = "high",
) -> str:
    """
    Correlation summary.
    TODO: compute actual scatterplot stats (median split, correlation, etc).
    """
    if level == "high":
        explanation = (
            "Each point represents a rule group, relating rule count to average violations per rule."
        )
    else:
        explanation = (
            "Each point represents a NodeShape with constraints on x-axis and violations per constraint on y-axis."
        )
    return explanation + " (TODO: do real math here later)."


def shapes_correlation_constraints_vs_violations(
    shapes_overview_service,
    report_uri: str,
    level: str,
) -> str:
    return summarize_shapes_correlation_templates(
        shapes_overview_service=shapes_overview_service,
        report_uri=report_uri,
        level=level,
    )


