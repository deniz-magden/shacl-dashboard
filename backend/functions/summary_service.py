from __future__ import annotations
from typing import Iterable, List
import re

# ============================================================
# Helper Functions
# ============================================================

_IRI_LAST_SEG_RE = re.compile(r"[#/](?!.*[/#])")


def _last_segment(s: str) -> str:
    """
    Just grabs the last part of an IRI. 
    TODO: probably need something smarter here later.
    """
    if "://" in s or "#" in s or "/" in s:
        return _IRI_LAST_SEG_RE.split(s)[-1]
    if ":" in s:
        return s.split(":", 1)[-1]
    return s


def analyze_label_semantics(raw: str) -> str:
    """
    Placeholder for semantic/NLP classification.
    TODO: hook up spaCy properly when I have time.
    """
    return "generic"


def category_display_word(category: str, plural: bool = False) -> str:
    """
    Returns something human-readable for now.
    TODO: eventually integrate inflect or something so pluralization isn't hacked.
    """
    base = "data object"
    if plural:
        return "data objects"
    return base


def clean_label(raw: str) -> str:
    """
    Cleans up a QName-ish string.
    TODO: this should probably use spaCy later but this is fine for now.
    """
    base = _last_segment(raw).replace("_", " ").replace("-", " ")
    base = re.sub(r"(?<=[a-z0-9])([A-Z])", r" \1", base)
    out = re.sub(r"\s+", " ", base).strip()
    return out or raw


def clean_pairs(pairs: Iterable[tuple[str, int]]) -> List[tuple[str, int]]:
    """
    Keep this around for later.
    TODO: maybe sort or normalize or something when real logic exists.
    """
    return [(clean_label(n), int(c)) for (n, c) in pairs]


def dominant_category(labels: Iterable[str]) -> str:
    """
    TODO: run all labels through analyze_label_semantics and count them.
          (right now everything is just 'generic')
    """
    return "generic"


# ============================================================
# HOME VIEW – Node Shape histogram
# ============================================================

def summarize_nodeshape_with_templates(
    homepage_service,
    shapes_graph_uri: str,
    report_uri: str,
    level: str = "high",
) -> str:
    """
    Summary for node shape histogram.
    TODO: actually compute the histogram stuff (min/max/cluster/outliers etc).
          For now just sending boilerplate text.
    """
    if level == "high":
        explanation = (
            "This chart groups rule sets by how many violations they have per node shape."
        )
    else:
        explanation = (
            "This view aggregates SHACL NodeShapes by their total number of validation results."
        )
    return explanation + " More detailed analysis coming once I implement it."


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
# HOME VIEW – Path histogram
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
# HOME VIEW – Focus node histogram
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
# HOME VIEW – Constraint component histogram
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
# HOME VIEW– Path Toplist
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


# ============================================================
# SHAPES VIEW – Diversity vs intensity
# ============================================================

def summarize_diversity_intensity_templates(
    shapes_overview_service,
    report_uri: str,
    level: str = "high",
) -> str:
    """
    TODO: compute entropy + violation intensity + categorize points.
    """
    if level == "high":
        explanation = (
            "This chart compares how many different rules are affected with how strong the violations are."
        )
    else:
        explanation = (
            "X-axis: entropy (diversity), Y-axis: violations per constraint (intensity)."
        )
    return explanation + " (TODO: finish diversity–intensity analysis)."


def shapes_diversity_intensity(
    shapes_overview_service,
    report_uri: str,
    level: str,
) -> str:
    return summarize_diversity_intensity_templates(
        shapes_overview_service=shapes_overview_service,
        report_uri=report_uri,
        level=level,
    )


# ============================================================
# SHAPES VIEW – Heatmap
# ============================================================

def summarize_heatmap_templates(
    shapes_overview_service,
    report_uri: str,
    level: str = "high",
) -> str:
    """
    Heatmap summary.
    TODO: load matrix, find hot rows/columns, do some basic heuristics.
    """
    if level == "high":
        explanation = (
            "In this heatmap, each row represents a rule group and each column represents a data field."
        )
    else:
        explanation = (
            "Heatmap shows NodeShapes as rows and PropertyShapes as columns with violation counts."
        )
    return explanation + " TODO: hotspot detection not implemented yet."


def shapes_heatmap_summary(
    shapes_overview_service,
    report_uri: str,
    level: str,
) -> str:
    return summarize_heatmap_templates(
        shapes_overview_service=shapes_overview_service,
        report_uri=report_uri,
        level=level,
    )


# ============================================================
# SHAPES VIEW – PropertyShape contribution inside NodeShape
# ============================================================

def summarize_property_contribution_templates(
    shapes_overview_service,
    node_shape: str,
    report_uri: str,
    level: str = "high",
    top_k: int = 3,
) -> str:
    """
    TODO: find top-k property shapes for this node shape + calculate percentages.
    """
    if level == "high":
        explanation = (
            "This chart shows which data fields contribute most to the violations inside the selected NodeShape."
        )
    else:
        explanation = (
            "This view aggregates violation counts per PropertyShape for the selected NodeShape."
        )
    return explanation + f" TODO: real top-{top_k} contribution calculation goes here."


def shapes_property_contribution(
    shapes_overview_service,
    node_shape: str,
    report_uri: str,
    level: str,
    top_k: int = 3,
) -> str:
    return summarize_property_contribution_templates(
        shapes_overview_service=shapes_overview_service,
        node_shape=node_shape,
        report_uri=report_uri,
        level=level,
        top_k=top_k,
    )
