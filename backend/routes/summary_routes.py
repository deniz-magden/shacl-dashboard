from __future__ import annotations

from flask import Blueprint, request, jsonify

from functions import homepage_service, shapes_overview_service
from functions.summary_service import (
    home_nodeshape_hist,
    home_path_hist,
    home_focusnode_hist,
    home_constraint_hist,
    home_paths_top,
    shapes_distribution_per_constraint,
    shapes_correlation_constraints_vs_violations,
    shapes_diversity_intensity,
    shapes_heatmap_summary,
    shapes_property_contribution,
)

summary_bp = Blueprint("summary_bp", __name__, url_prefix="/api/summaries")


def _level_param() -> str:
    level = request.args.get("level", "high").strip().lower()
    return "low" if level == "low" else "high"


# ============================================================
#                          HOME VIEW
# ============================================================

@summary_bp.get("/home/nodeshape")
def summarize_home_nodeshape():
    level = _level_param()
    shapes_uri = request.args.get("shapes_graph_uri", default = "http://ex.org/ShapesGraph")
    report_uri = request.args.get("validation_report_uri", default = "http://ex.org/ValidationReport")
    text = home_nodeshape_hist(homepage_service, shapes_uri, report_uri, level)
    return jsonify({"level": level, "summary": text})


@summary_bp.get("/home/path")
def summarize_home_path():
    level = _level_param()
    report_uri = request.args.get("validation_report_uri", default = "http://ex.org/ValidationReport")
    text = home_path_hist(homepage_service, report_uri, level)
    return jsonify({"level": level, "summary": text})


@summary_bp.get("/home/focus-node")
def summarize_home_focus():
    level = _level_param()
    report_uri = request.args.get("validation_report_uri", default = "http://ex.org/ValidationReport")
    text = home_focusnode_hist(homepage_service, report_uri, level)
    return jsonify({"level": level, "summary": text})


@summary_bp.get("/home/constraint")
def summarize_home_constraint():    
    level = _level_param()
    report_uri = request.args.get("validation_report_uri", default = "http://ex.org/ValidationReport")
    text = home_constraint_hist(homepage_service, report_uri, level)
    return jsonify({"level": level, "summary": text})


@summary_bp.get("/home/path/top")
def summarize_home_path_top():
    level = _level_param()
    report_uri = request.args.get("validation_report_uri", default = "http://ex.org/ValidationReport")
    top_k = int(request.args.get("top_k", 3))
    text = home_paths_top(homepage_service, report_uri, level, top_k)
    return jsonify({"level": level, "summary": text})


# ============================================================
#                       SHAPES VIEW
# ============================================================

@summary_bp.get("/shapes/distribution-constraint")
def summarize_shapes_distribution():
    level = _level_param()
    node_shape = request.args.get("node_shape")
    if not node_shape:
        return jsonify({"error": "node_shape is required"}), 400
    report_uri = request.args.get("validation_report_uri", default = "http://ex.org/ValidationReport")
    text = shapes_distribution_per_constraint(
        shapes_overview_service, node_shape, report_uri, level
    )
    return jsonify({"level": level, "summary": text})


@summary_bp.get("/shapes/correlation")
def summarize_shapes_correlation():
    level = _level_param()
    report_uri = request.args.get("validation_report_uri", default = "http://ex.org/ValidationReport")
    text = shapes_correlation_constraints_vs_violations(
        shapes_overview_service, report_uri, level
    )
    return jsonify({"level": level, "summary": text})


@summary_bp.get("/shapes/diversity-intensity")
def summarize_shapes_diversity_intensity_route():
    level = _level_param()
    report_uri = request.args.get("validation_report_uri", default = "http://ex.org/ValidationReport")
    text = shapes_diversity_intensity(
        shapes_overview_service, report_uri, level
    )
    return jsonify({"level": level, "summary": text})


@summary_bp.get("/shapes/heatmap")
def summarize_shapes_heatmap():
    level = _level_param()
    report_uri = request.args.get("validation_report_uri", default = "http://ex.org/ValidationReport")
    text = shapes_heatmap_summary(
        shapes_overview_service, report_uri, level
    )
    return jsonify({"level": level, "summary": text})


@summary_bp.get("/shapes/contribution")
def summarize_shapes_contribution():
    level = _level_param()
    node_shape = request.args.get("node_shape")
    if not node_shape:
        return jsonify({"error": "node_shape is required"}), 400
    report_uri = request.args.get("validation_report_uri", default = "http://ex.org/ValidationReport")
    top_k = int(request.args.get("top_k", 3))
    text = shapes_property_contribution(
        shapes_overview_service, node_shape, report_uri, level, top_k
    )
    return jsonify({"level": level, "summary": text})
