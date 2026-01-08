from __future__ import annotations

import os
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
    # shapes_heatmap_summary,  # TODO: Not yet implemented
    # shapes_property_contribution,  # TODO: Not yet implemented
)

summary_bp = Blueprint("summary_bp", __name__)


def _level_param() -> str:
    level = request.args.get("level", "high").strip().lower()
    return "low" if level == "low" else "high"


def _use_llm_param() -> bool:
    """
    Parse use_llm parameter from request.
    
    Defaults to False to avoid LLM overhead (~1-3s latency).
    Users can explicitly set use_llm=true to enable LLM-based category detection.
    """
    use_llm = request.args.get("use_llm", "false").strip().lower()
    return use_llm in ("true", "1", "yes")


def _include_category_param() -> bool:
    """Parse include_category parameter from request."""
    include_category = request.args.get("include_category", "true").strip().lower()
    return include_category in ("true", "1", "yes")

# ============================================================
#                          HOME VIEW
# ============================================================

@summary_bp.route("/summaries/home/nodeshape", methods=['GET'])
def summarize_home_nodeshape():
    level = _level_param()
    use_llm = _use_llm_param()
    include_category = _include_category_param()
    shapes_uri = request.args.get("shapes_graph_uri", default = "http://ex.org/ShapesGraph")
    report_uri = request.args.get("validation_report_uri", default = "http://ex.org/ValidationReport")
    
    # Optional LLM configuration
    llm_api_key = request.args.get("llm_api_key", default=None)
    llm_model = request.args.get("llm_model", default="gpt-4o")
    
    text = home_nodeshape_hist(
        homepage_service, 
        shapes_uri, 
        report_uri, 
        level,
        use_llm=use_llm,
        include_category=include_category,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
    )
    return jsonify({
        "level": level, 
        "summary": text,
        "use_llm": use_llm,
        "include_category": include_category
    })


@summary_bp.route("/summaries/home/path", methods=['GET'])
def summarize_home_path():
    level = _level_param()
    use_llm = _use_llm_param()
    include_category = _include_category_param()
    shapes_uri = request.args.get("shapes_graph_uri", default=None)
    report_uri = request.args.get("validation_report_uri", default = "http://ex.org/ValidationReport")
    
    # Optional LLM configuration
    llm_api_key = request.args.get("llm_api_key", default=None)
    llm_model = request.args.get("llm_model", default="gpt-4o")
    
    text = home_path_hist(
        homepage_service, 
        report_uri, 
        level,
        use_llm=use_llm,
        include_category=include_category,
        shapes_graph_uri=shapes_uri,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
    )
    return jsonify({
        "level": level, 
        "summary": text,
        "use_llm": use_llm,
        "include_category": include_category
    })


@summary_bp.route("/summaries/home/focus-node", methods=['GET'])
def summarize_home_focus():
    level = _level_param()
    use_llm = _use_llm_param()
    include_category = _include_category_param()
    shapes_uri = request.args.get("shapes_graph_uri", default=None)
    report_uri = request.args.get("validation_report_uri", default = "http://ex.org/ValidationReport")
    
    # Optional LLM configuration
    llm_api_key = request.args.get("llm_api_key", default=None)
    llm_model = request.args.get("llm_model", default="gpt-4o")
    
    text = home_focusnode_hist(
        homepage_service, 
        report_uri, 
        level,
        use_llm=use_llm,
        include_category=include_category,
        shapes_graph_uri=shapes_uri,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
    )
    return jsonify({
        "level": level, 
        "summary": text,
        "use_llm": use_llm,
        "include_category": include_category
    })


@summary_bp.route("/summaries/home/constraint", methods=['GET'])
def summarize_home_constraint():    
    level = _level_param()
    use_llm = _use_llm_param()
    include_category = _include_category_param()
    shapes_uri = request.args.get("shapes_graph_uri", default=None)
    report_uri = request.args.get("validation_report_uri", default = "http://ex.org/ValidationReport")
    
    # Optional LLM configuration
    llm_api_key = request.args.get("llm_api_key", default=None)
    llm_model = request.args.get("llm_model", default="gpt-4o")
    
    text = home_constraint_hist(
        homepage_service, 
        report_uri, 
        level,
        use_llm=use_llm,
        include_category=include_category,
        shapes_graph_uri=shapes_uri,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
    )
    return jsonify({
        "level": level, 
        "summary": text,
        "use_llm": use_llm,
        "include_category": include_category
    })


@summary_bp.route("/summaries/home/path/top", methods=['GET'])
def summarize_home_path_top():
    level = _level_param()
    use_llm = _use_llm_param()
    include_category = _include_category_param()
    shapes_uri = request.args.get("shapes_graph_uri", default=None)
    report_uri = request.args.get("validation_report_uri", default = "http://ex.org/ValidationReport")
    top_k = int(request.args.get("top_k", 3))
    
    # Optional LLM configuration
    llm_api_key = request.args.get("llm_api_key", default=None)
    llm_model = request.args.get("llm_model", default="gpt-4o")
    
    text = home_paths_top(
        homepage_service, 
        report_uri, 
        level, 
        top_k,
        use_llm=use_llm,
        include_category=include_category,
        shapes_graph_uri=shapes_uri,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
    )
    return jsonify({
        "level": level, 
        "summary": text,
        "use_llm": use_llm,
        "include_category": include_category
    })


# ============================================================
#                       SHAPES VIEW
# ============================================================

@summary_bp.route("/summaries/shapes/distribution-constraint", methods=['GET'])
def summarize_shapes_distribution():
    level = _level_param()
    use_llm = _use_llm_param()
    include_category = _include_category_param()
    node_shape = request.args.get("node_shape")
    # if not node_shape:
        # return jsonify({"error": "node_shape is required"}), 400
    shapes_uri = request.args.get("shapes_graph_uri", default=None)
    report_uri = request.args.get("validation_report_uri", default = "http://ex.org/ValidationReport")
    
    # Optional LLM configuration
    llm_api_key = request.args.get("llm_api_key", default=None)
    llm_model = request.args.get("llm_model", default="gpt-4o")
    
    text = shapes_distribution_per_constraint(
        shapes_overview_service, 
        node_shape, 
        report_uri, 
        level,
        use_llm=use_llm,
        include_category=include_category,
        shapes_graph_uri=shapes_uri,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
    )
    return jsonify({
        "level": level, 
        "summary": text,
        "use_llm": use_llm,
        "include_category": include_category
    })


@summary_bp.route("/summaries/shapes/correlation", methods=['GET'])
def summarize_shapes_correlation():
    level = _level_param()
    use_llm = _use_llm_param()
    include_category = _include_category_param()
    shapes_uri = request.args.get("shapes_graph_uri", default=None)
    report_uri = request.args.get("validation_report_uri", default = "http://ex.org/ValidationReport")
    
    # Optional LLM configuration
    llm_api_key = request.args.get("llm_api_key", default=None)
    llm_model = request.args.get("llm_model", default="gpt-4o")
    
    text = shapes_correlation_constraints_vs_violations(
        shapes_overview_service, 
        report_uri, 
        level,
        use_llm=use_llm,
        include_category=include_category,
        shapes_graph_uri=shapes_uri,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
    )
    return jsonify({
        "level": level, 
        "summary": text,
        "use_llm": use_llm,
        "include_category": include_category
    })


@summary_bp.route("/summaries/shapes/diversity-intensity", methods=['GET'])
def summarize_shapes_diversity_intensity_route():
    level = _level_param()
    use_llm = _use_llm_param()
    include_category = _include_category_param()
    shapes_uri = request.args.get("shapes_graph_uri", default=None)
    report_uri = request.args.get("validation_report_uri", default = "http://ex.org/ValidationReport")
    
    # Optional LLM configuration
    llm_api_key = request.args.get("llm_api_key", default=None)
    llm_model = request.args.get("llm_model", default="gpt-4o")
    
    text = shapes_diversity_intensity(
        shapes_overview_service, 
        report_uri, 
        level,
        use_llm=use_llm,
        include_category=include_category,
        shapes_graph_uri=shapes_uri,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
    )
    return jsonify({
        "level": level, 
        "summary": text,
        "use_llm": use_llm,
        "include_category": include_category
    })


# Debug endpoint to test API key visibility
@summary_bp.route("/summaries/debug/api-key", methods=['GET'])
def debug_api_key():
    """Debug endpoint to check if API key is visible to backend"""
    env_key = os.getenv("OPENAI_API_KEY", "")
    param_key = request.args.get("llm_api_key", None)
    
    return jsonify({
        "environment_variable": {
            "set": env_key != "",
            "preview": env_key[:10] + "..." if env_key else None,
            "length": len(env_key) if env_key else 0
        },
        "query_parameter": {
            "provided": param_key is not None,
            "preview": param_key[:10] + "..." if param_key else None,
            "length": len(param_key) if param_key else 0
        },
        "recommendation": "Use query parameter (llm_api_key) if environment variable is not set"
    })


