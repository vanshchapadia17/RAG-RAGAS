from flask import Blueprint, render_template, jsonify
from src.experiment_tracker import get_all_experiments, get_best_run

experiments_bp = Blueprint("experiments", __name__)


@experiments_bp.route("/experiments")
def experiments_page():
    return render_template("experiments.html")


@experiments_bp.route("/api/experiments")
def get_experiments():
    df = get_all_experiments()
    if df.empty:
        return jsonify([])
    if "start_time" in df.columns:
        df["start_time"] = df["start_time"].astype(str)
    return jsonify(df.fillna("").to_dict(orient="records"))


@experiments_bp.route("/api/best-run")
def best_run():
    return jsonify(get_best_run())