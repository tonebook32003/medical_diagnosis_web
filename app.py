"""
Ứng dụng web hỗ trợ chẩn đoán & điều trị (FP-Growth + luật kết hợp).
Chạy: flask --app app run  hoặc  python app.py
"""
from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from engine import (
    PREFIX_D,
    PREFIX_T,
    PREFIX_V,
    enrich_user_items,
    filter_rules_consequent,
    load_dataset,
    mine_rules,
    predict_diagnoses,
    predict_for_prefix,
    transactions_from_df,
)

app = Flask(__name__)
app.secret_key = os.urandom(24)

_STATE: dict = {
    "ready": False,
    "rules_dx": None,
    "rules_sev": None,
    "rules_tx": None,
    "meta": None,
    "n_transactions": 0,
    "n_itemsets": 0,
    "n_rules_dx": 0,
    "n_rules_sev": 0,
    "n_rules_tx": 0,
    "csv_path": None,
    "min_support": None,
    "min_confidence": None,
    "error": None,
}


def _default_csv() -> Path:
    return Path(__file__).resolve().parents[1] / "Kaggle_DataSet" / "disease_diagnosis.csv"


@app.route("/")
def index():
    default_csv = str(_default_csv())
    symptoms: list[str] = []
    try:
        df = load_dataset(_default_csv())
        _, meta = transactions_from_df(df)
        symptoms = meta["symptoms"]
    except Exception:
        pass
    return render_template(
        "index.html",
        default_csv=default_csv,
        symptoms=symptoms,
        state=_STATE,
    )


@app.post("/api/train")
def api_train():
    global _STATE
    data = request.get_json(silent=True) or {}
    csv_path = Path(data.get("csv_path") or _default_csv())
    try:
        min_sup = float(data.get("min_support", 0.02))
        min_conf = float(data.get("min_confidence", 0.2))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Tham số support/confidence không hợp lệ."}), 400

    _STATE["error"] = None
    try:
        df = load_dataset(csv_path)
        txs, meta = transactions_from_df(df)
        itemsets, rules = mine_rules(txs, min_support=min_sup, min_confidence=min_conf)

        rules_dx = filter_rules_consequent(rules, PREFIX_D)
        rules_sev = filter_rules_consequent(rules, PREFIX_V)
        rules_tx = filter_rules_consequent(rules, PREFIX_T)

        _STATE.update(
            ready=True,
            rules_dx=rules_dx,
            rules_sev=rules_sev,
            rules_tx=rules_tx,
            meta=meta,
            n_transactions=len(txs),
            n_itemsets=len(itemsets),
            n_rules_dx=len(rules_dx),
            n_rules_sev=len(rules_sev),
            n_rules_tx=len(rules_tx),
            csv_path=str(csv_path.resolve()),
            min_support=min_sup,
            min_confidence=min_conf,
            error=None,
        )
    except Exception as e:
        _STATE["ready"] = False
        _STATE["error"] = str(e)
        return jsonify({"ok": False, "error": str(e)}), 400

    return jsonify(
        {
            "ok": True,
            "n_transactions": _STATE["n_transactions"],
            "n_itemsets": _STATE["n_itemsets"],
            "n_rules_dx": _STATE["n_rules_dx"],
            "n_rules_sev": _STATE["n_rules_sev"],
            "n_rules_tx": _STATE["n_rules_tx"],
            "csv_path": _STATE["csv_path"],
        }
    )


@app.post("/api/predict")
def api_predict():
    if not _STATE.get("ready"):
        return jsonify({"ok": False, "error": "Chưa huấn luyện mô hình."}), 400

    data = request.get_json(silent=True) or {}
    symptoms = data.get("symptoms") or []
    if not isinstance(symptoms, list) or not symptoms:
        return jsonify({"ok": False, "error": "Chọn ít nhất một triệu chứng."}), 400

    rules_dx = _STATE["rules_dx"]
    rules_sev = _STATE["rules_sev"]
    rules_tx = _STATE["rules_tx"]
    assert rules_dx is not None and rules_sev is not None and rules_tx is not None

    ranked_dx = predict_diagnoses(symptoms, rules_dx)
    top_dx = ranked_dx[0][0] if ranked_dx else None

    u_sym = enrich_user_items(symptoms, None)
    u_full = enrich_user_items(symptoms, top_dx) if top_dx else u_sym

    sev_from_sym = predict_for_prefix(u_sym, rules_sev, PREFIX_V)
    tx_from_sym = predict_for_prefix(u_sym, rules_tx, PREFIX_T)
    sev_enriched = predict_for_prefix(u_full, rules_sev, PREFIX_V)
    tx_enriched = predict_for_prefix(u_full, rules_tx, PREFIX_T)

    def merge_rankings(
        a: list[tuple[str, float, float]],
        b: list[tuple[str, float, float]],
    ) -> list[tuple[str, float, float]]:
        merged: dict[str, tuple[float, float]] = {}
        for label, c, l in a + b:
            if label not in merged or c > merged[label][0]:
                merged[label] = (c, l)
        return sorted([(k, v[0], v[1]) for k, v in merged.items()], key=lambda t: (-t[1], -t[2]))

    ranked_sev = merge_rankings(sev_from_sym, sev_enriched)
    ranked_tx = merge_rankings(tx_from_sym, tx_enriched)

    return jsonify(
        {
            "ok": True,
            "diagnoses": [{"label": d, "confidence": c, "lift": l} for d, c, l in ranked_dx],
            "severities": [{"label": s, "confidence": c, "lift": l} for s, c, l in ranked_sev[:5]],
            "treatments": [{"label": t, "confidence": c, "lift": l} for t, c, l in ranked_tx[:5]],
            "top_diagnosis": top_dx,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
