"""
FP-Growth + luật kết hợp: triệu chứng → chẩn đoán; gợi ý mức độ & điều trị từ cùng mô hình giao dịch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

PREFIX_S = "S__"
PREFIX_D = "D__"
PREFIX_V = "V__"
PREFIX_T = "T__"


def _default_csv_path() -> Path:
    return Path(__file__).resolve().parents[1] / "Kaggle_DataSet" / "disease_diagnosis.csv"


def load_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or _default_csv_path()
    df = pd.read_csv(path)
    required = [
        "Symptom_1",
        "Symptom_2",
        "Symptom_3",
        "Diagnosis",
        "Severity",
        "Treatment_Plan",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Thiếu cột {c} trong {path}")
    return df


def transactions_from_df(df: pd.DataFrame) -> Tuple[List[List[str]], Dict[str, List[str]]]:
    """Mỗi dòng → giao dịch có tiền tố để tránh trùng ý nghĩa giữa các loại mục."""
    sym_cols = ["Symptom_1", "Symptom_2", "Symptom_3"]
    txs: List[List[str]] = []
    for _, row in df.iterrows():
        items: List[str] = []
        for c in sym_cols:
            items.append(PREFIX_S + str(row[c]).strip())
        items.append(PREFIX_D + str(row["Diagnosis"]).strip())
        items.append(PREFIX_V + str(row["Severity"]).strip())
        items.append(PREFIX_T + str(row["Treatment_Plan"]).strip())
        seen = set()
        uniq: List[str] = []
        for x in items:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        txs.append(uniq)

    symptoms = sorted({x[len(PREFIX_S) :] for t in txs for x in t if x.startswith(PREFIX_S)})
    diagnoses = sorted({x[len(PREFIX_D) :] for t in txs for x in t if x.startswith(PREFIX_D)})
    severities = sorted({x[len(PREFIX_V) :] for t in txs for x in t if x.startswith(PREFIX_V)})
    treatments = sorted({x[len(PREFIX_T) :] for t in txs for x in t if x.startswith(PREFIX_T)})
    meta = {
        "symptoms": symptoms,
        "diagnoses": diagnoses,
        "severities": severities,
        "treatments": treatments,
    }
    return txs, meta


def mine_rules(
    transactions: Sequence[Sequence[str]],
    min_support: float,
    min_confidence: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    te = TransactionEncoder()
    te_arr = te.fit(transactions).transform(transactions)
    ohe = pd.DataFrame(te_arr, columns=te.columns_)
    itemsets = fpgrowth(ohe, min_support=min_support, use_colnames=True)
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(["confidence", "lift"], ascending=[False, False])
    return itemsets, rules


def _is_single_prefixed(fs: frozenset, prefix: str) -> bool:
    if len(fs) != 1:
        return False
    (x,) = tuple(fs)
    return isinstance(x, str) and x.startswith(prefix)


def filter_rules_consequent(rules: pd.DataFrame, prefix: str) -> pd.DataFrame:
    mask = rules["consequents"].apply(lambda c: _is_single_prefixed(c, prefix))
    return rules.loc[mask].copy()


def user_symptom_items(user_symptoms: Iterable[str]) -> frozenset:
    return frozenset(PREFIX_S + s.strip() for s in user_symptoms if s and str(s).strip())


def predict_diagnoses(
    user_symptoms: Iterable[str],
    rules_dx: pd.DataFrame,
) -> List[Tuple[str, float, float]]:
    user = user_symptom_items(user_symptoms)
    best: Dict[str, Tuple[float, float]] = {}
    for _, row in rules_dx.iterrows():
        ant = row["antecedents"]
        cons = row["consequents"]
        if len(cons) != 1:
            continue
        raw = next(iter(cons))
        if not str(raw).startswith(PREFIX_D):
            continue
        disease = str(raw)[len(PREFIX_D) :]
        if not ant.issubset(user):
            continue
        conf = float(row["confidence"])
        lift = float(row["lift"])
        if disease not in best or conf > best[disease][0]:
            best[disease] = (conf, lift)
    ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))
    return [(d, c, l) for d, (c, l) in ranked]


def predict_for_prefix(
    user: frozenset,
    rules: pd.DataFrame,
    prefix: str,
) -> List[Tuple[str, float, float]]:
    best: Dict[str, Tuple[float, float]] = {}
    for _, row in rules.iterrows():
        ant = row["antecedents"]
        cons = row["consequents"]
        if len(cons) != 1:
            continue
        raw = next(iter(cons))
        if not str(raw).startswith(prefix):
            continue
        label = str(raw)[len(prefix) :]
        if not ant.issubset(user):
            continue
        conf = float(row["confidence"])
        lift = float(row["lift"])
        if label not in best or conf > best[label][0]:
            best[label] = (conf, lift)
    ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))
    return [(d, c, l) for d, (c, l) in ranked]


def enrich_user_items(
    user_symptoms: Iterable[str],
    diagnosis: str | None = None,
) -> frozenset:
    u = set(user_symptom_items(user_symptoms))
    if diagnosis:
        u.add(PREFIX_D + diagnosis.strip())
    return frozenset(u)
