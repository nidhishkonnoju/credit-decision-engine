from __future__ import annotations

from typing import Any

FEATURE_LABEL_MAP: dict[str, str] = {
    "Age_Oldest_TL": "age of the oldest credit account",
    "Age_Newest_TL": "age of the newest credit account",
    "Total_TL": "total credit limit across your accounts",
    "Consumer_TL": "consumer credit limit",
    "CC_TL": "credit card credit limit",
    "PL_TL": "personal loan balance",
    "Secured_TL": "secured credit limit",
    "Unsecured_TL": "unsecured credit limit",
    "Other_TL": "other credit limit",
    "pct_currentBal_all_TL": "percentage of your total balances currently in use",
    "max_delinquency_level": "maximum delinquency level on your accounts",
    "time_since_recent_payment": "time since your most recent payment",
    "time_since_recent_deliquency": "time since your most recent delinquency",
    "num_times_delinquent": "number of delinquency episodes",
    "NETMONTHLYINCOME": "monthly income",
    "Time_With_Curr_Empr": "time with your current employer",
    "pct_tl_open_L12M": "share of credit lines opened in the last 12 months",
    "max_unsec_exposure_inPct": "share of your exposure that is unsecured",
    "EDUCATION": "education level",
    "Total_TL_opened_L12M": "total credit lines opened in the last 12 months",
    "Total_TL_opened_L6M": "total credit lines opened in the last 6 months",
    "pct_of_active_TLs_ever": "share of active credit lines you have ever held",
    "pct_active_tl": "share of all credit lines that are currently active",
}


def _normalize_feature_key(feature_name: str) -> str:
    """Strip encoding prefixes and standardize to the original feature name."""
    clean = feature_name.split("__")[-1]
    clean = clean.replace("numeric__", "")
    clean = clean.replace("categorical__", "")
    return clean


def _humanize_feature_name(feature_name: str) -> str:
    """Fallback human-readable naming for unmapped features."""
    normalized = _normalize_feature_key(feature_name)
    text = normalized.replace("_", " ").strip()
    text = text.replace("pct ", "percentage of ")
    text = text.replace("num ", "number of ")
    text = text.replace("TL", "credit limit")
    text = text.replace("L12M", "in the last 12 months")
    text = text.replace("L6M", "in the last 6 months")
    return text.strip()


def _feature_label(feature_name: str) -> tuple[str, bool]:
    """Return an applicant-safe label and whether it came from the explicit mapping."""
    normalized = _normalize_feature_key(feature_name)
    label = FEATURE_LABEL_MAP.get(normalized)
    if label is not None:
        return label, True
    fallback = _humanize_feature_name(normalized)
    return fallback, False


def generate_credit_appraisal_memo(summary: dict[str, Any]) -> dict[str, Any]:
    """Create both technical and applicant-facing CAM outputs from the same SHAP summary."""
    fallback_features: list[str] = []
    applicant_reasons: list[str] = []

    for reason in summary.get("top_reasons", []):
        label, mapped = _feature_label(reason["feature"])
        if not mapped:
            fallback_features.append(reason["feature"])

        direction = reason.get("direction", "increases rejection risk")
        if direction == "increases rejection risk":
            sentence = f"Your {label} increases your risk of rejection."
        else:
            sentence = f"Your {label} reduces your risk of rejection."
        applicant_reasons.append(sentence)

    decision = summary.get("decision", "APPROVE")
    probability = summary.get("probability", 0.0)
    threshold = summary.get("threshold", 0.5)
    review_threshold = summary.get("review_threshold")

    if decision == "REVIEW" and review_threshold is not None:
        applicant_summary = (
            f"Based on the information provided, this application is {decision}. "
            f"The estimated rejection risk is {probability:.4f}, which sits between the approval cutoff "
            f"of {threshold:.2f} and the reject cutoff of {review_threshold:.2f}."
        )
    else:
        applicant_summary = (
            f"Based on the information provided, this application is {decision}. "
            f"The estimated rejection risk is {probability:.4f}, which is "
            f"{'above' if probability >= threshold else 'below'} the decision threshold of {threshold:.2f}."
        )

    internal = {
        "decision": decision,
        "probability": probability,
        "threshold": threshold,
        "review_threshold": review_threshold,
        "reasons": [
            {
                "feature": reason["feature"],
                "contribution": reason.get("contribution"),
                "absolute_contribution": reason.get("absolute_contribution"),
                "direction": reason.get("direction"),
            }
            for reason in summary.get("top_reasons", [])
        ],
    }

    applicant_facing = {
        "decision": decision,
        "probability": probability,
        "threshold": threshold,
        "review_threshold": review_threshold,
        "summary": applicant_summary,
        "reasons": applicant_reasons,
        "fallback_features": fallback_features,
    }

    return {
        **summary,
        "internal": internal,
        "applicant_facing": applicant_facing,
    }
