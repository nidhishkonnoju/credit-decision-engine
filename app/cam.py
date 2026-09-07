from __future__ import annotations

from typing import Any

FIVE_CATEGORIES = {
    "Character",
    "Capacity",
    "Capital",
    "Collateral",
    "Conditions",
}

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

# Each audited raw model feature is assigned to exactly one traditional 5C category.
FEATURE_TO_5C: dict[str, str] = {
    "time_since_recent_payment": "Character",
    "num_sub": "Character",
    "num_sub_6mts": "Character",
    "num_sub_12mts": "Character",
    "num_dbt": "Character",
    "num_dbt_6mts": "Character",
    "num_dbt_12mts": "Character",
    "num_lss": "Character",
    "num_lss_6mts": "Character",
    "num_lss_12mts": "Character",
    "EDUCATION": "Conditions",
    "AGE": "Conditions",
    "NETMONTHLYINCOME": "Capacity",
    "Time_With_Curr_Empr": "Capacity",
    "pct_of_active_TLs_ever": "Character",
    "pct_opened_TLs_L6m_of_L12m": "Conditions",
    "pct_currentBal_all_TL": "Capital",
    "CC_utilization": "Capital",
    "CC_Flag": "Collateral",
    "PL_utilization": "Capital",
    "PL_Flag": "Collateral",
    "max_unsec_exposure_inPct": "Collateral",
    "HL_Flag": "Collateral",
    "GL_Flag": "Collateral",
    "Total_TL": "Capital",
    "Tot_Closed_TL": "Character",
    "Tot_Active_TL": "Capital",
    "Total_TL_opened_L6M": "Conditions",
    "Tot_TL_closed_L6M": "Character",
    "pct_tl_open_L6M": "Conditions",
    "pct_tl_closed_L6M": "Character",
    "pct_active_tl": "Capital",
    "pct_closed_tl": "Character",
    "Total_TL_opened_L12M": "Conditions",
    "Tot_TL_closed_L12M": "Character",
    "pct_tl_open_L12M": "Conditions",
    "pct_tl_closed_L12M": "Character",
    "Auto_TL": "Collateral",
    "CC_TL": "Collateral",
    "Consumer_TL": "Collateral",
    "Gold_TL": "Collateral",
    "Home_TL": "Collateral",
    "PL_TL": "Collateral",
    "Secured_TL": "Collateral",
    "Unsecured_TL": "Collateral",
    "Other_TL": "Collateral",
    "Age_Oldest_TL": "Character",
    "Age_Newest_TL": "Character",
}

FEATURE_TO_5C_RATIONALE: dict[str, str] = {
    "time_since_recent_payment": "A recent payment indicates current repayment behavior and supports Character assessment.",
    "num_sub": "The number of substandard accounts records adverse repayment history and informs Character.",
    "num_sub_6mts": "Recent substandard accounts indicate current repayment difficulty and inform Character.",
    "num_sub_12mts": "Substandard accounts over the last year show the persistence of repayment problems and inform Character.",
    "num_dbt": "The number of doubtful accounts reflects serious past repayment deterioration and informs Character.",
    "num_dbt_6mts": "Doubtful accounts in the last six months indicate recent repayment stress and inform Character.",
    "num_dbt_12mts": "Doubtful accounts over the last year indicate sustained repayment stress and inform Character.",
    "num_lss": "Loss-classified accounts reflect severe adverse credit history and inform Character.",
    "num_lss_6mts": "Recent loss-classified accounts indicate severe current repayment problems and inform Character.",
    "num_lss_12mts": "Loss-classified accounts over the last year indicate persistent severe repayment problems and inform Character.",
    "EDUCATION": "Education is treated as a broad applicant-context variable rather than a direct repayment measure, so it is assigned to Conditions.",
    "AGE": "Age provides applicant context that can affect the interpretation of credit history and is assigned to Conditions.",
    "NETMONTHLYINCOME": "Monthly income is a direct indicator of repayment capacity and available cash flow.",
    "Time_With_Curr_Empr": "Employment tenure is a proxy for income stability and therefore supports Capacity assessment.",
    "pct_of_active_TLs_ever": "The share of historically active credit lines describes the applicant's established credit behavior and informs Character.",
    "pct_opened_TLs_L6m_of_L12m": "The recent share of newly opened lines captures changes in credit-seeking behavior and is assigned to Conditions.",
    "pct_currentBal_all_TL": "The share of balances currently outstanding indicates how much existing credit is being used and informs Capital.",
    "CC_utilization": "Credit-card utilization measures revolving-credit usage and informs Capital.",
    "CC_Flag": "The presence of credit-card exposure identifies an existing collateral-free obligation and is assigned to Collateral.",
    "PL_utilization": "Personal-loan utilization measures usage of an existing installment facility and informs Capital.",
    "PL_Flag": "The presence of a personal loan identifies an existing secured or unsecured obligation and is assigned to Collateral.",
    "max_unsec_exposure_inPct": "The share of unsecured exposure describes the composition of obligations without pledged security and informs Collateral.",
    "HL_Flag": "The presence of a home loan indicates housing-related secured exposure and informs Collateral.",
    "GL_Flag": "The presence of a gold loan indicates asset-backed borrowing and informs Collateral.",
    "Total_TL": "Total credit limit measures the scale of available borrowing capacity and informs Capital.",
    "Tot_Closed_TL": "The number of closed accounts records completed or discontinued credit relationships and informs Character.",
    "Tot_Active_TL": "The number of active accounts describes current obligations and informs Capital.",
    "Total_TL_opened_L6M": "Credit lines opened in the last six months capture recent changes in borrowing conditions.",
    "Tot_TL_closed_L6M": "Credit lines closed recently provide evidence about recent account behavior and inform Character.",
    "pct_tl_open_L6M": "The recent opening rate measures changes in borrowing activity and informs Conditions.",
    "pct_tl_closed_L6M": "The recent closing rate describes changes in the credit portfolio and informs Character.",
    "pct_active_tl": "The proportion of active lines describes the current structure of the applicant's credit portfolio and informs Capital.",
    "pct_closed_tl": "The proportion of closed lines describes completed credit relationships and informs Character.",
    "Total_TL_opened_L12M": "Credit lines opened over the last year capture recent borrowing activity and inform Conditions.",
    "Tot_TL_closed_L12M": "Credit lines closed over the last year provide a longer recent history of account changes and inform Character.",
    "pct_tl_open_L12M": "The one-year opening rate captures the pace of new borrowing and informs Conditions.",
    "pct_tl_closed_L12M": "The one-year closing rate captures the pace of completed or discontinued credit relationships and informs Character.",
    "Auto_TL": "The presence of an auto loan identifies vehicle-backed or vehicle-related exposure and informs Collateral.",
    "CC_TL": "The number of credit-card lines identifies revolving exposure without pledged collateral and informs Collateral.",
    "Consumer_TL": "Consumer-loan lines identify existing consumer obligations and inform Collateral.",
    "Gold_TL": "Gold-loan lines identify asset-backed obligations and inform Collateral.",
    "Home_TL": "Home-loan lines identify housing-related secured obligations and inform Collateral.",
    "PL_TL": "Personal-loan lines identify existing installment obligations and inform Collateral.",
    "Secured_TL": "Secured-loan lines identify obligations backed by pledged assets and inform Collateral.",
    "Unsecured_TL": "Unsecured-loan lines identify obligations without pledged assets and inform Collateral.",
    "Other_TL": "Other credit lines capture obligations outside the named product groups and are assigned to Collateral as an exposure inventory.",
    "Age_Oldest_TL": "The age of the oldest credit line measures the length of established credit history and informs Character.",
    "Age_Newest_TL": "The age of the newest credit line captures recent credit activity and informs Character.",
}


def validate_feature_mapping(feature_columns: list[str]) -> None:
    """Require every raw model feature to have exactly one valid 5C category."""
    missing = sorted(set(feature_columns) - set(FEATURE_TO_5C))
    extra = sorted(set(FEATURE_TO_5C) - set(feature_columns))
    invalid = sorted(
        feature for feature, category in FEATURE_TO_5C.items() if category not in FIVE_CATEGORIES
    )
    if missing or extra or invalid:
        raise ValueError(
            f"Invalid 5C feature mapping: missing={missing}, extra={extra}, invalid={invalid}"
        )


def feature_category(feature_name: str) -> str:
    """Resolve an encoded preprocessor feature to its raw feature's 5C category."""
    normalized = _normalize_feature_key(feature_name)
    if normalized not in FEATURE_TO_5C:
        raise KeyError(f"No 5C category is defined for feature: {feature_name}")
    return FEATURE_TO_5C[normalized]


def generate_cam(
    applicant: Any,
    model_result: dict[str, Any],
    stability_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a structured 5C CAM from stability-filtered local SHAP reasons.

    Lin and Wang (2025) motivate restricting customer-facing explanations to
    features whose rankings remain stable across model seeds. Decision thresholds
    remain owned by the model summary; this function only organizes explanations.
    """
    from app.modeling import summarize_decision

    stability = stability_result or model_result.get("stability_result", {})
    stable_features = stability.get(
        "stable_features",
        model_result.get("stable_feature_names", []),
    )
    summary = summarize_decision(
        model_result["model"],
        model_result["preprocessor"],
        applicant,
        threshold=model_result["threshold"],
        review_threshold=model_result.get("review_threshold"),
        stable_feature_names=stable_features,
    )
    appraisal = generate_credit_appraisal_memo(summary)
    reasons_by_category = {category: [] for category in sorted(FIVE_CATEGORIES)}
    applicant_reasons = appraisal["applicant_facing"]["reasons"]
    for reason, sentence in zip(summary["top_reasons"], applicant_reasons):
        category = feature_category(reason["feature"])
        reasons_by_category[category].append(
            {
                **reason,
                "text": sentence,
            }
        )

    sections = {
        category: {
            "reasons": reasons,
            "empty_message": (
                f"No high-confidence factors identified for {category}."
                if not reasons
                else None
            ),
        }
        for category, reasons in reasons_by_category.items()
    }
    return {
        "decision": summary["decision"],
        "probability": summary["probability"],
        "threshold": summary["threshold"],
        "review_threshold": summary["review_threshold"],
        "stability_filtered": summary["stability_filtered"],
        "stable_reason_features": [reason["feature"] for reason in summary["top_reasons"]],
        "summary": appraisal["applicant_facing"]["summary"],
        "sections": sections,
        "technical_summary": summary,
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
        "stability_filtered": summary.get("stability_filtered", False),
        "summary": applicant_summary,
        "reasons": applicant_reasons,
        "fallback_features": fallback_features,
        "no_high_confidence_factors": (
            "No high-confidence factors identified from the stability-filtered explanation."
            if summary.get("stability_filtered") and not applicant_reasons
            else None
        ),
    }

    return {
        **summary,
        "internal": internal,
        "applicant_facing": applicant_facing,
    }
