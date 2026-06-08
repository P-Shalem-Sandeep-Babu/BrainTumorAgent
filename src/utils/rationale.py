"""
Treatment Rationale Table.

Provides human-readable explanations for each (tumor_type, action) pair.
Used by both the Streamlit prediction page and any printed reports.

This is intentionally hand-curated -- it's a UI / explainability aid,
not a learned model output. The DQN agent picks the action; this
module just explains what the action means in clinical terms.
"""

from src.utils.config import Config


# Severity-ordered rationale for each tumor type and the actions it can take
RATIONALE = {
    "notumor": {
        "description": "No tumor detected. Brain tissue appears normal.",
        "actions": {
            0: ("Monitor patient",
                "Routine follow-up is appropriate when no tumor is detected."),
            1: ("Recommend specialist",
                "A specialist consult is safe but not strictly required."),
            2: ("Recommend biopsy",
                "Biopsy is invasive -- not indicated without suspicious findings."),
            3: ("Emergency attention",
                "Emergency resources are not warranted when no tumor is present."),
        },
    },
    "pituitary": {
        "description": "Pituitary tumor detected. Usually benign; hormonal effects are possible.",
        "actions": {
            0: ("Monitor patient",
                "Conservative monitoring may be acceptable for small, asymptomatic tumors."),
            1: ("Recommend specialist",
                "Endocrinology / neurosurgery consult is the standard next step."),
            2: ("Recommend biopsy",
                "Biopsy is rarely first-line for pituitary tumors; imaging follow-up is preferred."),
            3: ("Emergency attention",
                "Pituitary tumors are rarely emergencies unless causing acute vision loss."),
        },
    },
    "meningioma": {
        "description": "Meningioma detected. Usually slow-growing; often benign.",
        "actions": {
            0: ("Monitor patient",
                "Watchful waiting is risky if the tumor is large or symptomatic."),
            1: ("Recommend specialist",
                "Specialist consult is reasonable, but biopsy is more decisive."),
            2: ("Recommend biopsy",
                "Biopsy / surgical sampling is the standard of care for meningioma workup."),
            3: ("Emergency attention",
                "Overcautious -- meningiomas are rarely emergent unless causing herniation."),
        },
    },
    "glioma": {
        "description": "Glioma detected. Aggressive tumor requiring urgent evaluation.",
        "actions": {
            0: ("Monitor patient",
                "Dangerous -- delays treatment for an aggressive tumor."),
            1: ("Recommend specialist",
                "Too slow for an aggressive glioma; biopsy should follow quickly."),
            2: ("Recommend biopsy",
                "Acceptable next step, but emergency pathway is preferred."),
            3: ("Emergency attention",
                "Correct -- gliomas warrant urgent neurosurgical evaluation."),
        },
    },
}


def get_rationale(tumor_type: str, action_index: int) -> dict:
    """
    Get the rationale for recommending a given action for a tumor type.

    Args:
        tumor_type: one of "notumor", "pituitary", "meningioma", "glioma"
        action_index: int 0-3 (Monitor/Specialist/Biopsy/Emergency)

    Returns:
        dict with keys: tumor_type, action_name, severity, description, rationale
    """
    config = Config()
    tumor_type = tumor_type.lower()
    entry = RATIONALE.get(tumor_type, RATIONALE["notumor"])
    action_name, rationale_text = entry["actions"].get(
        action_index, ("Unknown", "No rationale available.")
    )
    return {
        "tumor_type": tumor_type,
        "action_name": action_name,
        "severity": config.TUMOR_SEVERITY.get(tumor_type, 0),
        "description": entry["description"],
        "rationale": rationale_text,
    }


def get_full_table() -> list:
    """
    Build a (tumor_type x action) matrix for the Streamlit display.

    Returns:
        list of dicts, one row per (tumor, action) pair, containing
        tumor, action, severity, and rationale text.
    """
    config = Config()
    rows = []
    for tumor in config.TUMOR_TYPES:
        for action_idx, action_name in enumerate(config.TREATMENT_ACTIONS):
            entry = RATIONALE.get(tumor, RATIONALE["notumor"])
            _, rationale_text = entry["actions"].get(action_idx, ("-", "-"))
            rows.append({
                "Tumor": tumor.capitalize(),
                "Severity": config.TUMOR_SEVERITY.get(tumor, 0),
                "Action": action_name,
                "Rationale": rationale_text,
            })
    return rows
