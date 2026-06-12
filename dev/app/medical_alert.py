TRIAGE_THRESHOLD = 0.85

SERIOUS_CONDITION_KEYS = {
    "melanoma_suspected",
    "suspicious_lesion",
    "urgent_skin_concern",
}


def get_default_medical_alert():
    return {
        "alert": False,
        "severity": "info",
        "title": "Not a medical diagnosis",
        "message": (
            "Lumière provides cosmetic guidance and educational prototype analysis only. "
            "It does not diagnose, treat, or replace advice from a licensed healthcare professional."
        ),
        "recommendation": (
            "If you notice persistent, painful, changing, spreading, bleeding, or concerning skin changes, "
            "please consult a licensed healthcare professional."
        ),
    }


def get_high_risk_medical_alert(triggered_condition, confidence):
    return {
        "alert": True,
        "severity": "warning",
        "title": "Medical review recommended",
        "message": (
            "The analysis detected a skin concern pattern that may require medical attention. "
            "Lumière cannot diagnose this condition."
        ),
        "recommendation": (
            "Please consult a licensed healthcare professional or dermatologist before using makeup "
            "over this area."
        ),
        "triggered_condition": triggered_condition,
        "confidence": confidence,
    }


def _extract_confidence(condition_value):
    if isinstance(condition_value, dict):
        return float(condition_value.get("confidence", 0) or 0)

    if isinstance(condition_value, (int, float)):
        return float(condition_value)

    return 0.0


def find_high_risk_condition(condition_map):
    if not isinstance(condition_map, dict):
        return None, 0.0

    for condition_name in SERIOUS_CONDITION_KEYS:
        confidence = _extract_confidence(condition_map.get(condition_name))

        if confidence > TRIAGE_THRESHOLD:
            return condition_name, confidence

    return None, 0.0


def build_condition_map_from_regions(regions):
    """
    Collects condition confidence values from region-level analysis results.

    Expected optional shape from any region:
    {
        "condition_map": {
            "melanoma_suspected": {"confidence": 0.1}
        }
    }

    If multiple regions provide the same condition, the highest confidence is kept.
    """
    condition_map = {}

    if not isinstance(regions, dict):
        return condition_map

    for region_data in regions.values():
        if not isinstance(region_data, dict):
            continue

        region_condition_map = region_data.get("condition_map", {})

        if not isinstance(region_condition_map, dict):
            continue

        for condition_name, condition_value in region_condition_map.items():
            confidence = _extract_confidence(condition_value)

            current = condition_map.get(condition_name, {"confidence": 0.0})
            current_confidence = _extract_confidence(current)

            if confidence > current_confidence:
                condition_map[condition_name] = {
                    "confidence": confidence,
                    "source": "region_analysis",
                }

    return condition_map


def apply_medical_triage(response):
    """
    Adds the medical alert layer to the final response.

    If a serious condition is detected with confidence above the threshold,
    makeup recommendations are blocked.
    """
    if not isinstance(response, dict):
        return response

    condition_map = response.get("condition_map", {})
    triggered_condition, confidence = find_high_risk_condition(condition_map)

    if triggered_condition:
        response["medical_alert"] = get_high_risk_medical_alert(
            triggered_condition=triggered_condition,
            confidence=confidence,
        )
        response["recommendations_blocked"] = True
        response["recommendations"] = []
        response["makeup_recommendations"] = None
        return response

    response["medical_alert"] = get_default_medical_alert()
    response["recommendations_blocked"] = False

    return response


# Backward-compatible name for older imports
def get_medical_alert():
    return get_default_medical_alert()