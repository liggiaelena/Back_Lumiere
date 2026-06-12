def get_medical_alert():
    return {
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