def risk_bucket(prob):
    """
    Convert fraud probability into risk category
    """
    if prob < 0.2:
        return "LOW"
    elif prob < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"


def decision_from_risk(risk):
    """
    Map risk level to system decision
    """
    if risk == "LOW":
        return "ALLOW"
    elif risk == "MEDIUM":
        return "REQUIRE_VERIFICATION"
    else:
        return "FLAG_FOR_REVIEW"
