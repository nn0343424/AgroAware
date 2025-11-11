# ml_service/fertilizer.py
from fertilizer_data import CROP_NPK_REQUIREMENTS

def get_fertilizer_recommendation(crop, N, P, K):
    crop = crop.lower()
    if crop not in CROP_NPK_REQUIREMENTS:
        return {
            "crop": crop,
            "message": "No specific fertilizer dataset for this crop. Use balanced NPK (10:26:26) or local soil advisory."
        }

    ideal = CROP_NPK_REQUIREMENTS[crop]
    result = {"crop": crop, "nutrients": {}, "recommendations": []}

    def check(value, r):
        low, high = r
        if value < low: return "low"
        if value > high: return "high"
        return "optimal"

    statusN = check(N, ideal["N"])
    statusP = check(P, ideal["P"])
    statusK = check(K, ideal["K"])

    result["nutrients"] = {
        "N": {"value": N, "ideal_range": ideal["N"], "status": statusN},
        "P": {"value": P, "ideal_range": ideal["P"], "status": statusP},
        "K": {"value": K, "ideal_range": ideal["K"], "status": statusK},
    }

    if statusN == "low":
        result["recommendations"].append("Apply Urea (46% N): 40–60 kg/acre")
    if statusP == "low":
        result["recommendations"].append("Apply DAP (18-46-0): 20–40 kg/acre")
    if statusK == "low":
        result["recommendations"].append("Apply Muriate of Potash (MOP, 60% K2O): 20–35 kg/acre")

    if not result["recommendations"]:
        result["recommendations"].append("Soil nutrients appear balanced — no major fertilizer adjustment needed.")

    return result
