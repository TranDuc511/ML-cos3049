# stats.py — Analytics endpoints for the dashboard
from fastapi import APIRouter
from services.storage import transactions

router = APIRouter(prefix="/api/stats", tags=["Stats"])


@router.get("/summary")
def get_summary():
    total = len(transactions)
    frauds = sum(1 for tx in transactions if tx["is_fraud"] == 1)
    rate = round(frauds / total * 100, 2) if total > 0 else 0
    safe = total - frauds
    return {"status": "success", "data": {
        "total_transactions": total,
        "fraud_count": frauds,
        "safe_count": safe,
        "fraud_rate": rate,
    }}


@router.get("/by_hour")
def get_by_hour():
    """All transactions + fraud breakdown by hour (0-23)."""
    # Pre-fill all 24 hours so chart is always complete
    hourly = {str(h).zfill(2): {"total": 0, "fraud": 0} for h in range(24)}
    for tx in transactions:
        if not tx.get("timestamp"):
            continue
        try:
            hour = tx["timestamp"].split()[1].split(":")[0].zfill(2)
            if hour in hourly:
                hourly[hour]["total"] += 1
                if tx["is_fraud"] == 1:
                    hourly[hour]["fraud"] += 1
        except (IndexError, AttributeError):
            continue
    data = [{"hour": h, **v} for h, v in sorted(hourly.items())]
    return {"status": "success", "data": data}


@router.get("/history_trend")
def get_history_trend():
    stats = {}
    for tx in transactions:
        if not tx.get("timestamp"):
            continue
        try:
            day = tx["timestamp"].split()[0]
            stats[day] = stats.get(day, 0) + 1
        except (IndexError, AttributeError):
            continue
    data = [{"day": d, "count": c} for d, c in sorted(stats.items())]
    return {"status": "success", "data": data}


@router.get("/amount_distribution")
def get_amount_distribution():
    ranges = {
        "0-50K":    {"clean": 0, "fraud": 0},
        "50K-200K": {"clean": 0, "fraud": 0},
        "200K-1M":  {"clean": 0, "fraud": 0},
        "1M-5M":    {"clean": 0, "fraud": 0},
        "5M-10M":   {"clean": 0, "fraud": 0},
        "10M+":     {"clean": 0, "fraud": 0},
    }
    for tx in transactions:
        amt = tx.get("amount", 0)
        is_fraud = tx.get("is_fraud", 0)
        key = "clean" if is_fraud == 0 else "fraud"

        if amt < 50_000:           ranges["0-50K"][key]    += 1
        elif amt < 200_000:        ranges["50K-200K"][key] += 1
        elif amt < 1_000_000:      ranges["200K-1M"][key]  += 1
        elif amt < 5_000_000:      ranges["1M-5M"][key]    += 1
        elif amt < 10_000_000:     ranges["5M-10M"][key]   += 1
        else:                      ranges["10M+"][key]     += 1
        
    data = [{"range": k, "clean": v["clean"], "fraud": v["fraud"]} for k, v in ranges.items()]
    return {"status": "success", "data": data}


# Keep old fraud_by_hour route for backward compatibility
@router.get("/fraud_by_hour")
def get_fraud_by_hour():
    stats = {}
    for tx in transactions:
        if tx.get("is_fraud") == 1 and tx.get("timestamp"):
            try:
                hour = tx["timestamp"].split()[1].split(":")[0]
                stats[hour] = stats.get(hour, 0) + 1
            except (IndexError, AttributeError):
                continue
    data = [{"hour": h, "count": c} for h, c in sorted(stats.items())]
    return {"status": "success", "data": data}
