from fastapi import APIRouter

from services.storage import transactions

router = APIRouter(prefix="/api", tags=["History"])

@router.get("/history")
def get_history():
    # Return last 100 transactions from memory
    return {"status": "success", "data": transactions[-100:][::-1]}

@router.delete("/history")
def delete_history():
    transactions.clear()
    return {"status": "success", "data": "History cleared."}
