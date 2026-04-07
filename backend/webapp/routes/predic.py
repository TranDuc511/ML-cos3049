from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.prediction_service import process_and_predict

router = APIRouter(prefix="/api", tags=["Predict"])

class CustomerData(BaseModel):
    # Match the keys generated in JSON or expected by the model
    # Use Field(alias=...) so the frontend can send exactly what is in DB
    customer_id: str = Field(alias="Customer ID")
    date_of_birth: str = Field(alias="Date of Birth", default="2000-01-01")
    gender: str = Field(alias="Gender", default="M")
    location: str = Field(alias="Location", default="Unknown")
    working_status: str = Field(alias="Working Status", default="Unknown")
    salary: float = Field(alias="Salary (per month)", default=0.0)

class TransactionData(BaseModel):
    transaction_id: str = Field(alias="Transaction ID", default="T000")
    timestamp: str = Field(alias="Timestamp")
    sender_id: str = Field(alias="Sender Account ID")
    receiver_id: str = Field(alias="Receiver Account ID", default="R000")
    amount: float = Field(alias="Transaction amount")
    detail: str = Field(alias="Transaction Detail", default="Payment")
    geological: str = Field(alias="Geological", default="Unknown")
    device_use: str = Field(alias="Device Use", default="Mobile")
    account_balance: float = Field(alias="Account balance", default=0.0)

class PredictRequest(BaseModel):
    # Pydantic validates incoming raw JSON. 
    # Example format: {"customer": {"Customer ID": "C123", ...}, "transaction": {"Transaction ID": "T123", ...}}
    customer: CustomerData
    transaction: TransactionData

@router.post("/predicts")
def predict_fraud(request: PredictRequest):
    # 1. Convert validated Pydantic models back to standard dicts matching the training keys
    # using by_alias=True keeps the original keys ("Customer ID", "Transaction ID", etc.)
    customer_dict = request.customer.dict(by_alias=True)
    transaction_dict = request.transaction.dict(by_alias=True)

    # 2. Process & Predict
    result = process_and_predict(customer_dict, transaction_dict)

    from services.storage import transactions
    transactions.append({
        "transaction_id": transaction_dict.get("Transaction ID"),
        "timestamp": transaction_dict.get("Timestamp"),
        "amount": float(transaction_dict.get("Transaction amount", 0)),
        "device_use": transaction_dict.get("Device Use"),
        "is_fraud": int(result["is_fraud"]),
        "votes": result["votes"]
    })

    # 3. Return JSON response
    return {
        "status": "success",
        "data": result
    }

