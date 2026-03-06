# -*- coding: utf-8 -*-
import json
import random
from datetime import datetime, timedelta

def generate_aligned_transactions(customer_file, output_file, n_txns=25000):
    # 1. Đọc file customers.json để lấy thông tin Label từng người
    try:
        with open(customer_file, 'r', encoding='utf-8') as f:
            customers = json.load(f)
        
        # Tạo từ điển để tra cứu nhanh: {ID: Label}
        customer_lookup = {c['Customer ID']: c.get('Label', 'Normal') for c in customers}
        all_ids = list(customer_lookup.keys())
        print(f"Đã load {len(all_ids)} khách hàng. Đang tạo giao dịch khớp với Label...")
    except Exception as e:
        print(f"Lỗi: {e}")
        return

    # 2. Định nghĩa kho nội dung theo từng Label
    content_map = {
        "Normal": {
            "details": ["Supermarket", "Electricity Bill", "Monthly Salary", "Restaurant", "Starbucks", "Gas Station", "Netflix Subscription"],
            "locations": ["Hanoi - VN", "HCMC - VN", "Da Nang - VN", "Can Tho - VN"],
            "amount_range": (20000, 1000000000)
        },
        "Gambling": {
            "details": ["Casino Online Top-up", "Betting Wallet Deposit", "Gaming Chip Purchase", "Virtual Slot Funding", "P2P Game Transfer"],
            "locations": ["Singapore - SG", "Macau - CN", "Manila - PH", "Cambodia - KH"],
            "amount_range": (20000, 1000000000)
        },
        "Loan Sharking": {
            "details": ["Quick Loan Disbursement", "Private Finance Support", "Urgent Cash Out", "P2P Lending Transfer", "Interest Payment Received"],
            "locations": ["Hanoi - VN", "HCMC - VN", "Hai Phong - VN"], # Tín dụng đen thường nội địa
            "amount_range": (20000, 1000000000)
        }
    }

    transactions = []
    devices = ["iPhone 15", "Samsung S23", "MacBook Air", "Web Browser", "Android Phone"]

    # 3. Tạo 25,000 giao dịch khớp 100% với Label của Sender
    for i in range(n_txns):
        sender_id = random.choice(all_ids)
        label = customer_lookup[sender_id] # Lấy nhãn của người này
        
        # Lấy bộ dữ liệu tương ứng với nhãn
        config = content_map.get(label, content_map["Normal"])
        
        detail = random.choice(config["details"])
        amount = random.randint(config["amount_range"][0], config["amount_range"][1])
        loc = random.choice(config["locations"])
        
        # Logic thời gian: Tội phạm thường hoạt động giờ nhạy cảm hơn
        if label != "Normal":
            hour = random.choice([23, 0, 1, 2, 3, 4, 12, 13]) 
        else:
            hour = random.randint(7, 22)

        date = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 364))
        timestamp = date.replace(hour=hour, minute=random.randint(0, 59)).strftime("%Y-%m-%d %H:%M:%S")

        transactions.append({
            "Transaction ID": f"TXN_{300001 + i}",
            "Sender Account ID": sender_id,
            "Receiver Account ID": f"REC_{random.randint(1000, 9999)}",
            "Transaction amount": amount,
            "Timestamp": timestamp,
            "Transaction Detail": detail,
            "Geological": loc,
            "Device Use": random.choice(devices)
        })

    # 4. Lưu kết quả
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transactions, f, indent=4, ensure_ascii=False)
    
    print(f"Hoàn tất! Đã tạo 25,000 giao dịch khớp hoàn toàn với Label của từng Customer.")
    print(f"File lưu tại: {output_file}")

# Chạy code
generate_aligned_transactions('customers.json', 'transaction.json')