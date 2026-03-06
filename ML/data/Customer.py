# -*- coding: utf-8 -*-
import json
import random

def generate_unlabeled_customers(transaction_file, output_customer_file):
    try:
        # 1. Đọc file transaction để lấy danh sách ID và đếm Transaction Count thực tế
        with open(transaction_file, 'r', encoding='utf-8') as f:
            transactions = json.load(f)
        
        # Đếm số lượng giao dịch của từng người
        counts = {}
        for txn in transactions:
            s_id = txn["Sender Account ID"]
            counts[s_id] = counts.get(s_id, 0) + 1
        
        # Lấy danh sách ID duy nhất từ file giao dịch
        unique_ids = list(counts.keys())
        
        # 2. Tạo thêm ID nếu danh sách ID từ giao dịch chưa đủ 10,000
        total_needed = 10000
        current_count = len(unique_ids)
        
        if current_count < total_needed:
            for i in range(current_count + 1, total_needed + 1):
                new_id = f"ACC_{i:05d}"
                unique_ids.append(new_id)
                counts[new_id] = 0 # Những người này không có giao dịch nào
        
        # 3. Định nghĩa các giá trị ngẫu nhiên
        work_status = ["Employed", "Self-employed", "Freelancer", "Unemployed", "Student", "Retired"]
        locations = ["Hanoi", "HCMC", "Da Nang", "Hai Phong", "Can Tho", "Nha Trang", "Vung Tau"]
        
        final_customers = []
        
        # 4. Tạo dữ liệu chi tiết cho từng ID (KHÔNG CÓ LABEL)
        for c_id in unique_ids:
            # Giả lập logic tài chính
            salary_val = random.randint(5, 50) * 1000000
            balance_val = random.randint(1, 500) * 1000000
            
            customer_obj = {
                "Customer ID": c_id,
                "Date of Birth": f"{random.randint(1965, 2005)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                "Gender": random.choice(["Male", "Female", "Other"]),
                "Location": random.choice(locations),
                "Account balance": f"{balance_val:,}",
                "Transaction Count": counts.get(c_id, 0), # Khớp 100% với file transaction
                "Working Status": random.choice(work_status),
                "Salary (per month)": f"{salary_val:,}"
                # Cột Label đã được loại bỏ hoàn toàn
            }
            final_customers.append(customer_obj)

        # 5. Lưu ra file JSON
        with open(output_customer_file, 'w', encoding='utf-8') as f:
            json.dump(final_customers, f, indent=4, ensure_ascii=False)
            
        print(f"--- HOÀN TẤT ---")
        print(f"Đã tạo 10,000 khách hàng KHÔNG CÓ NHÃN tại: {output_customer_file}")
        print(f"Transaction Count đã được đồng bộ hóa với file: {transaction_file}")

    except Exception as e:
        print(f"Lỗi: {e}")

# Chạy lệnh
# Lưu ý: Bạn cần có file 'transaction.json' trong cùng thư mục trước khi chạy
generate_unlabeled_customers('transaction.json', 'customers.json')