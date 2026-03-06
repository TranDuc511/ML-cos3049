import json
import random
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_customers(transaction_file, output_file, total=10000):
    # Sync transaction counts if the transaction file already exists
    counts = {}
    unique_ids = []
    if os.path.exists(transaction_file):
        with open(transaction_file, 'r', encoding='utf-8') as f:
            transactions = json.load(f)
        for txn in transactions:
            sid = txn['Sender Account ID']
            counts[sid] = counts.get(sid, 0) + 1
        unique_ids = list(counts.keys())

    # Fill up to total
    for i in range(len(unique_ids) + 1, total + 1):
        new_id = f'ACC_{i:05d}'
        unique_ids.append(new_id)
        counts[new_id] = 0

    work_statuses = ['Employed', 'Self-employed', 'Freelancer', 'Unemployed', 'Student', 'Retired']
    locations     = ['Hanoi', 'HCMC', 'Da Nang', 'Hai Phong', 'Can Tho', 'Nha Trang', 'Vung Tau']

    customers = []
    for cid in unique_ids:
        salary  = random.randint(5, 50) * 1_000_000
        balance = random.randint(1, 500) * 1_000_000
        customers.append({
            'Customer ID':       cid,
            'Date of Birth':     f'{random.randint(1965, 2005)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}',
            'Gender':            random.choice(['Male', 'Female', 'Other']),
            'Location':          random.choice(locations),
            'Account balance':   balance,
            'Transaction Count': counts.get(cid, 0),
            'Working Status':    random.choice(work_statuses),
            'Salary (per month)': salary,
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(customers, f, indent=4, ensure_ascii=False)

    print(f'Generated {len(customers):,} customers -> {output_file}')


if __name__ == '__main__':
    generate_customers(
        os.path.join(HERE, 'transaction.json'),
        os.path.join(HERE, 'customers.json'),
    )