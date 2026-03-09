import json
import random
import os
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_transactions(customer_file, output_file, n=25000):
    with open(customer_file, 'r', encoding='utf-8') as f:
        customers = json.load(f)

    # Build a quick lookup: {Customer ID -> Label}
    customer_lookup = {c['Customer ID']: c.get('Label', 'Normal') for c in customers}
    all_ids = list(customer_lookup.keys())
    print(f'Loaded {len(all_ids):,} customers.')

    # Transaction content per label
    content_map = {
        'Normal': {
            'details': ['Supermarket', 'Electricity Bill', 'Monthly Salary', 'Restaurant',
                        'Starbucks', 'Gas Station', 'Netflix Subscription'],
            'locations':    ['Hanoi - VN', 'HCMC - VN', 'Da Nang - VN', 'Can Tho - VN'],
            'amount_range': (20_000, 10_000_000),       
        },
        'Gambling': {
            'details': ['Casino Online Top-up', 'Betting Wallet Deposit', 'Gaming Chip Purchase',
                        'Virtual Slot Funding', 'P2P Game Transfer'],
            'locations':    ['Singapore - SG', 'Macau - CN', 'Manila - PH', 'Cambodia - KH'],
            'amount_range': (100_000, 50_000_000),      
        },
        'Loan Sharking': {
            'details': ['Quick Loan Disbursement', 'Private Finance Support', 'Urgent Cash Out',
                        'P2P Lending Transfer', 'Interest Payment Received'],
            'locations':    ['Hanoi - VN', 'HCMC - VN', 'Hai Phong - VN'],
            'amount_range': (5_000_000, 100_000_000),  
        },
    }

    devices = ['iPhone 15', 'Samsung S23', 'MacBook Air', 'Web Browser', 'Android Phone']

    transactions = []
    for i in range(n):
        sender_id = random.choice(all_ids)
        label     = customer_lookup[sender_id]
        config    = content_map.get(label, content_map['Normal'])

        # Fraudulent transactions tend to happen at odd hours
        hour = random.choice([23, 0, 1, 2, 3, 4, 12, 13]) if label != 'Normal' else random.randint(7, 22)

        date      = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 364))
        timestamp = date.replace(hour=hour, minute=random.randint(0, 59)).strftime('%Y-%m-%d %H:%M:%S')

        transactions.append({
            'Transaction ID':     f'TXN_{300001 + i}',
            'Sender Account ID':  sender_id,
            'Receiver Account ID': f'REC_{random.randint(1000, 9999)}',
            'Transaction amount': random.randint(*config['amount_range']),
            'Timestamp':          timestamp,
            'Transaction Detail': random.choice(config['details']),
            'Geological':         random.choice(config['locations']),
            'Device Use':         random.choice(devices),
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transactions, f, indent=4, ensure_ascii=False)

    print(f'Generated {len(transactions):,} transactions -> {output_file}')


if __name__ == '__main__':
    generate_transactions(
        os.path.join(HERE, 'customers.json'),
        os.path.join(HERE, 'transaction.json'),
    )