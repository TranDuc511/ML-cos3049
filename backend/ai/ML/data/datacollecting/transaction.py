import json
import random
import os
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_transactions(customer_file, output_file, labels_file=None, n=25000):
    """
    Generate synthetic transactions based on internal customer behavior labels.
    
    The labels guide transaction patterns but are NOT saved to the output.
    This enables unsupervised ML models to discover fraud patterns on their own.
    
    Transaction patterns vary by internal customer label:
    - Normal: Regular domestic spending, business hours, modest amounts
    - Gambling: International transfers, odd hours, high amounts
    - Loan Sharking: Large transfers, risky patterns, specific details
    """
    with open(customer_file, 'r', encoding='utf-8') as f:
        customers = json.load(f)

    # Load internal labels (if provided)
    customer_labels = {}
    if labels_file and os.path.exists(labels_file):
        with open(labels_file, 'r', encoding='utf-8') as f:
            customer_labels = json.load(f)
    
    # Build lookup with default 'Normal' if labels unavailable
    customer_lookup = {c['Customer ID']: customer_labels.get(c['Customer ID'], 'Normal') 
                       for c in customers}
    all_ids = list(customer_lookup.keys())
    print(f'Loaded {len(all_ids):,} customers.')

    # Transaction patterns per internal label
    content_map = {
        'Normal': {
            'details': [
                'Supermarket', 'Electricity Bill', 'Monthly Salary', 'Restaurant',
                'Starbucks', 'Gas Station', 'Netflix Subscription', 'Grocery Store',
                'Phone Bill', 'Water Payment', 'Internet Subscription'
            ],
            'locations': ['Hanoi - VN', 'HCMC - VN', 'Da Nang - VN', 'Can Tho - VN', 'Hai Phong - VN'],
            'amount_range': (20_000, 10_000_000),
            'preferred_hours': list(range(7, 23)),  # Daytime transactions
        },
        'Gambling': {
            'details': [
                'Casino Online Top-up', 'Betting Wallet Deposit', 'Gaming Chip Purchase',
                'Virtual Slot Funding', 'P2P Game Transfer', 'Sports Betting Platform',
                'Online Poker Deposit'
            ],
            'locations': ['Singapore - SG', 'Macau - CN', 'Manila - PH', 'Cambodia - KH', 'Bangkok - TH'],
            'amount_range': (100_000, 50_000_000),
            'preferred_hours': [23, 0, 1, 2, 3, 4, 12, 13],  # Odd hours + afternoon
        },
        'Loan Sharking': {
            'details': [
                'Quick Loan Disbursement', 'Private Finance Support', 'Urgent Cash Out',
                'P2P Lending Transfer', 'Interest Payment Received', 'Emergency Fund Advance'
            ],
            'locations': ['Hanoi - VN', 'HCMC - VN', 'Hai Phong - VN', 'Da Nang - VN'],
            'amount_range': (5_000_000, 100_000_000),
            'preferred_hours': [23, 0, 1, 2, 3, 4, 22],  # Late night transactions
        },
    }

    devices = ['iPhone 15', 'Samsung S23', 'MacBook Air', 'Web Browser', 'Android Phone']

    transactions = []
    for i in range(n):
        sender_id = random.choice(all_ids)
        label = customer_lookup[sender_id]
        config = content_map.get(label, content_map['Normal'])

        # Select hour based on customer label behavior (internal only)
        hour = random.choice(config['preferred_hours'])

        # Generate random date in 2025
        date = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 364))
        timestamp = date.replace(hour=hour, minute=random.randint(0, 59)).strftime('%Y-%m-%d %H:%M:%S')

        # ✅ NO LABEL IN OUTPUT - models will discover patterns themselves
        transactions.append({
            'Transaction ID': f'TXN_{300001 + i}',
            'Sender Account ID': sender_id,
            'Receiver Account ID': f'REC_{random.randint(1000, 9999)}',
            'Transaction amount': random.randint(*config['amount_range']),
            'Timestamp': timestamp,
            'Transaction Detail': random.choice(config['details']),
            'Geological': random.choice(config['locations']),
            'Device Use': random.choice(devices),
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transactions, f, indent=4, ensure_ascii=False)

    print(f'Generated {len(transactions):,} transactions -> {output_file}')

    # Update customer transaction counts
    counts = {}
    for txn in transactions:
        sid = txn['Sender Account ID']
        counts[sid] = counts.get(sid, 0) + 1

    for c in customers:
        c['Transaction Count'] = counts.get(c['Customer ID'], 0)

    with open(customer_file, 'w', encoding='utf-8') as f:
        json.dump(customers, f, indent=4, ensure_ascii=False)

    print(f'Synced Transaction Counts back to -> {customer_file}')

    # Print transaction distribution by internal label (for reference only)
    label_txn_counts = {}
    for txn in transactions:
        sender_id = txn['Sender Account ID']
        label = customer_lookup[sender_id]
        label_txn_counts[label] = label_txn_counts.get(label, 0) + 1
    
    print(f'(Internal transaction distribution by label: {label_txn_counts})')


if __name__ == '__main__':
    # For standalone use - requires labels_file parameter
    # Recommended: use generate_data.py instead for full pipeline
    generate_transactions(
        os.path.join(HERE, '..', 'customers.json'),
        os.path.join(HERE, '..', 'transaction.json'),
        labels_file=None,  # Will use 'Normal' default for all
    )