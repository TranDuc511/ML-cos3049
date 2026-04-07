"""
Unified data generation script for transaction anomaly detection.

This script generates:
1. customers.json - Customer profiles (NO labels in output)
2. transaction.json - Transaction records (NO labels in output)

Internally, it uses customer behavior labels (Normal/Gambling/Loan Sharking)
to generate realistic patterns. These labels guide the generation but are
NOT included in the final output, allowing ML models to do unsupervised
clustering/anomaly detection.
"""

import json
import random
import os
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_customers_with_labels(output_file, total=10000):
    """
    Generate customers with internal labels (not saved to output).
    Returns internal labels for use by transaction generation.
    """
    unique_ids = [f'ACC_{i:05d}' for i in range(1, total + 1)]
    work_statuses = ['Employed', 'Self-employed', 'Freelancer', 'Unemployed', 'Student', 'Retired']
    locations = ['Hanoi', 'HCMC', 'Da Nang', 'Hai Phong', 'Can Tho', 'Nha Trang', 'Vung Tau']
    
    # Internal label distribution (70/15/15)
    label_distribution = ['Normal'] * 7 + ['Gambling'] * 2 + ['Loan Sharking'] * 1
    labels = label_distribution * (total // len(label_distribution) + 1)
    random.shuffle(labels)

    customers = []
    internal_labels = {}
    
    for idx, cid in enumerate(unique_ids):
        salary = random.randint(5, 50) * 1_000_000
        balance = random.randint(1, 500) * 1_000_000
        label = labels[idx]
        
        internal_labels[cid] = label
        
        customers.append({
            'Customer ID': cid,
            'Date of Birth': f'{random.randint(1965, 2005)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}',
            'Gender': random.choice(['Male', 'Female']),
            'Location': random.choice(locations),
            'Account balance': balance,
            'Transaction Count': 0,
            'Working Status': random.choice(work_statuses),
            'Salary (per month)': salary,
        })

    # Save customers (NO labels)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(customers, f, indent=4, ensure_ascii=False)

    # CRITICAL: Save internal labels to file for transaction generation
    labels_file = output_file.replace('customers.json', '.internal_labels.json')
    with open(labels_file, 'w', encoding='utf-8') as f:
        json.dump(internal_labels, f, indent=4, ensure_ascii=False)

    # Print summary
    label_counts = {}
    for label in labels[:total]:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f'✓ Generated {len(customers):,} customers -> {output_file}')
    print(f'  (Internal label distribution: {label_counts})')
    print(f'  (Labels saved to .internal_labels.json for transaction generation)')
    
    return internal_labels


def generate_transactions_with_labels(customer_file, output_file, internal_labels=None, n=25000):
    """
    Generate transactions using internal labels to create realistic patterns.
    Labels are NOT saved to output.
    """
    with open(customer_file, 'r', encoding='utf-8') as f:
        customers = json.load(f)

    all_ids = [c['Customer ID'] for c in customers]
    
    # Load internal labels from file if not provided
    if internal_labels is None:
        labels_file = customer_file.replace('customers.json', '.internal_labels.json')
        if os.path.exists(labels_file):
            with open(labels_file, 'r', encoding='utf-8') as f:
                internal_labels = json.load(f)
        else:
            # Fallback: create with all Normal labels
            internal_labels = {cid: 'Normal' for cid in all_ids}
            print(f'⚠ No internal labels file found. Using default "Normal" for all customers.')
    
    print(f'✓ Loaded {len(all_ids):,} customers.')
    
    # Verify all customers have labels
    missing_labels = [cid for cid in all_ids if cid not in internal_labels]
    if missing_labels:
        print(f'⚠ WARNING: {len(missing_labels)} customers missing labels. Using default "Normal".')
        for cid in missing_labels:
            internal_labels[cid] = 'Normal'

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
            'preferred_hours': list(range(7, 23)),
        },
        'Gambling': {
            'details': [
                'Casino Online Top-up', 'Betting Wallet Deposit', 'Gaming Chip Purchase',
                'Virtual Slot Funding', 'P2P Game Transfer', 'Sports Betting Platform',
                'Online Poker Deposit'
            ],
            'locations': ['Singapore - SG', 'Macau - CN', 'Manila - PH', 'Cambodia - KH', 'Bangkok - TH'],
            'amount_range': (100_000, 50_000_000),
            'preferred_hours': [23, 0, 1, 2, 3, 4, 12, 13],
        },
        'Loan Sharking': {
            'details': [
                'Quick Loan Disbursement', 'Private Finance Support', 'Urgent Cash Out',
                'P2P Lending Transfer', 'Interest Payment Received', 'Emergency Fund Advance'
            ],
            'locations': ['Hanoi - VN', 'HCMC - VN', 'Hai Phong - VN', 'Da Nang - VN'],
            'amount_range': (5_000_000, 100_000_000),
            'preferred_hours': [23, 0, 1, 2, 3, 4, 22],
        },
    }

    devices = ['iPhone 15', 'Samsung S23', 'MacBook Air', 'Web Browser', 'Android Phone']

    transactions = []
    for i in range(n):
        sender_id = random.choice(all_ids)
        label = internal_labels.get(sender_id, 'Normal')
        config = content_map.get(label, content_map['Normal'])

        hour = random.choice(config['preferred_hours'])
        date = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 364))
        timestamp = date.replace(hour=hour, minute=random.randint(0, 59)).strftime('%Y-%m-%d %H:%M:%S')

        # ✓ NO LABEL IN OUTPUT
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

    # Save transactions (NO labels)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transactions, f, indent=4, ensure_ascii=False)

    print(f'✓ Generated {len(transactions):,} transactions -> {output_file}')

    # Update customer transaction counts
    counts = {}
    for txn in transactions:
        sid = txn['Sender Account ID']
        counts[sid] = counts.get(sid, 0) + 1

    for c in customers:
        c['Transaction Count'] = counts.get(c['Customer ID'], 0)

    with open(customer_file, 'w', encoding='utf-8') as f:
        json.dump(customers, f, indent=4, ensure_ascii=False)

    # Print internal distribution (for reference only)
    label_txn_counts = {}
    for txn in transactions:
        sender_id = txn['Sender Account ID']
        label = internal_labels.get(sender_id, 'Normal')
        label_txn_counts[label] = label_txn_counts.get(label, 0) + 1
    
    print(f'✓ Synced Transaction Counts back to customers.json')
    print(f'  (Internal transaction distribution: {label_txn_counts})')


def main():
    """Run complete data generation pipeline."""
    customer_file = os.path.join(HERE, 'customers.json')
    transaction_file = os.path.join(HERE, 'transaction.json')
    
    print('\n' + '='*60)
    print('Transaction Anomaly Detection - Data Generation')
    print('='*60 + '\n')
    
    print('[1/2] Generating customers...')
    internal_labels = generate_customers_with_labels(customer_file, total=10000)
    
    print('\n[2/2] Generating transactions...')
    generate_transactions_with_labels(customer_file, transaction_file, internal_labels=internal_labels, n=25000)
    
    print('\n' + '='*60)
    print('✓ Data generation complete!')
    print('='*60)
    print(f'\nOutput files:')
    print(f'  - {customer_file} ({len(internal_labels):,} customers, NO labels)')
    print(f'  - {transaction_file} (25,000 transactions, NO labels)')
    print(f'\nTransaction patterns generated:')
    
    # Count transaction types by internal label
    label_counts = {}
    for label in internal_labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    
    for label, count in sorted(label_counts.items()):
        pct = (count / len(internal_labels)) * 100
        print(f'  - {label}: ~{count:,} customers (~{pct:.1f}%) → varied patterns')
    
    print(f'\nReady for unsupervised ML clustering/anomaly detection!')
    print('='*60 + '\n')


if __name__ == '__main__':
    main()