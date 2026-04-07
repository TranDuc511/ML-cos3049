import json
import random
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_customers(output_file, total=10000):
    """
    Generate synthetic customer data with behavioral labels (internal only).
    
    Internal labels guide transaction generation but are NOT saved to output:
    - 'Normal': Regular banking customers (70%)
    - 'Gambling': Customers with gambling/betting patterns (15%)
    - 'Loan Sharking': Customers with high-risk lending patterns (15%)
    
    Final output has NO labels - for unsupervised ML clustering.
    """
    unique_ids = [f'ACC_{i:05d}' for i in range(1, total + 1)]

    work_statuses = ['Employed', 'Self-employed', 'Freelancer', 'Unemployed', 'Student', 'Retired']
    locations = ['Hanoi', 'HCMC', 'Da Nang', 'Hai Phong', 'Can Tho', 'Nha Trang', 'Vung Tau']
    
    # Internal label distribution for realistic data generation
    # (NOT saved to output - only used to guide transaction generation)
    label_distribution = ['Normal'] * 7 + ['Gambling'] * 2 + ['Loan Sharking'] * 1
    labels = label_distribution * (total // len(label_distribution) + 1)
    random.shuffle(labels)

    customers = []
    internal_labels = {}  # Track labels internally only
    
    for idx, cid in enumerate(unique_ids):
        salary = random.randint(5, 50) * 1_000_000
        balance = random.randint(1, 500) * 1_000_000
        label = labels[idx]
        
        internal_labels[cid] = label  # Store internally for transaction generation
        
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

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(customers, f, indent=4, ensure_ascii=False)

    # Print internal label distribution (for reference only)
    label_counts = {}
    for label in labels[:total]:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f'Generated {len(customers):,} customers -> {output_file}')
    print(f'(Internal label distribution: {label_counts})')
    
    return internal_labels  # Return for use in transaction generation


if __name__ == '__main__':
    # For standalone use - generates customers without labels
    labels = generate_customers(
        os.path.abspath(os.path.join(HERE, '..', 'customers.json')),
    )