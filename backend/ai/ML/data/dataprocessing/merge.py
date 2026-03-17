import os
import pandas as pd

HERE = os.path.dirname(__file__)

cust  = pd.read_json(os.path.join(HERE, '..', 'customers.json'))
trans = pd.read_json(os.path.join(HERE, '..', 'transaction.json'))

df = pd.merge(trans, cust, left_on='Sender Account ID', right_on='Customer ID', how='left')
df = df.drop(columns=['Customer ID'])
df = df.fillna(0)

df.to_json(os.path.join(HERE, '..', 'data_2', 'data.json'), orient='records', indent=4)
print("Done")
