import pandas as pd

cust  = pd.read_json('ML/datacollecting/customers.json')
trans = pd.read_json('ML/datacollecting/transaction.json')

df = pd.merge(trans, cust, left_on='Sender Account ID', right_on='Customer ID', how='left')
df = df.drop(columns=['Customer ID'])
df = df.fillna(0)

df.to_json('ML/data/data.json', orient='records', indent=4)
print("Done")
