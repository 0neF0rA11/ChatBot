import pandas as pd

# Выбирает 2500 уникальных цитат из kaggle данных
df=pd.read_json('quotes/quotes.json')
unique_quotes = df['Quote'].unique()

result_df = pd.DataFrame(unique_quotes, columns=['Quote'])
result_df = result_df.sample(n=2500)

result_df.Quote.to_csv('file.txt', index=False, header=False)