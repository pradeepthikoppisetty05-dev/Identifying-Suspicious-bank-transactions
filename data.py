import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import json

fake = Faker()

class BankTransactionGenerator:
    def __init__(self, num_transactions=10000):
        self.num_transactions = num_transactions
        self.transaction_types = ['ATM_WITHDRAWAL', 'ONLINE_PURCHASE', 'TRANSFER', 'DEPOSIT', 'BILL_PAYMENT', 'CASH_WITHDRAWAL']
        self.merchants = ['Amazon', 'Walmart', 'Target', 'Starbucks', 'Shell', 'McDonald\'s', 'Best Buy', 'Home Depot', 'Costco', 'CVS']
        
    def generate_normal_transaction(self):
        """Generate a normal transaction pattern"""
        return {
            'transaction_id': fake.uuid4(),
            'account_id': fake.random_int(min=1000, max=9999),
            'amount': round(random.uniform(10, 1000), 2),
            'transaction_type': random.choice(self.transaction_types),
            'merchant': random.choice(self.merchants),
            'location': fake.city(),
            'timestamp': fake.date_time_between(start_date='-1y', end_date='now'),
            'account_balance_before': round(random.uniform(1000, 50000), 2),
            'is_weekend': random.choice([0, 1]),
            'hour_of_day': random.randint(6, 23),
            'is_suspicious': 0
        }
    
    def generate_suspicious_transaction(self):
        """Generate suspicious transaction patterns"""
        suspicious_patterns = [
            # Large amount transactions
            {'amount': round(random.uniform(5000, 20000), 2), 'reason': 'large_amount'},
            # Unusual time transactions
            {'hour_of_day': random.randint(0, 5), 'amount': round(random.uniform(500, 2000), 2), 'reason': 'unusual_time'},
            # Rapid consecutive transactions
            {'amount': round(random.uniform(100, 500), 2), 'reason': 'rapid_transactions'},
            # Unusual location
            {'location': fake.city(), 'amount': round(random.uniform(200, 1500), 2), 'reason': 'unusual_location'},
        ]
        
        pattern = random.choice(suspicious_patterns)
        
        transaction = {
            'transaction_id': fake.uuid4(),
            'account_id': fake.random_int(min=1000, max=9999),
            'amount': pattern.get('amount', round(random.uniform(10, 1000), 2)),
            'transaction_type': random.choice(self.transaction_types),
            'merchant': random.choice(self.merchants),
            'location': pattern.get('location', fake.city()),
            'timestamp': fake.date_time_between(start_date='-1y', end_date='now'),
            'account_balance_before': round(random.uniform(1000, 50000), 2),
            'is_weekend': random.choice([0, 1]),
            'hour_of_day': pattern.get('hour_of_day', random.randint(6, 23)),
            'is_suspicious': 1
        }
        
        return transaction
    
    def generate_dataset(self):
        """Generate complete dataset with normal and suspicious transactions"""
        transactions = []
        
        # Generate 85% normal transactions
        normal_count = int(self.num_transactions * 0.85)
        for _ in range(normal_count):
            transactions.append(self.generate_normal_transaction())
        
        # Generate 15% suspicious transactions
        suspicious_count = self.num_transactions - normal_count
        for _ in range(suspicious_count):
            transactions.append(self.generate_suspicious_transaction())
        
        # Shuffle the transactions
        random.shuffle(transactions)
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Add derived features
        df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        df['balance_ratio'] = df['amount'] / df['account_balance_before']
        df['is_high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        df['is_night_transaction'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)
        
        return df
    
    def save_dataset(self, filename='bank_transactions.csv'):
        """Generate and save dataset to CSV"""
        df = self.generate_dataset()
        df.to_csv(filename, index=False)
        print(f"Dataset generated and saved to {filename}")
        print(f"Total transactions: {len(df)}")
        print(f"Suspicious transactions: {df['is_suspicious'].sum()}")
        print(f"Normal transactions: {len(df) - df['is_suspicious'].sum()}")
        return df

if __name__ == "__main__":
    # Generate dataset
    generator = BankTransactionGenerator(num_transactions=15000)
    dataset = generator.save_dataset('bank_transactions.csv')
    
    # Display sample data
    print("\nSample transactions:")
    print(dataset.head(10))
    
    print("\nDataset info:")
    print(dataset.info())
    
    print("\nSuspicious vs Normal distribution:")
    print(dataset['is_suspicious'].value_counts())