import pandas as pd
import numpy as np
from faker import Faker
import random
import os

fake = Faker()
Faker.seed(42)
random.seed(42)

DATA_DIR = "data_source"
os.makedirs(DATA_DIR, exist_ok=True)

def generate_data(num_records=12000):
    print(f"Generating {num_records} synthetic records...")
    
    data = []
    for _ in range(num_records):
        company_base = fake.company()
        suffix = random.choice([" Inc.", " LLC", " Corp", "", " Limited"])
        
        # Source 1 (Supply Chain)
        name_s1 = company_base + suffix
        address = fake.address().replace("\n", ", ")
        suppliers = [fake.company() for _ in range(random.randint(1, 5))]
        
        # Source 2 (Financial) - 70% chance of overlap
        if random.random() < 0.7:
            name_s2 = company_base.upper() if random.random() < 0.5 else company_base + random.choice(["", "."])
            revenue = round(random.uniform(10000, 5000000), 2)
            profit = revenue * random.uniform(-0.1, 0.4) 
            main_customers = [fake.company() for _ in range(random.randint(1, 3))]
        else:
            name_s2 = None
            revenue = None
            profit = None
            main_customers = []

        data.append({
            "name_s1": name_s1,
            "address": address,
            "top_suppliers": ",".join(suppliers),
            "name_s2": name_s2,
            "revenue": revenue,
            "profit": profit,
            "main_customers": ",".join(main_customers) if main_customers else None
        })

    df = pd.DataFrame(data)

    # Split into two distinct CSV sources
    df_s1 = df[['name_s1', 'address', 'top_suppliers']].copy()
    df_s1.columns = ['corporate_name_S1', 'address', 'top_suppliers']
    
    df_s2 = df[['name_s2', 'revenue', 'profit', 'main_customers']].dropna(subset=['name_s2']).copy()
    df_s2.columns = ['corporate_name_S2', 'revenue', 'profit', 'main_customers']

    df_s1.to_csv(f"{DATA_DIR}/source_supply_chain.csv", index=False)
    df_s2.to_csv(f"{DATA_DIR}/source_financial.csv", index=False)
    
    print(f"Data Generated at {DATA_DIR}/")

if __name__ == "__main__":
    generate_data()