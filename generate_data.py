"""
Run this once: python generate_data.py
Generates 2000 realistic Indian expense transactions → data/transactions.csv
"""

import pandas as pd
import random
from datetime import datetime, timedelta
import os

random.seed(42)

CATEGORIES = {
    "Food": [
        "Swiggy order", "Zomato delivery", "Dominos Pizza", "McDonald's",
        "KFC order", "Burger King", "Subway", "Local dhaba", "Cafe Coffee Day",
        "Starbucks", "Haldiram's", "Barbeque Nation", "Pizza Hut", "Dunkin Donuts"
    ],
    "Transport": [
        "Ola cab", "Uber ride", "Rapido bike", "BMTC bus pass", "Metro card recharge",
        "IRCTC train ticket", "IndiGo flight", "Air India ticket", "Petrol - HP pump",
        "Petrol - Indian Oil", "Fastag recharge", "RedBus ticket", "Ola outstation"
    ],
    "Shopping": [
        "Flipkart order", "Amazon India", "Myntra clothes", "Meesho purchase",
        "Ajio fashion", "Nykaa beauty", "Reliance Digital", "Croma electronics",
        "Big Bazaar", "DMart grocery", "Spencer's retail", "Lifestyle store"
    ],
    "Groceries": [
        "BigBasket order", "Blinkit grocery", "Zepto delivery", "JioMart",
        "Dunzo grocery", "Nature's Basket", "More supermarket", "Local kirana store",
        "Swiggy Instamart", "BlinkIt vegetables"
    ],
    "Bills & Utilities": [
        "BSNL broadband", "Jio recharge", "Airtel recharge", "Vi recharge",
        "BESCOM electricity", "TATA Power bill", "Piped gas bill", "Water tax",
        "Municipal tax", "Society maintenance", "DTH recharge - Tata Sky",
        "DTH recharge - Dish TV"
    ],
    "EMI & Finance": [
        "SBI home loan EMI", "HDFC car loan EMI", "ICICI personal loan EMI",
        "Axis Bank credit card bill", "Kotak credit card", "SBI credit card payment",
        "LIC premium", "HDFC Life insurance", "Term insurance premium",
        "Mutual fund SIP - Zerodha", "SIP - Groww", "PPF deposit"
    ],
    "Health": [
        "Apollo pharmacy", "MedPlus medicine", "1mg order", "Netmeds delivery",
        "Practo consultation", "Doctor consultation fee", "Fortis hospital",
        "Manipal hospital", "Diagnostic lab - SRL", "Thyrocare test"
    ],
    "Entertainment": [
        "BookMyShow movie", "PVR Cinemas", "INOX movies", "Netflix subscription",
        "Amazon Prime", "Hotstar subscription", "Spotify premium", "YouTube Premium",
        "Sony LIV", "Zee5 subscription", "Gaming - Steam"
    ],
    "Education": [
        "Udemy course", "Coursera subscription", "BYJU's fee", "Unacademy Plus",
        "College fee", "Coaching institute fee", "Books - Amazon", "Stationery",
        "LinkedIn Learning", "GeeksForGeeks premium"
    ],
    "Personal Care": [
        "Salon haircut", "Beauty parlour", "Laundry service - Urban Company",
        "Gym membership", "Cult.fit subscription", "Yoga class fee",
        "Nykaa skincare", "Myntra personal care"
    ]
}

PAYMENT_MODES = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash"]
PAYMENT_WEIGHTS = [0.50, 0.20, 0.15, 0.08, 0.07]

AMOUNT_RANGES = {
    "Food":               (80, 1200),
    "Transport":          (50, 4500),
    "Shopping":           (200, 8000),
    "Groceries":          (150, 3000),
    "Bills & Utilities":  (200, 3500),
    "EMI & Finance":      (2000, 25000),
    "Health":             (100, 5000),
    "Entertainment":      (99, 1500),
    "Education":          (500, 15000),
    "Personal Care":      (100, 2500),
}


def random_date(start_year=2023, end_year=2024):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return (start + timedelta(days=random.randint(0, delta.days))).strftime("%Y-%m-%d")


def generate_transactions(n=2000):
    rows = []
    for _ in range(n):
        category = random.choice(list(CATEGORIES.keys()))
        description = random.choice(CATEGORIES[category])
        low, high = AMOUNT_RANGES[category]
        amount = round(random.uniform(low, high), 2)
        payment_mode = random.choices(PAYMENT_MODES, weights=PAYMENT_WEIGHTS)[0]
        date = random_date()
        rows.append({
            "date": date,
            "description": description,
            "amount": amount,
            "category": category,
            "payment_mode": payment_mode,
            "is_anomaly": 0
        })

    # Inject 40 real anomalies — very high amounts in normally cheap categories
    for _ in range(40):
        category = random.choice(["Food", "Entertainment", "Personal Care", "Transport"])
        description = random.choice(CATEGORIES[category])
        _, high = AMOUNT_RANGES[category]
        amount = round(random.uniform(high * 2.5, high * 5), 2)
        rows.append({
            "date": random_date(),
            "description": description + " (unusual)",
            "amount": amount,
            "category": category,
            "payment_mode": random.choices(PAYMENT_MODES, weights=PAYMENT_WEIGHTS)[0],
            "is_anomaly": 1
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_transactions(2000)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv("data/transactions.csv", index=False)
    print(f"Generated {len(df)} transactions → data/transactions.csv")
    print(df["category"].value_counts())
    print(f"\nAmount range: ₹{df['amount'].min():.0f} – ₹{df['amount'].max():.0f}")
    print(f"Anomalies injected: {df['is_anomaly'].sum()}")