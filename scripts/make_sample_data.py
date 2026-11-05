"""Generate a small synthetic real estate dataset so the pipeline has something
to chew on. Prices are roughly: base by bedrooms + size + a bit of noise."""
import numpy as np
import pandas as pd
import os

rng = np.random.default_rng(42)
n = 500

bedrooms = rng.integers(1, 6, n)
bathrooms = np.clip(bedrooms * 0.7 + rng.normal(0, 0.5, n), 1, 5).round(1)
square_feet = rng.integers(500, 4500, n)
year_built = rng.integers(1920, 2024, n)

price = (
    50_000
    + bedrooms * 22_000
    + bathrooms * 15_000
    + square_feet * 110
    - (2024 - year_built) * 300
    + rng.normal(0, 25_000, n)
).round(0)

df = pd.DataFrame({
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "square_feet": square_feet,
    "year_built": year_built,
    "price": price,
})

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/properties.csv", index=False)
print(f"Wrote {len(df)} rows to data/raw/properties.csv")
