from sklearn.datasets import fetch_california_housing
import pandas as pd

# Download as a pandas DataFrame
housing = fetch_california_housing(as_frame=True)

# Full table: features + target
df = housing.frame

# Save to CSV with header, without row index
df.to_csv("regression/california_housing/california_housing.csv", index=False)

print("Saved california_housing.csv")
print(df.head())