"""Create sample datasets for testing data ingestion"""
import pandas as pd
import numpy as np
from pathlib import Path

# Create data directory
data_dir = Path("./data/sample_datasets")
data_dir.mkdir(parents=True, exist_ok=True)

# Sample 1: Iris-like dataset
print("Creating Iris-like dataset...")
iris_data = {
    'sepal_length': np.random.uniform(4.3, 7.9, 150),
    'sepal_width': np.random.uniform(2.0, 4.4, 150),
    'petal_length': np.random.uniform(1.0, 6.9, 150),
    'petal_width': np.random.uniform(0.1, 2.5, 150),
    'species': np.random.choice(['setosa', 'versicolor', 'virginica'], 150)
}
iris_df = pd.DataFrame(iris_data)
iris_path = data_dir / "iris_sample.csv"
iris_df.to_csv(iris_path, index=False)
print(f"✅ Created: {iris_path} ({len(iris_df)} rows)")

# Sample 2: Titanic-like dataset
print("Creating Titanic-like dataset...")
titanic_data = {
    'PassengerId': range(1, 892),
    'Pclass': np.random.choice([1, 2, 3], 891),
    'Name': [f"Passenger_{i}" for i in range(1, 892)],
    'Sex': np.random.choice(['male', 'female'], 891),
    'Age': np.random.uniform(0.42, 80, 891),
    'SibSp': np.random.choice([0, 1, 2, 3, 4, 5], 891),
    'Parch': np.random.choice([0, 1, 2, 3, 4, 5, 6], 891),
    'Fare': np.random.uniform(0, 512, 891),
    'Embarked': np.random.choice(['C', 'Q', 'S'], 891),
    'Survived': np.random.choice([0, 1], 891)
}
titanic_df = pd.DataFrame(titanic_data)
titanic_path = data_dir / "titanic_sample.csv"
titanic_df.to_csv(titanic_path, index=False)
print(f"✅ Created: {titanic_path} ({len(titanic_df)} rows)")

# Sample 3: House prices dataset
print("Creating House Prices dataset...")
prices_data = {
    'Id': range(1, 1461),
    'MSSubClass': np.random.choice([20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 150, 160, 170, 180, 190], 1460),
    'LotFrontage': np.random.uniform(21, 313, 1460),
    'LotArea': np.random.uniform(1300, 215245, 1460),
    'Street': np.random.choice(['Pave', 'Grvl'], 1460),
    'Alley': np.random.choice(['NA', 'Pave', 'Grvl'], 1460),
    'YearBuilt': np.random.uniform(1872, 2010, 1460).astype(int),
    'Condition1': np.random.choice(['Artery', 'Feedr', 'Norm', 'RRAe', 'RRan', 'RRnn'], 1460),
    'SalePrice': np.random.uniform(34900, 755000, 1460)
}
prices_df = pd.DataFrame(prices_data)
prices_path = data_dir / "house_prices_sample.csv"
prices_df.to_csv(prices_path, index=False)
print(f"✅ Created: {prices_path} ({len(prices_df)} rows)")

print(f"\n✅ All sample datasets created in: {data_dir}")
print(f"\nYou can now use these local file paths in the ingestion panel:")
print(f"  - {iris_path}")
print(f"  - {titanic_path}")
print(f"  - {prices_path}")
