import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Define file paths
raw_data_path = "data/raw_ad_data.csv"
processed_data_path = "data/processed_data.csv"

# If raw data does not exist, generate synthetic data for demonstration
if not os.path.exists(raw_data_path):
    print("Generating synthetic raw data...")
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        "user_id": np.random.randint(1, 100, n_samples),
        "ad_id": np.random.randint(1, 50, n_samples),
        "click": np.random.binomial(1, 0.2, n_samples),
        "time_of_day": np.random.choice(["morning", "afternoon", "evening", "night"], n_samples),
        "device": np.random.choice(["mobile", "desktop", "tablet"], n_samples),
        "ad_category": np.random.choice(["sports", "fashion", "tech", "food"], n_samples),
        "price": np.random.uniform(0.5, 5.0, n_samples),
        "text_feature": np.random.choice(["discount", "new", "offer", "sale"], n_samples)
    })
    df.to_csv(raw_data_path, index=False)
else:
    df = pd.read_csv(raw_data_path)

print("Initial data shape:", df.shape)

# Handle missing values using forward fill
df.ffill(inplace=True)

# Feature engineering: create a time numeric feature
time_mapping = {"morning": 0, "afternoon": 1, "evening": 2, "night": 3}
df["time_numeric"] = df["time_of_day"].map(time_mapping)

# One-hot encode categorical variables: device, ad_category, text_feature
categorical_features = ["device", "ad_category", "text_feature"]
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[categorical_features])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

# Concatenate encoded features and drop original categorical columns
df_processed = pd.concat([df.drop(columns=categorical_features + ["time_of_day"]), encoded_df], axis=1)

# Save the processed data
os.makedirs("data", exist_ok=True)
df_processed.to_csv(processed_data_path, index=False)
print("Processed data saved to", processed_data_path)