import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def descriptive_analysis():
    # Load features
    df = pd.read_csv('../results/cow_features.csv')
    
    print("--- TASK 4: Descriptive Analysis ---")
    print(f"Dataset Size: {df.shape[0]} images")
    print(f"Number of Animals: {df['cow_id'].nunique()}")
    
    # 1. Feature Correlation (See which features are redundant)
    numeric_df = df.drop(columns=['filename', 'cow_id'])
    correlation = numeric_df.corr()
    print("\nFeature Correlation Matrix (Top Relationships):")
    print(correlation.unstack().sort_values(ascending=False).drop_duplicates().head(10))

    # 2. Variance Analysis (High variance = better for identification)
    variation = numeric_df.std() / numeric_df.mean() # Coefficient of Variation
    print("\nCoefficient of Variation (Higher means more unique across cows):")
    print(variation.sort_values(ascending=False))

    # 3. Cow Profile Example
    print("\nMean Biometric Profile for first 5 cows:")
    profiles = df.groupby('cow_id').mean(numeric_only=True)
    print(profiles.head())

    # Optional: Save a summary to markdown
    profiles.to_markdown('../docs/cow_biometric_summary.md')
    print("\nSummary saved to '../docs/cow_biometric_summary.md'")

if __name__ == "__main__":
    descriptive_analysis()
