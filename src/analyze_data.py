import pandas as pd
import os

file_path = r'c:\Users\lewon\keiba\learn\data\raw\results_2016_2025.csv'

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)

print("--- DataFrame Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Sample Data (Head) ---")
print(df.head())

print("\n--- Unique Values in 'distance' (Top 20) ---")
print(df['distance'].value_counts().head(20))

print("\n--- Unique Values in 'weather' ---")
print(df['weather'].unique())

print("\n--- Unique Values in 'condition' ---")
print(df['condition'].unique())

print("\n--- Sample 'odds' columns ---")
print(df['odds'].unique()[:20])

print("\n--- Sample 'popularity' columns ---")
print(df['popularity'].unique()[:20])

print("\n--- 'date' column description ---")
print(df['date'].describe())
print(df['date'].unique()[:20])

# Time - Distance analysis
def time_to_seconds(t_str):
    try:
        if pd.isna(t_str): return None
        parts = t_str.split(':')
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(t_str)
    except:
        return None

df['seconds'] = df['time'].apply(time_to_seconds)
print("\n--- Distance vs Mean Time ---")
print(df.groupby('distance')['seconds'].agg(['count', 'mean', 'min', 'max']).sort_values('mean'))

print("\n--- Odds Non-Null Count ---")
print(df['odds'].count())

print("\n--- Race ID vs Date ---")
# Check if sorting by race_id aligns with sorting by date
df_sorted = df.sort_values('race_id')
df_sorted['date_int'] = pd.to_numeric(df_sorted['date'], errors='coerce')
print("Is Date monotonic increasing with Race ID?", df_sorted['date_int'].is_monotonic_increasing)
print(df_sorted[['race_id', 'date']].head(10))
print(df_sorted[['race_id', 'date']].tail(10))
