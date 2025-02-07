import pandas as pd

# Load the CSV file
file_path = "data/weather_hourly_darksky.csv"  # Update with your actual file path
df = pd.read_csv(file_path)


# Drop unwanted columns
columns_to_remove = ["visibility", "windBearing", "dewPoint", "pressure" , "windSpeed" , "icon" , "humidity"]
df_cleaned = df.drop(columns=columns_to_remove, errors="ignore")

# Save the cleaned dataset
cleaned_file_path = "cleaned_daily_dataset.csv"
df_cleaned.to_csv(cleaned_file_path, index=False)

print(f"Cleaned dataset saved as {cleaned_file_path}")
