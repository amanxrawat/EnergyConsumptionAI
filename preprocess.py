import os
import pandas as pd

# Define the folder path containing the CSV files
folder_path = "data/daily_dataset"  # Update with the actual path

# Define columns to remove
columns_to_remove = ["LCLid", "energy_median", "energy_mean", "energy_max" ,"energy_min" ,"energy_count"]


# Create a folder to save cleaned files
cleaned_folder_path = "cleaned_daily_dataset"
os.makedirs(cleaned_folder_path, exist_ok=True)

# Process each CSV file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):  # Process only CSV files
        file_path = os.path.join(folder_path, filename)
        
        # Load CSV file
        df = pd.read_csv(file_path)
        
        # Drop unwanted columns
        df_cleaned = df.drop(columns=columns_to_remove, errors="ignore")
        
        # Save the cleaned file
        cleaned_file_path = os.path.join(cleaned_folder_path, filename)
        df_cleaned.to_csv(cleaned_file_path, index=False)
        
        print(f"Processed: {filename} -> Saved to {cleaned_file_path}")

print("All files processed successfully!")
