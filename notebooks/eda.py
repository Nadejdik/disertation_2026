import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import zipfile

# Set matplotlib style for better academic appearance
plt.style.use('seaborn-v0_8')

# Create plots directory if it doesn't exist
plots_dir = "plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Function to extract all CSV files from ZIP archives
def extract_csv_files_from_zip(zip_path):
    """
    Extract all CSV files from a ZIP archive and return a list of file paths.
    """
    csv_files = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get all files in the ZIP archive
            file_list = zip_ref.namelist()
            
            # Filter for CSV files
            for file in file_list:
                if file.endswith('.csv'):
                    # Extract the file to a temporary directory
                    temp_dir = "temp_extracted"
                    os.makedirs(temp_dir, exist_ok=True)
                    zip_ref.extract(file, temp_dir)
                    
                    # Add the extracted file path to the list
                    csv_files.append(os.path.join(temp_dir, file))
                    
                    # Optional: Print extraction status
                    print(f"Extracted: {file}")
    except Exception as e:
        print(f"Error extracting files from {zip_path}: {str(e)}")
    
    return csv_files

# Function to perform EDA on a single dataset
def perform_eda_on_dataset(df, dataset_name):
    """
    Perform comprehensive Exploratory Data Analysis (EDA) on a single dataset.
    """
    print(f"\n=== EDA for {dataset_name} ===")
    
    # 1. Print dataset shape, column names, and data types
    print(f"Dataset shape: {df.shape}")
    print(f"Column names: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    
    # 2. Check for missing values and duplicate rows
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    
    # 3. Generate descriptive statistics for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        print(f"\nDescriptive statistics for numeric columns:\n{df[numeric_columns].describe().round(2)}")
    else:
        print("No numeric columns found for descriptive statistics.")
    
    # 4. Plot distributions for numeric columns (histograms with KDE)
    if len(numeric_columns) > 0:
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(numeric_columns):
            plt.subplot(len(numeric_columns), 1, i+1)
            sns.histplot(df[col], kde=True, bins=30, color='blue')
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{dataset_name}_distributions.png'))
        plt.close()
        print(f"Saved distribution plots for {dataset_name} to {plots_dir}")
    else:
        print("No numeric columns to plot distributions for.")
    
    # 5. Plot counts for categorical columns (bar charts)
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_columns) > 0:
        plt.figure(figsize=(12, len(categorical_columns)*2))
        for i, col in enumerate(categorical_columns):
            plt.subplot(len(categorical_columns), 1, i+1)
            df[col].value_counts().plot(kind='bar', color='green')
            plt.title(f"Count of {col}")
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{dataset_name}_categorical_counts.png'))
        plt.close()
        print(f"Saved categorical count plots for {dataset_name} to {plots_dir}")
    else:
        print("No categorical columns to plot counts for.")
    
    # 6. Plot correlation heatmap for numeric columns
    if len(numeric_columns) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title(f"Correlation Heatmap for {dataset_name}")
        plt.savefig(os.path.join(plots_dir, f'{dataset_name}_correlation_heatmap.png'))
        plt.close()
        print(f"Saved correlation heatmap for {dataset_name} to {plots_dir}")
    else:
        print("Not enough numeric columns for correlation heatmap.")
    
    # Print summary statistics
    print(f"\nSummary for {dataset_name}:")
    print(f"- Shape: {df.shape}")
    print(f"- Missing values: {df.isnull().sum().sum()}")
    print(f"- Duplicates: {df.duplicated().sum()}")
    
    # Return the dataframe for further processing
    return df

# Main function to perform EDA on all CSV files
def main():
    """
    Main function to perform EDA on all CSV files in the specified directory.
    """
    # Define the directory path
    data_dir = "disertation_2026/datasets/AIOps-Challenge-2020-Data-main"
    
    # Get all CSV files directly in the directory
    csv_files = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(data_dir, file))
    
    # Get all ZIP files in the directory
    zip_files = []
    for file in os.listdir(data_dir):
        if file.endswith('.zip'):
            zip_files.append(os.path.join(data_dir, file))
    
    # Extract CSV files from ZIP archives
    for zip_file in zip_files:
        extracted_csv_files = extract_csv_files_from_zip(zip_file)
        csv_files.extend(extracted_csv_files)
    
    # Check if any CSV files were found
    if not csv_files:
        print("No CSV files found in the specified directory or ZIP archives.")
        return
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Get the dataset name (filename without extension)
            dataset_name = os.path.basename(csv_file).replace('.csv', '')
            
            # Perform EDA on the dataset
            perform_eda_on_dataset(df, dataset_name)
            
            print(f"\n=== Completed EDA for {dataset_name} ===\n")
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

# Run the main function
if __name__ == "__main__":
    main()