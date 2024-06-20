import pandas as pd
import numpy as np
import random
import os
from sklearn.preprocessing import LabelEncoder

# Ensure directories exist
os.makedirs(r'C:\Users\User\PycharmProjects\Bacteriocin2\Data\processed', exist_ok=True)


# Function to load raw data
def load_raw_data(bacteriocin_file, non_bacteriocin_file):
    bacteriocin_data = pd.read_csv(bacteriocin_file)
    non_bacteriocin_data = pd.read_csv(non_bacteriocin_file)

    # Add labels
    bacteriocin_data['Label'] = 1
    non_bacteriocin_data['Label'] = 0

    # Combine data
    data = pd.concat([bacteriocin_data, non_bacteriocin_data], ignore_index=True)
    return data


# Function to clean data
def clean_data(data):
    # Drop duplicates
    data = data.drop_duplicates()

    # Fill or drop missing values
    data = data.dropna(subset=['Sequence'])

    # Convert text to lowercase (if applicable)
    data['Sequence'] = data['Sequence'].str.lower()

    return data


# Function to augment data (simple example using sequence shuffling)
def augment_data(data, num_augments=1):
    augmented_data = []

    for _, row in data.iterrows():
        sequence = row['Sequence']
        label = row['Label']
        augmented_data.append((sequence, label))

        for _ in range(num_augments):
            augmented_sequence = ''.join(random.sample(sequence, len(sequence)))
            augmented_data.append((augmented_sequence, label))

    augmented_df = pd.DataFrame(augmented_data, columns=['Sequence', 'Label'])
    return augmented_df


# Main processing function
def process_data(bacteriocin_file, non_bacteriocin_file, output_file):
    data = load_raw_data(bacteriocin_file, non_bacteriocin_file)
    print(f"Loaded data with {len(data)} records")

    data = clean_data(data)
    print(f"Cleaned data with {len(data)} records")

    augmented_data = augment_data(data, num_augments=2)  # Adjust num_augments as needed
    print(f"Augmented data with {len(augmented_data)} records")

    augmented_data.to_csv(output_file, index=False)
    print(f"Saved processed data to {output_file}")


# Paths to input and output files
bacteriocin_file = r'C:\Users\User\PycharmProjects\Bacteriocin2\Data\raw\bacteriocin_amino_acid_sequences.csv'
non_bacteriocin_file = r'C:\Users\User\PycharmProjects\Bacteriocin2\Data\raw\non_bacteriocin_amino_acid_sequences.csv'
output_file = r'C:\Users\User\PycharmProjects\Bacteriocin2\Data\processed\augmented_data.csv'

# Process the data
process_data(bacteriocin_file, non_bacteriocin_file, output_file)
