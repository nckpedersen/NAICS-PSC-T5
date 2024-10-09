import pandas as pd
from transformers import T5Tokenizer
from sklearn.model_selection import train_test_split

def load_data(csv_file):
    """Load the dataset from CSV."""
    data = pd.read_csv(csv_file)
    return data

def preprocess_data(data, tokenizer):
    """Preprocess dataset for different input-output combinations."""
    processed_data = []

    for idx, row in data.iterrows():
        naics = str(row['naics'])
        psc = str(row['psc'])
        description = str(row['description'])

        # Create multiple combinations for input-output pairs
        combinations = [
            (f"NAICS: {naics} Description: {description}", f"PSC: {psc}"),
            (f"NAICS: {naics} PSC: {psc}", f"Description: {description}"),
            (f"PSC: {psc} Description: {description}", f"NAICS: {naics}"),
            (f"NAICS: {naics}", f"PSC: {psc} Description: {description}"),
            (f"PSC: {psc}", f"NAICS: {naics} Description: {description}"),
            (f"Description: {description}", f"NAICS: {naics} PSC: {psc}")
        ]

        # Tokenize input-output pairs with padding and truncation to ensure same length
        for input_text, output_text in combinations:
            input_tokens = tokenizer.encode(input_text, return_tensors="pt", padding='max_length', truncation=True, max_length=512).squeeze().tolist()
            output_tokens = tokenizer.encode(output_text, return_tensors="pt", padding='max_length', truncation=True, max_length=512).squeeze().tolist()

            # Append processed pair to the list
            processed_data.append({'input_ids': input_tokens, 'labels': output_tokens})

    # Convert processed_data list to a Pandas DataFrame and return it
    processed_df = pd.DataFrame(processed_data)
    return processed_df

def split_data(data):
    """Split the dataset into training, validation, and test sets."""
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    return train_data, val_data, test_data