import pandas as pd
import os

def convert_to_csv():
    input_path = 'data/SMSSpamCollection'
    output_path = 'data/spam.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    try:
        df = pd.read_csv(input_path, sep='\t', names=['label', 'message'])
        df.to_csv(output_path, index=False)
        print(f"Successfully converted {input_path} to {output_path}")
        print(f"Shape: {df.shape}")
        print(df.head())
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    convert_to_csv()
