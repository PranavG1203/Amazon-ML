import os
import random
import pandas as pd
from Processor import process_image


def predictor(image_link, category_id, entity_name):

    result = process_image(image_link, entity_name)

    return result

if __name__ == "__main__":
    
    # Load the dataset
    test = pd.read_csv('../dataset/batches/test_20023.csv')
    
    output_filename = 'test_out_29011.csv'

    # Write header only once, if file does not exist
    if not os.path.exists(output_filename):
        with open(output_filename, 'w') as f:
            f.write('index,prediction\n')

    # Loop over each row and write predictions incrementally
    for idx, row in test.iterrows():
        # Get prediction for the current row
        prediction = predictor(row['image_link'], row['group_id'], row['entity_name'])
        
        # Append the prediction to the CSV file
        with open(output_filename, 'a') as f:
            f.write(f"{row['index']},{prediction}\n")
        
        # Optionally print progress
        if idx % 100 == 0:  # Print every 100th image
            print(f"Processed {idx + 1}/{len(test)} images.")
    
    print("Prediction process complete.")
