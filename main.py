import os
import pandas as pd
from Processor import process_image


def predictor(image_link, category_id, entity_name):

    result = process_image(image_link, entity_name)

    return result

if __name__ == "__main__":

    test = pd.read_csv('test.csv')
    
    output_filename = 'test_out.csv'

    if not os.path.exists(output_filename):
        with open(output_filename, 'w') as f:
            f.write('index,prediction\n')

    for idx, row in test.iterrows():
        prediction = predictor(row['image_link'], row['group_id'], row['entity_name'])

        with open(output_filename, 'a') as f:
            f.write(f"{row['index']},{prediction}\n")
        

        if idx % 100 == 0:  # Print every 100th image
            print(f"Processed {idx + 1}/{len(test)} images.")
    
    print("Prediction process complete.")
