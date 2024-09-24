import os
import random
import pandas as pd
from Processor import process_image


def predictor(image_link, category_id, entity_name):
    '''
    Call your model/approach here
    '''
    #TODO

    result = process_image(image_link, entity_name)

    return result

if __name__ == "__main__":

    DATASET_FOLDER = r'C:\Users\admin\Desktop\Amazon_ML\66e31d6ee96cd_student_resource_3\student_resource 3\dataset'

    # Read the test CSV file
    test = pd.read_csv('suraj.csv')
    
    
    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    
    # output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv('test_out.csv', index=False)

# C:\Users\admin\Desktop\Amazon_ML\66e31d6ee96cd_student_resource_3\student_resource 3\dataset\test.csv