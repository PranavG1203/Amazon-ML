import pandas as pd
import pytesseract
from PIL import Image
import requests
from io import BytesIO
import re
import cv2
import numpy as np
from PIL import ImageEnhance, ImageFilter
from fuzzywuzzy import fuzz, process
from textblob import TextBlob
import  os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Unit conversion dictionary for standardizing unit names
unit_conversion = {
    'gm': 'gram',
    'g': 'gram',
    'mg': 'milligram',
    'kg': 'kilogram',
    'oz': 'ounce',
    'lb': 'pound',
    'ton': 'ton',
    'ml': 'millilitre',
    'l': 'litre',
    'fl oz': 'fluid ounce',
    'cm': 'centimetre',
    'm': 'metre',
    'ft': 'foot',
    'in': 'inch',
    'yd': 'yard',
    'mm': 'millimetre',
    'kV': 'kilovolt',
    'V': 'volt',
    'kW': 'kilowatt',
    'W': 'watt',
    'cL': 'centilitre',
    'cu ft': 'cubic foot',
    'cu in': 'cubic inch',
    'cup': 'cup',
    'dL': 'decilitre',
    'gal': 'gallon',
    'pint': 'pint',
    'quart': 'quart'
}

# Function to clean and normalize the extracted text
def clean_extracted_text(text):
    # Remove special characters and normalize spaces
    text = re.sub(r'[^\d.,a-zA-Z\s]', '', text)  # Remove non-alphanumeric except periods/commas
    text = re.sub(r'\s+', ' ', text).strip()     # Normalize spaces
    return text

# Function to normalize the number format (replace commas with periods for decimals)
def normalize_number_format(text):
    return re.sub(r'(\d+)[,](\d+)', r'\1.\2', text)

# Spelling correction using TextBlob
def correct_spelling(text):
    corrected_text = str(TextBlob(text).correct())
    return corrected_text

# Fuzzy matching function to compare extracted text with the target
def fuzzy_match(extracted, target_value):
    # Calculate fuzzy matching score
    match, score = process.extractOne(extracted, [target_value])
    return match if score > 80 else extracted  # Adjust threshold as needed

# Function to apply all 8 steps for preprocessing images
def extract_text_from_image(image_url):
    try:
        # Step 1: Fetch the image from the URL
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # Convert image to OpenCV format (numpy array)
        img = np.array(img)

        # Step 2: Convert image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Step 3: Apply thresholding to binarize the image
        _, binary_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)

        # Step 4: Resize image using INTER_LANCZOS4
        height, width = binary_img.shape
        scale_percent = 150  # Scaling up by 150%
        new_width = int(width * scale_percent / 100)
        new_height = int(height * scale_percent / 100)
        resized_img = cv2.resize(binary_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        # Step 5: Apply contrast enhancement using CLAHE (adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(resized_img)

        # Step 6: Apply noise removal (Gaussian blur)
        noise_removed_img = cv2.GaussianBlur(enhanced_img, (5, 5), 0)

        # Step 7: Sharpen the image
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened_img = cv2.filter2D(noise_removed_img, -1, kernel)

        # Step 8: Final enhancement using ImageEnhance (contrast)
        final_pil_img = Image.fromarray(sharpened_img)
        enhancer = ImageEnhance.Contrast(final_pil_img)
        final_img = enhancer.enhance(2)

        # Extract text from the final preprocessed image
        text = pytesseract.image_to_string(final_img)

        return text

    except Exception as e:
        print(f"Error processing image from URL {image_url}: {e}")
        return None

# Function to format extracted text using regular expressions
def format_extracted_text(text, feature):
    # Regular expressions for each feature
    patterns = {
        'item_weight': re.compile(r'(\d{1,4}[\.,]?\d{0,2})\s*(mg|g|gm|kg|oz|lb|ton)', re.IGNORECASE),
        'maximum_weight_recommendation': re.compile(r'(\d{1,4}[\.,]?\d{0,2})\s*(mg|g|gm|kg|oz|lb|ton)', re.IGNORECASE),
        'voltage': re.compile(r'(\d{1,4}[\.,]?\d{0,2})\s*(kV|V|millivolt)', re.IGNORECASE),
        'wattage': re.compile(r'(\d{1,4}[\.,]?\d{0,2})\s*(kW|W)', re.IGNORECASE),
        'item_volume': re.compile(r'(\d{1,4}[\.,]?\d{0,2})\s*(mL|ml|L|fl oz|cup|pint|quart|gal|imp gal)', re.IGNORECASE),
        'width': re.compile(r'(\d{1,4}[\.,]?\d{0,2})\s*(cm|m|mm|ft|in|yd)', re.IGNORECASE),
        'height': re.compile(r'(\d{1,4}[\.,]?\d{0,2})\s*(cm|m|mm|ft|in|yd)', re.IGNORECASE),
        'depth': re.compile(r'(\d{1,4}[\.,]?\d{0,2})\s*(cm|m|mm|ft|in|yd)', re.IGNORECASE),
    }

    if feature not in patterns:
        return "Feature not supported"

    pattern = patterns[feature]
    match = pattern.search(text)

    if match:
        value = match.group(1).replace(',', '.')  # Ensure consistent decimal format
        unit = match.group(2).lower()
        standardized_unit = unit_conversion.get(unit, unit)
        value = round(float(value), 2)  # Round to 2 decimal places for consistency
        return f"{value} {standardized_unit}"

    return ""

# Function to compare extracted text with target and calculate accuracy
def process_images(image_link, entity_name):

    results = []
    i = 1

    image_url = image_link
    feature = entity_name

    extracted_text = extract_text_from_image(image_url)

    # Check if extracted_text is None or empty, skip if so
    if extracted_text is None or extracted_text.strip() == "":
        print("No text found in image or error in processing.")
        formatted_text = ""
    else:
        # Clean and normalize the text
        extracted_text = clean_extracted_text(extracted_text)
        extracted_text = normalize_number_format(extracted_text)
        extracted_text = correct_spelling(extracted_text)

        formatted_text = format_extracted_text(extracted_text, feature)


    results.append({'image_link': image_url, 'prediction': formatted_text})


    return formatted_text

def predictor(image_link, category_id, entity_name):
    '''
    Call your model/approach here
    '''
    # TODO

    result = process_images(image_link, entity_name)

    return result

if __name__ == "__main__":
    # DATASET_FOLDER = r'C:\Users\admin\Desktop\Amazon_ML\66e31d6ee96cd_student_resource_3\student_resource 3\dataset'

    # Read the test CSV file
    test = pd.read_csv('test_45444.csv')

    output_filename = 'test_out_45444.csv'

    # Write the header once if the file does not exist
    if not os.path.exists(output_filename):
        with open(output_filename, 'w') as f:
            f.write('index,prediction\n')

    # Loop over each row and append the predictions to the CSV file incrementally
    for idx, row in test.iterrows():
        # Get the prediction for the current row
        row_index = row['index']

        # Get the prediction for the current row
        prediction = predictor(row['image_link'], row['group_id'], row['entity_name'])

        # Append the result to the CSV file
        with open(output_filename, 'a') as f:
            f.write(f"{row_index},{prediction}\n")

        # Print progress for every 100th row
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(test)} rows.")

        print("Prediction process complete.")