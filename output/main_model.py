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
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Function to clean and normalize the extracted text
def clean_extracted_text(text):
    # Remove special characters and normalize spaces
    text = re.sub(r'[^\d.,a-zA-Z\s]', '', text)  # Remove non-alphanumeric except periods/commas
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
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
    extracted = extracted.strip()
    if not extracted:
        return extracted  # or return target_value, based on use case

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
        config = r'--oem 1 --psm 11'  # Change psm to 6 for better handling of text blocks
        text = pytesseract.image_to_string(final_img, config=config)

        return text

    except Exception as e:
        print(f"Error processing image from URL {image_url}: {e}")
        return None


# Extended unit conversion dictionary
unit_conversion = {
    'gm': 'gram', 'g': 'gram', 'gram': 'gram',
    'mg': 'milligram', 'milligram': 'milligram',
    'kg': 'kilogram', 'kilogram': 'kilogram',
    'oz': 'ounce', 'ounce': 'ounce',
    'lb': 'pound', 'pound': 'pound',
    'ton': 'ton', 'ton': 'ton',
    'ml': 'millilitre', 'millilitre': 'millilitre', 'millimeter': 'millilitre',
    'l': 'litre', 'litre': 'litre', 'liter': 'litre',
    'fl oz': 'fluid ounce', 'fluid ounce': 'fluid ounce',
    'cm': 'centimetre', 'centimetre': 'centimetre', 'centimeter': 'centimetre',
    'm': 'metre', 'metre': 'metre', 'meter': 'metre',
    'ft': 'foot', 'foot': 'foot',
    'in': 'inch', 'inch': 'inch',
    'yd': 'yard', 'yard': 'yard',
    'mm': 'millimetre', 'millimetre': 'millimetre', 'millimeter': 'millimetre',
    'kV': 'kilovolt', 'kilovolt': 'kilovolt', 'volt': 'volt', 'v': 'volt', 'millivolt': 'volt',
    'kW': 'kilowatt', 'kilowatt': 'kilowatt', 'kwatt': 'kilowatt', 'watt': 'watt', 'w': 'watt',
    'cL': 'centilitre', 'centilitre': 'centilitre',
    'cu ft': 'cubic foot', 'cubic foot': 'cubic foot',
    'cu in': 'cubic inch', 'cubic inch': 'cubic inch',
    'cup': 'cup', 'pint': 'pint', 'quart': 'quart',
    'gal': 'gallon', 'gallon': 'gallon',
    'pint': 'pint', 'quart': 'quart'
}


# Function to format extracted text with enhanced language support
def format_extracted_text(text, feature):
    # Extend regex patterns to support more variations
    patterns = {
        'item_weight': re.compile(r'(\d+[\.,]?\d*)\s*(mg|g|gm|kilogram|gram|milligram|ounce|pound|ton)', re.IGNORECASE),
        'maximum_weight_recommendation': re.compile(
            r'(\d+[\.,]?\d*)\s*(mg|g|gm|kilogram|gram|milligram|ounce|pound|ton)', re.IGNORECASE),
        'voltage': re.compile(r'(\d+[\.,]?\d*)\s*(kV|kilovolt|V|volt|millivolt)', re.IGNORECASE),
        'wattage': re.compile(r'(\d+[\.,]?\d*)\s*(kW|kilowatt|W|watt)', re.IGNORECASE),
        'item_volume': re.compile(
            r'(\d+[\.,]?\d*)\s*(mL|ml|L|litre|fluid ounce|cup|pint|quart|gallon|centilitre|decilitre)', re.IGNORECASE),
        'width': re.compile(r'(\d+[\.,]?\d*)\s*(cm|centimetre|m|metre|mm|millimetre|ft|foot|in|inch|yd|yard)',
                            re.IGNORECASE),
        'height': re.compile(r'(\d+[\.,]?\d*)\s*(cm|centimetre|m|metre|mm|millimetre|ft|foot|in|inch|yd|yard)',
                             re.IGNORECASE),
        'depth': re.compile(r'(\d+[\.,]?\d*)\s*(cm|centimetre|m|metre|mm|millimetre|ft|foot|in|inch|yd|yard)',
                            re.IGNORECASE),
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


def process_images(csv_file, output_csv):
    """Process images from CSV and predict values."""
    df = pd.read_csv(csv_file)
    correct_predictions = 0
    total_predictions = 0
    results = []

    idx = 0
    for _, row in df.iterrows():
        image_url = row['image_link']
        feature = row['entity_name']
        target_value = row['entity_value']

        extracted_text = extract_text_from_image(image_url)

        if extracted_text:
            cleaned_text = clean_extracted_text(extracted_text)
            normalized_text = normalize_number_format(cleaned_text)
            # print(f"Cleaned text: {cleaned_text}")

            # Apply spelling correction
            corrected_text = correct_spelling(normalized_text)
            formatted_prediction = format_extracted_text(corrected_text, feature)

            # Fuzzy matching to compare predictions
            final_prediction = fuzzy_match(formatted_prediction, target_value)

            # Check if prediction is correct
            correct = final_prediction == target_value
            correct_predictions += correct
            total_predictions += 1

            # Append results to the list
            results.append({
                'image_link': image_url,
                'feature': feature,
                'predicted_value': final_prediction,
                'target_value': target_value,
                'correct': correct
            })

            print(f"{idx}] formatted text: {final_prediction}")
            idx+=1

            # Printing every 100th image processed
            if total_predictions % 100 == 0:
                print(f"Processed {total_predictions} images. Correct predictions: {correct_predictions}")

    # Save the results into a new CSV file
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False)

    print(f"Processing completed. Correct predictions: {correct_predictions} out of {total_predictions}.")


# Example usage

csv_file = "../dataset/Trial.csv"
output_csv = "../dataset/predictions.csv"
process_images(csv_file, output_csv)

