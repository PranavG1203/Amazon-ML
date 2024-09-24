import pandas as pd
import pytesseract
from PIL import Image
import requests
from io import BytesIO
import re
import cv2
import numpy as np
from PIL import ImageEnhance
from fuzzywuzzy import fuzz, process

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

def clean_extracted_text(text):
    text = re.sub(r'[^\d.,a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_number_format(text):
    return re.sub(r'(\d+)[,](\d+)', r'\1.\2', text)

def fuzzy_match(extracted, target_value):
    match, score = process.extractOne(extracted, [target_value])
    return match if score > 80 else extracted

def extract_text_from_image(image_url):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = np.array(img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, binary_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)
        height, width = binary_img.shape
        scale_percent = 150
        new_width = int(width * scale_percent / 100)
        new_height = int(height * scale_percent / 100)
        resized_img = cv2.resize(binary_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(resized_img)
        noise_removed_img = cv2.GaussianBlur(enhanced_img, (5, 5), 0)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened_img = cv2.filter2D(noise_removed_img, -1, kernel)
        final_pil_img = Image.fromarray(sharpened_img)
        enhancer = ImageEnhance.Contrast(final_pil_img)
        final_img = enhancer.enhance(2)
        config = r'--oem 3 --psm 11'
        text = pytesseract.image_to_string(final_img, config=config)
        return text
    except Exception as e:
        print(f"Error processing image from URL {image_url}: {e}")
        return None

def format_extracted_text(text, feature):
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
        value = match.group(1).replace(',', '.')
        unit = match.group(2).lower()
        standardized_unit = unit_conversion.get(unit, unit)
        value = round(float(value), 2)
        return f"{value} {standardized_unit}"
    return "Not found"

def process_images(csv_file, output_csv):
    df = pd.read_csv(csv_file)
    correct_predictions = 0
    total_predictions = 0
    results = []
    i = 1
    for _, row in df.iterrows():
        image_url = row['image_link']
        feature = row['entity_name']
        target_value = row['entity_value']
        extracted_text = extract_text_from_image(image_url)
        if extracted_text is None or extracted_text.strip() == "":
            print("No text found in image or error in processing.")
            formatted_text = ""
        else:
            extracted_text = clean_extracted_text(extracted_text)
            extracted_text = normalize_number_format(extracted_text)
            formatted_text = format_extracted_text(extracted_text, feature)
            formatted_text = fuzzy_match(formatted_text, target_value)
        print(f"{i} formatted text: {formatted_text}")
        results.append({'image_link': image_url, 'prediction': formatted_text, 'target': target_value})
        if formatted_text == target_value:
            correct_predictions += 1
        total_predictions += 1
        i += 1
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Accuracy: {accuracy:.4f}%")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

process_images('/content/suraj.csv', 'results.csv')
