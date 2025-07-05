# Amazon-ML: Image Entity Value Extraction

## Overview

Amazon-ML is a machine learning project designed to extract product attribute values (e.g., weight, dimensions, voltage) from product images using advanced OCR and image processing. This is particularly useful for automating structured data collection in e-commerce and related fields.

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset Description](#dataset-description)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Output Format](#output-format)
- [Evaluation Criteria](#evaluation-criteria)
- [Appendix: Allowed Units](#appendix-allowed-units)
- [License](#license)

## Project Structure

```
output/
│
├── main.py                       # Main file for batch prediction on test data
├── main_model.py                 # Full image-to-entity pipeline (training/evaluation)
├── Processor.py                  # Core image processing and extraction logic
├── main_processor_explanation.md # Human-readable explanation of the process
├── Vanguards_ML.zip              # (Optional) Zipped model or resources
├── test_out.csv                  # Example output file
dataset/
├── train.csv                     # Labeled training data
├── test.csv                      # Test data (no labels)
├── Trial.csv                     # Example trial data
├── predictions.csv               # Sample predictions
```

## Dataset Description

Each data entry contains:

- **index**: Unique identifier
- **image_link**: URL to the product image
- **group_id**: Product category code
- **entity_name**: Attribute to extract (e.g., "item_weight")
- **entity_value**: (Only in training) The ground truth value

### File Descriptions

- `train.csv`: Training samples with ground truth.
- `test.csv`: Test samples for prediction.
- `Trial.csv`, `predictions.csv`: Extra samples for experimentation.

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/PranavG1203/Amazon-ML.git
   cd Amazon-ML
   ```

2. **Install requirements**

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data**

   - Place datasets in the `dataset/` directory.

4. **Run Predictions**
   - For batch processing, use:
     ```bash
     python output/main.py
     ```
   - For full training/evaluation (with accuracy reporting):
     ```bash
     python output/main_model.py
     ```

## Usage

- `main.py` loads the test CSV, predicts values for each image/entity, and writes results to `test_out_*.csv`.
- `Processor.py` contains all core logic for image downloading, preprocessing, OCR, and feature extraction.
- See `main_processor_explanation.md` in the `output/` directory for a plain-language walkthrough of the pipeline.

## Output Format

Predictions must be a CSV with:

- **index**: Sample ID (matches test set)
- **prediction**: String in `"x unit"` format (e.g., `34 gram`). Use only allowed units. Return `""` if not found.

Example:

```
index,prediction
1,34 gram
2,12 inch
3,
```

## Evaluation Criteria

F1 score is used for evaluation (see the hackathon statement for details). Only valid units and correct formatting are accepted.

## Appendix: Allowed Units

Only these units are valid for each entity:

```
entity_unit_map = {
  "width": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
  "depth": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
  "height": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
  "item_weight": {"milligram", "kilogram", "microgram", "gram", "ounce", "ton", "pound"},
  "maximum_weight_recommendation": {"milligram", "kilogram", "microgram", "gram", "ounce", "ton", "pound"},
  "voltage": {"millivolt", "kilovolt", "volt"},
  "wattage": {"kilowatt", "watt"},
  "item_volume": {"cubic foot", "microlitre", "cup", "fluid ounce", "centilitre", "imperial gallon",
    "pint", "decilitre", "litre", "millilitre", "quart", "cubic inch", "gallon"}
}
```
