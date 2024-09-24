import pandas as pd
import json

def convert_csv_to_coco(csv_file, output_json):
    df = pd.read_csv(csv_file)

    images = []
    annotations = []
    categories = []
    image_id = 1
    annotation_id = 1
    category_map = {}  # Mapping for categories

    for _, row in df.iterrows():
        # Add image information
        image_info = {
            "id": image_id,
            "file_name": row["image_link"],  # Use full URL
            "width": 640,  # Placeholder value, update with actual width if available
            "height": 480  # Placeholder value, update with actual height if available
        }
        images.append(image_info)

        # Check if category already exists
        entity_name = row["entity_name"]
        if entity_name not in category_map:
            category_id = len(category_map) + 1
            category_map[entity_name] = category_id
            categories.append({"id": category_id, "name": entity_name})

        # Create a placeholder bounding box (x, y, width, height)
        # You need actual bounding box values here, possibly from text detection
        bbox = [100, 100, 200, 50]  # Example bbox, replace with actual data

        annotation_info = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_map[entity_name],
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0
        }
        annotations.append(annotation_info)

        image_id += 1
        annotation_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # Save to JSON file
    with open(output_json, 'w') as f:
        json.dump(coco_format, f)

# Example usage
convert_csv_to_coco(r"C:\Users\admin\Desktop\Amazon_ML\66e31d6ee96cd_student_resource_3\student_resource 3\example\data\train.csv", "output_coco.json")
