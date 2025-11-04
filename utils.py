from PIL import Image
from ultralytics.engine.results import Results


def crop_cells(image: Image.Image, results: Results, class_name: str, padding: int = 20, output_size: int = None) -> list[Image.Image]:
    """
    Crops the input image around each detected White Blood Cell (WBC).

    """
    from healthbox.blood_stain.registry import CLASS_IDS
    cell_class_id = CLASS_IDS[class_name]
    cell_crops = []


    img_width, img_height = image.size

    if results is None or results.boxes is None:
        return cell_crops

    for box in results.boxes.data:
        # The box format is [x1, y1, x2, y2, confidence, class_id]
        x1, y1, x2, y2, _, class_id = box
        if class_id == cell_class_id:
            # Calculate new coordinates with padding
            new_x1 = int(x1) - padding
            new_y1 = int(y1) - padding
            new_x2 = int(x2) + padding
            new_y2 = int(y2) + padding

            # Clamp coordinates to be within the image boundaries to avoid errors
            new_x1 = max(0, new_x1)
            new_y1 = max(0, new_y1)
            new_x2 = min(img_width, new_x2)
            new_y2 = min(img_height, new_y2)

            # Crop the image using the new, padded coordinates
            cropped_image = image.crop((new_x1, new_y1, new_x2, new_y2))

            if output_size:
                # Use a high-quality downsampling filter
                cropped_image = cropped_image.resize((output_size, output_size), Image.Resampling.LANCZOS)

            cell_crops.append(cropped_image)

    return cell_crops