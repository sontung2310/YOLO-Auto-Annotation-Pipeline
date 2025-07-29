import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data")
parser.add_argument("--annotation_dir", type=str, default="auto_annotate_labels")
parser.add_argument("--output_dir", type=str, default="yolo_annotations")


args = parser.parse_args()

def parse_mask_annotations(file_path):
    """
    Parse the mask annotation file and extract polygon coordinates
    """
    masks = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Split the line into values
            values = line.split()
            if len(values) < 3:  # Need at least class_id and one coordinate pair
                continue
                
            class_id = int(values[0])
            coordinates = []
            
            # Extract coordinate pairs (x, y)
            for i in range(1, len(values), 2):
                if i + 1 < len(values):
                    x = float(values[i])
                    y = float(values[i + 1])
                    coordinates.append([x, y])
            
            if len(coordinates) >= 3:  # Need at least 3 points for a polygon
                masks.append({
                    'class_id': class_id,
                    'coordinates': np.array(coordinates)
                })
    
    return masks

def mask_to_bounding_box(coordinates):
    """
    Convert polygon coordinates to bounding box (x_center, y_center, width, height)
    All values are normalized (0-1)
    """
    coords = np.array(coordinates)
    
    # Find bounding box
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    
    # Calculate center and dimensions
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return [x_center, y_center, width, height]

def convert_to_yolo_format(masks):
    """
    Convert mask annotations to YOLO format
    YOLO format: class_id x_center y_center width height
    """
    yolo_annotations = []
    
    for mask in masks:
        bbox = mask_to_bounding_box(mask['coordinates'])
        yolo_line = f"{mask['class_id']} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
        yolo_annotations.append(yolo_line)
    
    return yolo_annotations

def save_yolo_annotations(annotations, output_file):
    """
    Save YOLO annotations to file
    """
    with open(output_file, 'w') as f:
        for annotation in annotations:
            f.write(annotation + '\n')

def visualize_annotations_on_image(masks, yolo_annotations, image_path):
    """
    Visualize both original masks and YOLO bounding boxes on the image
    """
    # Load the image
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Colors for different classes
    colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple', 'brown', 'pink']
    
    # Plot 1: Original masks
    ax1.set_title('Original Mask Annotations')
    ax1.imshow(img_array)
    
    for i, mask in enumerate(masks):
        coords = mask['coordinates']
        # Convert normalized coordinates to pixel coordinates
        pixel_coords = (coords * np.array([width, height])).astype(np.int32)
        
        # Plot polygon
        color = colors[i % len(colors)]
        ax1.plot(pixel_coords[:, 0], pixel_coords[:, 1], 
                color=color, linewidth=2, 
                label=f'Mask {i+1} (Class {mask["class_id"]})')
        ax1.fill(pixel_coords[:, 0], pixel_coords[:, 1], 
                color=color, alpha=0.3)
    
    ax1.legend()
    ax1.axis('off')
    
    # Plot 2: YOLO bounding boxes
    ax2.set_title('YOLO Detection Annotations')
    ax2.imshow(img_array)
    
    for i, yolo_ann in enumerate(yolo_annotations):
        parts = yolo_ann.split()
        class_id = int(parts[0])
        x_center, y_center, w, h = map(float, parts[1:])
        
        # Convert normalized coordinates to pixel coordinates
        x_center_px = x_center * width
        y_center_px = y_center * height
        w_px = w * width
        h_px = h * height
        
        # Calculate top-left corner
        x1 = x_center_px - w_px / 2
        y1 = y_center_px - h_px / 2
        
        # Draw rectangle
        color = colors[i % len(colors)]
        rect = Rectangle((x1, y1), w_px, h_px, 
                        linewidth=2, edgecolor=color, 
                        facecolor='none', 
                        label=f'Box {i+1} (Class {class_id})')
        ax2.add_patch(rect)
        
        # Add class label
        ax2.text(x1, y1-5, f'Class {class_id}', 
                color=color, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
    
    ax2.legend()
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def main(args):
    
    image_file_lst = glob.glob(args.data + "/*")
    os.makedirs(args.output_dir, exist_ok=True)
    for image_file in image_file_lst:
        image_name = image_file.split("/")[-1]
        annotation_file = f"{args.annotation_dir}/{image_name.split('.')[0]}.txt"
        output_file = f"{args.output_dir}/{image_name.split('.')[0]}.txt"
        
        if not os.path.exists(annotation_file):
            print(f"Annotation file not found: {annotation_file}")
            continue
        
        try:
            # Parse the mask annotations
            print("Parsing mask annotations...")
            masks = parse_mask_annotations(annotation_file)
            print(f"Found {len(masks)} masks")
            
            if len(masks) == 0:
                print("No valid masks found in the file!")
                return
            
            # Display information about each mask
            for i, mask in enumerate(masks):
                print(f"Mask {i+1}: Class {mask['class_id']}, {len(mask['coordinates'])} points")
            
            # Convert to YOLO format
            print("\nConverting to YOLO format...")
            yolo_annotations = convert_to_yolo_format(masks)
            
            # Display YOLO annotations
            print("YOLO annotations:")
            for i, ann in enumerate(yolo_annotations):
                print(f"  {i+1}: {ann}")
            
            # Save YOLO annotations
            save_yolo_annotations(yolo_annotations, output_file)
            print(f"\nSaved YOLO annotations to {output_file}")
            
            # Visualize annotations
            print("\nVisualizing annotations...")
            fig = visualize_annotations_on_image(masks, yolo_annotations, image_file)
            
            if fig is not None:
                # Save the visualization
                image_name_without_extension = image_name.split(".")[0]
                fig.savefig(f'annotation_comparison_{image_name_without_extension}.png', dpi=300, bbox_inches='tight')
                print(f"Saved visualization to annotation_comparison_{image_name_without_extension}.png")
            
        except FileNotFoundError as e:
            print(f"Error: File not found! {e}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main(args) 