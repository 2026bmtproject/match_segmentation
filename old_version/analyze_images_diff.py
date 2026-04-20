import cv2
import csv
import numpy as np
import glob
import os
from pathlib import Path
import sys

def load_images(image_dir):
    """Load all images from directory and return sorted list of (filename, image) tuples"""
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png"))) + \
                  sorted(glob.glob(os.path.join(image_dir, "*.jpg"))) + \
                  sorted(glob.glob(os.path.join(image_dir, "*.jpeg")))
    
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            filename = os.path.basename(img_path)
            images.append((filename, img))
        else:
            print(f"Warning: Failed to load {img_path}")
    
    return images

def calculate_frame_diff(img1, img2, method='sad'):
    """
    Calculate frame difference between two images using various methods.
    
    Methods:
    - sad: Sum of Absolute Differences (default)
    - mad: Mean Absolute Difference
    - mse: Mean Squared Error
    - rmse: Root Mean Squared Error
    - histogram: Histogram Chi-Square Distance
    - ssim: Structural Similarity Index
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    if method == 'sad':
        # Sum of Absolute Differences
        diff = cv2.absdiff(gray1, gray2)
        return np.sum(diff)
    
    elif method == 'mad':
        # Mean Absolute Difference
        diff = cv2.absdiff(gray1, gray2)
        return np.mean(diff)
    
    elif method == 'mse':
        # Mean Squared Error
        diff = cv2.absdiff(gray1.astype(np.float32), gray2.astype(np.float32))
        return np.mean(diff ** 2)
    
    elif method == 'rmse':
        # Root Mean Squared Error
        diff = cv2.absdiff(gray1.astype(np.float32), gray2.astype(np.float32))
        return np.sqrt(np.mean(diff ** 2))
    
    elif method == 'histogram':
        # Histogram Chi-Square Distance
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    
    elif method == 'ssim':
        # Structural Similarity Index (需要scikit-image)
        try:
            from skimage.metrics import structural_similarity as ssim
            # SSIM 返回0-1，1表示完全相同，所以用1-ssim表示差異
            return 1.0 - ssim(gray1, gray2)
        except ImportError:
            print("Warning: scikit-image not found. Using SAD instead.")
            diff = cv2.absdiff(gray1, gray2)
            return np.sum(diff)
    
    else:
        raise ValueError(f"Unknown method: {method}")

def main():
    image_dir = "pp"
    output_csv = "images_diff_results.csv"
    method = "sad"  # Default method
    
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_csv = sys.argv[2]
    if len(sys.argv) > 3:
        method = sys.argv[3]
    
    # Load images
    print(f"Loading images from '{image_dir}'...")
    images = load_images(image_dir)
    
    if len(images) == 0:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(images)} images")
    print(f"Using diff method: {method}")
    
    # Calculate average difference for each image to all others
    results = []
    all_diff_scores = []
    
    print("Calculating average frame diff for each image...")
    for i in range(len(images)):
        img1_name, img1 = images[i]
        diff_scores_for_image = []
        
        for j in range(len(images)):
            if i != j:
                img2_name, img2 = images[j]
                diff_score = calculate_frame_diff(img1, img2, method)
                diff_scores_for_image.append(diff_score)
        
        avg_diff_for_image = np.mean(diff_scores_for_image) if diff_scores_for_image else 0
        results.append([img1_name, avg_diff_for_image])
        all_diff_scores.extend(diff_scores_for_image)
        
        print(f"Processed {i+1}/{len(images)} images...", end='\r')
    
    # Calculate overall statistics
    overall_avg_diff = np.mean(all_diff_scores) if all_diff_scores else 0
    max_avg_diff = np.max([r[1] for r in results])
    min_avg_diff = np.min([r[1] for r in results])
    std_avg_diff = np.std([r[1] for r in results])
    
    print(f"\nFinished processing {len(images)} images.")
    print(f"\n--- Overall Statistics ---")
    print(f"Overall Average Frame Diff: {overall_avg_diff:.2f}")
    print(f"Max Average Frame Diff: {max_avg_diff:.2f}")
    print(f"Min Average Frame Diff: {min_avg_diff:.2f}")
    print(f"Std Dev of Average Frame Diff: {std_avg_diff:.2f}")
    
    # Write to CSV
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Average_Difference_Score'])
            writer.writerows(results)
        print(f"\nResults saved to {output_csv}")
        
        # Print sorted results (highest average diff first)
        print("\n--- Image Average Differences (sorted by highest) ---")
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        print(f"{'Image':<20} {'Avg Diff Score':<15}")
        print("-" * 35)
        for row in sorted_results:
            print(f"{row[0]:<20} {row[1]:<15.2f}")
        
    except IOError as e:
        print(f"Error writing file: {e}")

if __name__ == "__main__":
    main()
