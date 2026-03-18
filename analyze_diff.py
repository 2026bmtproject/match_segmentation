import cv2
import csv
import sys
import numpy as np
import glob
import os

# 尋找目錄下的 .mp4 檔案
video_files = glob.glob("*.mp4")
DEFAULT_VIDEO = video_files[0] if video_files else 'test2.mp4'

VIDEO_PATH = "SHEALEX.mp4" # Replace with your video
OUTPUT_CSV = 'SHEALEX.csv'
    
if len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Unable to open video file {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {VIDEO_PATH}...")
    print(f"Total frames: {total_frames}, FPS: {fps}")

    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading first frame")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # List to store results: [Frame Number, Time (sec), Diff Score]
    results = []
    
    # Initialize with first frame having 0 diff
    results.append([0, 0.0, 0])

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference
        diff = cv2.absdiff(prev_gray, gray)
        
        # Metric: Sum of absolute differences (simple movement metric)
        diff_score = np.sum(diff)
        
        # Normalize score slightly? No, raw score is fine.
        # Maybe average per pixel might be more intuitive?
        # mean_diff = np.mean(diff) 
        
        time_sec = frame_count / fps
        
        results.append([frame_count, round(time_sec, 3), diff_score])
        
        prev_gray = gray
        
        # Progress indicator
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...", end='\r')

    cap.release()
    print(f"\nFinished processing {frame_count} frames.")
    
    # Write to CSV
    try:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Time_Sec', 'Difference_Score'])
            writer.writerows(results)
        print(f"Results saved to {OUTPUT_CSV}")
        
        # Optional: Print a small table preview to console
        print("\n--- Top 10 Frames with Highest Difference ---")
        # Sort by score descending
        sorted_results = sorted(results, key=lambda x: x[2], reverse=True)[:10]
        print(f"{'Frame':<10} {'Time(s)':<10} {'Score':<15}")
        print("-" * 35)
        for row in sorted_results:
            print(f"{row[0]:<10} {row[1]:<10} {row[2]:<15}")
            
    except IOError as e:
        print(f"Error writing file: {e}")

if __name__ == "__main__":
    main()
