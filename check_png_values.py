#!/usr/bin/env python3
"""Check PNG pixel values to see if antialiasing is working"""

import sys
from pathlib import Path

import cv2
import numpy as np


def analyze_png(png_path):
    """Analyze PNG file to check for antialiasing values"""
    if not Path(png_path).exists():
        print(f"File not found: {png_path}")
        return

    # Read the image
    img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Could not read image: {png_path}")
        return

    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")

    # Get unique values
    if len(img.shape) == 3:
        # Color image - check each channel
        for i, channel_name in enumerate(["Blue", "Green", "Red"]):
            channel = img[:, :, i]
            unique_vals = np.unique(channel)
            print(f"\n{channel_name} channel unique values ({len(unique_vals)} total):")
            if len(unique_vals) <= 20:
                print(f"  {unique_vals}")
            else:
                print(f"  Min: {unique_vals.min()}, Max: {unique_vals.max()}")
                print(f"  First 10: {unique_vals[:10]}")
                print(f"  Last 10: {unique_vals[-10:]}")

            # Check for intermediate values (antialiasing)
            if img.dtype == np.uint8:
                intermediate = unique_vals[(unique_vals > 0) & (unique_vals < 255)]
            elif img.dtype == np.uint16:
                intermediate = unique_vals[(unique_vals > 0) & (unique_vals < 65535)]
            elif img.dtype == np.float32:
                intermediate = unique_vals[(unique_vals > 0.0) & (unique_vals < 1.0)]
            else:
                intermediate = unique_vals

            print(f"  Intermediate values (antialiasing): {len(intermediate)} found")
            if len(intermediate) > 0 and len(intermediate) <= 10:
                print(f"    {intermediate}")
    else:
        # Grayscale
        unique_vals = np.unique(img)
        print(f"\nGrayscale unique values ({len(unique_vals)} total):")
        if len(unique_vals) <= 20:
            print(f"  {unique_vals}")
        else:
            print(f"  Min: {unique_vals.min()}, Max: {unique_vals.max()}")
            print(f"  First 10: {unique_vals[:10]}")
            print(f"  Last 10: {unique_vals[-10:]}")

    # Sample a small region around potential edges
    h, w = img.shape[:2]
    center_y, center_x = h // 2, w // 2
    sample_region = img[center_y - 50 : center_y + 50, center_x - 50 : center_x + 50]

    if len(sample_region.shape) == 3:
        sample_unique = np.unique(sample_region[:, :, 0])  # Just check one channel
    else:
        sample_unique = np.unique(sample_region)

    print(f"\nSample region ({sample_region.shape}) unique values:")
    print(f"  {sample_unique}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_png_values.py <path_to_png>")
        # Check if there are any PNG files in output directory
        output_dir = Path("output")
        if output_dir.exists():
            png_files = list(output_dir.glob("*.png"))
            if png_files:
                print("\nFound PNG files in output directory:")
                for png_file in png_files[:5]:  # Show first 5
                    print(f"  {png_file}")
                if len(png_files) > 0:
                    print(f"\nAnalyzing most recent: {png_files[0]}")
                    analyze_png(png_files[0])
        sys.exit(1)

    png_path = sys.argv[1]
    analyze_png(png_path)
    png_path = sys.argv[1]
    analyze_png(png_path)
