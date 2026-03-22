#!/usr/bin/env python3
"""
Script to copy PNG images from CelebAHQ dataset based on JPG filenames in an input folder.

Usage:
    python copy_selected_images.py --input_dir <path_to_jpg_folder> --output_dir <path_to_destination>
    
Example:
    python copy_selected_images.py --input_dir ./my_selected_images --output_dir ./copied_images
"""

import argparse
import shutil
from pathlib import Path
import sys


def copy_images(input_dir, output_dir, source_dir="/media/embedded/Datasets/Public/CelebAhq256/validation/"):
    """
    Copy PNG images from source directory based on JPG filenames in input directory.
    
    Args:
        input_dir: Directory containing JPG files to use as reference
        output_dir: Destination directory where PNG files will be copied
        source_dir: Source directory containing PNG files (CelebAHQ validation folder)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    source_path = Path(source_dir)
    
    # Validate input directory
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return False
    
    if not input_path.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        return False
    
    # Validate source directory
    if not source_path.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return False
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all JPG files in input directory
    jpg_files = list(input_path.glob("*.jpg"))
    
    if not jpg_files:
        print(f"No JPG files found in {input_dir}")
        return False
    
    print(f"Found {len(jpg_files)} JPG files in input directory")
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    copied_count = 0
    failed_count = 0
    
    for jpg_file in sorted(jpg_files):
        # Get filename without extension
        base_name = jpg_file.stem
        
        # Construct PNG filename
        png_filename = f"{base_name}.png"
        png_source = source_path / png_filename
        
        # Check if PNG exists in source directory
        if not png_source.exists():
            print(f"✗ Not found: {png_filename}")
            failed_count += 1
            continue
        
        # Copy PNG to output directory
        png_destination = output_path / png_filename
        try:
            shutil.copy2(png_source, png_destination)
            print(f"✓ Copied: {png_filename}")
            copied_count += 1
        except Exception as e:
            print(f"✗ Error copying {png_filename}: {e}")
            failed_count += 1
    
    print("-" * 60)
    print(f"Summary: {copied_count} files copied successfully, {failed_count} failed")
    
    return failed_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Copy PNG images from CelebAHQ based on JPG filenames in an input folder"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing JPG files to use as reference"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Destination directory where PNG files will be copied"
    )
    parser.add_argument(
        "--source_dir",
        default="/media/embedded/Datasets/Public/CelebAhq256/validation/",
        help="Source directory containing PNG files (default: CelebAHQ validation folder)"
    )
    
    args = parser.parse_args()
    
    success = copy_images(args.input_dir, args.output_dir, args.source_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
