#!/usr/bin/env python3
"""
VRT Path Fixer Script

This script reads a VRT file, fixes file paths by replacing Windows-style paths
with macOS/Unix-style paths, and writes the corrected VRT to a new file.

Usage:
    python fix_vrt_paths.py input.vrt output.vrt --old-root "F:/ElevationMaps" --new-root "/Volumes/Vault/ElevationMaps"

    # Or use auto-detection (looks for common Windows drive letters)
    python fix_vrt_paths.py input.vrt output.vrt --auto-fix
"""

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def fix_windows_path_to_unix(path_str, old_root=None, new_root=None):
    """
    Convert Windows-style path to Unix-style path

    Args:
        path_str: The original path string
        old_root: The Windows root to replace (e.g., "F:/ElevationMaps")
        new_root: The Unix root to replace with (e.g., "/Volumes/Vault/ElevationMaps")

    Returns:
        Fixed path string
    """
    if not path_str:
        return path_str

    # If specific root replacements are provided
    if old_root and new_root:
        # Normalize paths for comparison
        old_root_norm = old_root.replace("\\", "/").rstrip("/")
        path_norm = path_str.replace("\\", "/")

        if path_norm.startswith(old_root_norm):
            # Replace the old root with new root
            relative_path = path_norm[len(old_root_norm) :].lstrip("/")
            return f"{new_root.rstrip('/')}/{relative_path}"

    # Auto-fix common Windows patterns
    # Replace Windows drive letters (C:, D:, F:, etc.) with Unix equivalents
    # This is a simple heuristic - adjust patterns as needed

    # Convert backslashes to forward slashes (most important for relative paths)
    fixed_path = path_str.replace("\\", "/")

    # For absolute paths, handle drive letters
    if ":" in fixed_path and len(fixed_path) > 1 and fixed_path[1] == ":":
        # Common drive letter mappings (customize as needed)
        drive_mappings = {
            "F:/ElevationMaps": "/Volumes/Vault/ElevationMaps",
            "C:/ElevationMaps": "/Volumes/Vault/ElevationMaps",
            "D:/ElevationMaps": "/Volumes/Vault/ElevationMaps",
            "E:/ElevationMaps": "/Volumes/Vault/ElevationMaps",
        }

        for windows_path, unix_path in drive_mappings.items():
            if fixed_path.startswith(windows_path):
                return fixed_path.replace(windows_path, unix_path, 1)

        # If no specific mapping found, try generic drive letter replacement
        # Look for pattern like "X:/..." and replace with "/Volumes/Vault/..."
        drive_pattern = re.match(r"^[A-Z]:(/.+)", fixed_path)
        if drive_pattern:
            relative_path = drive_pattern.group(1).lstrip("/")
            return f"/Volumes/Vault/{relative_path}"

    # For relative paths, just return the path with forward slashes
    # (most VRT files use relative paths with relativeToVRT="1")
    return fixed_path


def fix_vrt_file(
    input_vrt_path, output_vrt_path, old_root=None, new_root=None, dry_run=False
):
    """
    Fix paths in a VRT file

    Args:
        input_vrt_path: Path to input VRT file
        output_vrt_path: Path to output VRT file
        old_root: Windows root path to replace
        new_root: Unix root path to replace with
        dry_run: If True, only print what would be changed without writing output

    Returns:
        Number of paths that were fixed
    """

    try:
        # Parse the VRT XML file
        tree = ET.parse(input_vrt_path)
        root = tree.getroot()

        fixes_made = 0

        # Find all elements that might contain file paths
        # VRT files typically have <SourceFilename> elements
        for elem in root.iter():
            if elem.tag == "SourceFilename" and elem.text:
                original_path = elem.text.strip()
                fixed_path = fix_windows_path_to_unix(original_path, old_root, new_root)

                if fixed_path != original_path:
                    # Check if this is a relative path
                    is_relative = elem.get("relativeToVRT") == "1"
                    path_type = "relative" if is_relative else "absolute"

                    if dry_run:
                        print(
                            f"Would fix ({path_type}): {original_path} -> {fixed_path}"
                        )
                    else:
                        elem.text = fixed_path
                        print(f"Fixed ({path_type}): {original_path} -> {fixed_path}")
                    fixes_made += 1

            # Also check for other potential path attributes
            for attr_name in ["source", "href", "relativeToVRT"]:
                if attr_name in elem.attrib:
                    original_path = elem.attrib[attr_name]
                    fixed_path = fix_windows_path_to_unix(
                        original_path, old_root, new_root
                    )

                    if fixed_path != original_path:
                        if dry_run:
                            print(
                                f"Would fix attribute {attr_name}: {original_path} -> {fixed_path}"
                            )
                        else:
                            elem.attrib[attr_name] = fixed_path
                            print(
                                f"Fixed attribute {attr_name}: {original_path} -> {fixed_path}"
                            )
                        fixes_made += 1

        if not dry_run and fixes_made > 0:
            # Write the fixed VRT file
            tree.write(output_vrt_path, encoding="utf-8", xml_declaration=True)
            print(f"\nWrote fixed VRT file to: {output_vrt_path}")
        elif dry_run:
            print(f"\nDry run complete. {fixes_made} paths would be fixed.")
        else:
            print("No paths needed fixing.")

        return fixes_made

    except ET.ParseError as e:
        print(f"Error parsing VRT file: {e}")
        return 0
    except Exception as e:
        print(f"Error processing VRT file: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Fix file paths in VRT files by converting Windows paths to Unix paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fix specific root path
    python fix_vrt_paths.py input.vrt output.vrt --old-root "F:/ElevationMaps" --new-root "/Volumes/Vault/ElevationMaps"
    
    # Auto-fix with built-in mappings
    python fix_vrt_paths.py input.vrt output.vrt --auto-fix
    
    # Dry run to see what would be changed
    python fix_vrt_paths.py input.vrt output.vrt --auto-fix --dry-run
        """,
    )

    parser.add_argument("input_vrt", help="Input VRT file path")
    parser.add_argument("output_vrt", help="Output VRT file path")
    parser.add_argument(
        "--old-root", help='Windows root path to replace (e.g., "F:/ElevationMaps")'
    )
    parser.add_argument(
        "--new-root",
        help='Unix root path to replace with (e.g., "/Volumes/Vault/ElevationMaps")',
    )
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Automatically fix common Windows drive patterns",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without writing output file",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.auto_fix and (not args.old_root or not args.new_root):
        parser.error("Either use --auto-fix or provide both --old-root and --new-root")

    input_path = Path(args.input_vrt)
    output_path = Path(args.output_vrt)

    if not input_path.exists():
        print(f"Error: Input VRT file does not exist: {input_path}")
        return 1

    if not args.dry_run and output_path.exists():
        response = input(
            f"Output file {output_path} already exists. Overwrite? (y/N): "
        )
        if response.lower() != "y":
            print("Aborted.")
            return 1

    print(f"Processing VRT file: {input_path}")
    if args.auto_fix:
        print("Using auto-fix mode with built-in Windows->Unix path mappings")
        old_root = new_root = None
    else:
        print(f"Replacing '{args.old_root}' with '{args.new_root}'")
        old_root = args.old_root
        new_root = args.new_root

    fixes_made = fix_vrt_file(input_path, output_path, old_root, new_root, args.dry_run)

    if fixes_made > 0:
        print(f"\nSuccess! Fixed {fixes_made} file path(s).")
        return 0
    else:
        print("\nNo changes were needed.")
        return 0


if __name__ == "__main__":
    exit(main())
