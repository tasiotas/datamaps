import argparse
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

# --- Configuration ---
LOGIN_URL = "https://www.eorc.jaxa.jp/ALOS/en/aw3d30/data/index.htm"
BASE_URL = "https://www.eorc.jaxa.jp/ALOS/en/aw3d30/data/"
DOWNLOAD_DIR = "/Volumes/Vault/EarthData/AW3D30"
TILES_FILE = "AW3D30_tiles.txt"
OUTPUT_FILE = "aw3d30_download_links.txt"
# --- End Configuration ---


def get_subpage_links(page_source):
    """Parses the main page HTML to extract links to sub-pages from the map."""
    soup = BeautifulSoup(page_source, "html.parser")
    map_element = soup.find("map", {"name": "map_apr2024"})
    if not map_element:
        print("Error: Could not find the map element on the page.")
        return []
    area_tags = map_element.find_all("area")
    links = [tag.get("href") for tag in area_tags if tag.get("href")]
    return [os.path.join(BASE_URL, link.lstrip("./")) for link in links]


def get_download_links_from_subpage(page_source):
    """Parses a sub-page HTML to extract all download links."""
    soup = BeautifulSoup(page_source, "html.parser")
    download_buttons = soup.find_all("input", {"class": "but_img"})
    links = []

    for download_button in download_buttons:
        if download_button and "onclick" in download_button.attrs:
            onclick_attr = download_button["onclick"]
            match = re.search(r"location.href='(.*?)'", onclick_attr)
            if match:
                links.append(match.group(1))

    return links


def scrape_links(driver, limit=None):
    """Scrapes download links and saves them to a file."""
    print("Starting link scraping process...")
    driver.get(LOGIN_URL)
    input(
        "Please log in manually in the browser window. "
        "After you have successfully logged in, press Enter in this terminal to continue."
    )
    time.sleep(2)

    print("Fetching sub-page links from the main data page...")
    page_source = driver.page_source
    subpage_links = get_subpage_links(page_source)

    if not subpage_links:
        print("No sub-page links found. This could be due to a failed login.")
        return

    print(f"Found {len(subpage_links)} sub-page links.")
    all_download_links = []

    for i, link in enumerate(subpage_links):
        if limit and len(all_download_links) >= limit:
            print(f"Reached limit of {limit} links. Stopping.")
            break
        print(f"Processing sub-page {i + 1}/{len(subpage_links)}: {link}")

        driver.get(link)
        time.sleep(1)  # Wait for sub-page to load
        download_links = get_download_links_from_subpage(driver.page_source)
        if download_links:
            all_download_links.extend(download_links)
            print(f"  Found {len(download_links)} download links:")
            for dl_link in download_links:
                print(f"    {dl_link}")
        else:
            print(f"  No download links found on {link}")

    print(f"\nSaving {len(all_download_links)} download links to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        for link in all_download_links:
            f.write(link + "\n")
    print("Links saved.")


def download_and_process_single_file(link):
    """Downloads and processes a single file."""
    filename = os.path.basename(link)
    # The folder name is the zip filename without extension
    folder_name = os.path.splitext(filename)[0]
    destination_folder = os.path.join(DOWNLOAD_DIR, folder_name)

    # Check if the destination folder already exists
    if os.path.isdir(destination_folder):
        print(
            f"Skipping {filename} as output folder '{destination_folder}' already exists."
        )
        return f"Skipped {filename} (folder exists)"

    zip_path = os.path.join(DOWNLOAD_DIR, filename)
    print(f"\nProcessing: {filename}")

    # Download the file
    try:
        print(f"  Downloading {filename}...")
        response = requests.get(link, stream=True)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Download complete: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"  Error downloading {link}: {e}")
        return f"Error downloading {filename}: {e}"

    # Unzip and process
    try:
        print(f"  Unzipping {filename}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            dsm_files = [f for f in zip_ref.namelist() if f.endswith("_DSM.tif")]
            if not dsm_files:
                print(f"  Warning: No _DSM.tif file found in {filename}.")
                return f"Warning: No _DSM.tif file found in {filename}"
            for dsm_file in dsm_files:
                zip_ref.extract(dsm_file, DOWNLOAD_DIR)
                print(f"    Extracted: {dsm_file}")
        print(f"  Unzip complete: {filename}")
        return f"Success: {filename}"
    except zipfile.BadZipFile:
        print(f"  Error: {filename} is not a valid zip file.")
        return f"Error: {filename} is not a valid zip file"
    finally:
        # Clean up the zip file
        if os.path.exists(zip_path):
            print(f"  Deleting zip file: {zip_path}")
            os.remove(zip_path)


def download_and_process_files(limit=None, max_workers=10):
    """Downloads and processes files from the links file using parallel processing."""
    if not os.path.exists(OUTPUT_FILE):
        print(
            f"Error: Links file not found at {OUTPUT_FILE}. Please run with --get-links first."
        )
        return

    with open(OUTPUT_FILE, "r") as f:
        links = [line.strip() for line in f if line.strip()]

    if limit:
        links = links[:limit]
        print(f"Processing a limit of {len(links)} files.")

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    print(f"Starting parallel downloads with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_link = {
            executor.submit(download_and_process_single_file, link): link
            for link in links
        }

        # Process completed downloads
        completed = 0
        total = len(links)

        for future in as_completed(future_to_link):
            link = future_to_link[future]
            completed += 1
            try:
                result = future.result()
                print(f"[{completed}/{total}] {result}")
            except Exception as exc:
                print(
                    f"[{completed}/{total}] Error processing {os.path.basename(link)}: {exc}"
                )

    print(f"\nCompleted processing {total} files with {max_workers} parallel workers.")


def verify_downloads():
    """Verifies that all tiles from the tiles file have been downloaded."""
    if not os.path.exists(TILES_FILE):
        print(f"Error: Tiles file not found at {TILES_FILE}.")
        return

    print(f"Verifying downloads against {TILES_FILE}...")

    # Read the tiles file and extract tile names
    with open(TILES_FILE, "r") as f:
        lines = f.readlines()

    # Skip header line and extract tile names
    tile_names = []
    for line in lines[1:]:  # Skip first line (header)
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) >= 1:
                tile_names.append(parts[0])  # First column is tile name

    print(f"Found {len(tile_names)} tiles to verify in {TILES_FILE}")

    # Walk the directory once and build a set of all DSM files
    print("Scanning download directory for DSM files...")
    found_dsm_files = set()

    for root, dirs, files in os.walk(DOWNLOAD_DIR):
        for file in files:
            if file.endswith("_DSM.tif") and file.startswith("ALPSMLC30_"):
                found_dsm_files.add(file)

    print(f"Found {len(found_dsm_files)} DSM files in download directory")

    # Check which tiles have been downloaded
    missing_tiles = []
    found_tiles = []

    for tile_name in tile_names:
        # The file should be named ALPSMLC30_{tile_name}_DSM.tif
        expected_filename = f"ALPSMLC30_{tile_name}_DSM.tif"

        if expected_filename in found_dsm_files:
            found_tiles.append(tile_name)
        else:
            missing_tiles.append(tile_name)

    # Report results
    print("\n=== VERIFICATION RESULTS ===")
    print(f"Total tiles in list: {len(tile_names)}")
    print(f"Found (downloaded): {len(found_tiles)}")
    print(f"Missing: {len(missing_tiles)}")
    print(
        f"Completion: {len(found_tiles)}/{len(tile_names)} ({100 * len(found_tiles) / len(tile_names):.1f}%)"
    )

    if missing_tiles:
        print(f"\n=== MISSING TILES ({len(missing_tiles)}) ===")
        for i, tile in enumerate(missing_tiles[:20]):  # Show first 20
            print(f"  {tile}")
        if len(missing_tiles) > 20:
            print(f"  ... and {len(missing_tiles) - 20} more")

        # Save missing tiles to a file
        missing_file = "missing_tiles.txt"
        with open(missing_file, "w") as f:
            for tile in missing_tiles:
                f.write(f"{tile}\n")
        print(f"\nMissing tiles saved to: {missing_file}")
    else:
        print("\n✅ All tiles have been downloaded!")

    return len(missing_tiles) == 0


def main():
    """Main function to orchestrate the process based on command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Scrape, download, and process AW3D30 dataset files.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--get-links",
        action="store_true",
        help="Scrape the website to get download links and save them to a file.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download files from the links file, unzip, and keep only DSM.tif files.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify that all tiles from the tiles file have been downloaded.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of links to get or files to download.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel download workers (default: 10).",
    )
    args = parser.parse_args()

    if not args.get_links and not args.download and not args.verify:
        parser.print_help()
        return

    driver = None
    try:
        if args.get_links:
            print("Setting up Selenium WebDriver for link scraping...")
            driver = webdriver.Chrome(
                service=ChromeService(ChromeDriverManager().install())
            )
            scrape_links(driver, limit=args.limit)

        if args.download:
            download_and_process_files(limit=args.limit, max_workers=args.workers)

        if args.verify:
            verify_downloads()

    finally:
        if driver:
            print("Closing the browser.")
            driver.quit()


if __name__ == "__main__":
    main()
