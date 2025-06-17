import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin, urlparse
import logging
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import xml.etree.ElementTree as ET # Added for sitemap parsing

# Configure logging
logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO
)

def clean_filename(url):
    """Create a safe filename from a URL."""
    parsed = urlparse(url)
    path = parsed.path
    
    # Handle root path
    if path == "" or path == "/":
        return "index.html"
    
    # Remove trailing slash if present
    if path.endswith("/"):
        path = path[:-1]
    
    # Remove leading slash
    if path.startswith("/"):
        path = path[1:]
    
    # Replace special characters
    path = path.replace("/", "_").replace("?", "_").replace("&", "_")
    
    # Add .html extension if missing
    if not path.endswith(".html") and not path.endswith(".htm"):
        path += ".html"
    
    return path

def scrape_page(url, output_dir, retries=3, delay=1):
    """Scrape a single page and return a list of found links. Includes retry logic."""
    for attempt in range(retries):
        try:
            # Add some random delay to be respectful to the server
            time.sleep(random.uniform(0.5, 2.0))

            # Use a realistic user agent
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml',
                'Accept-Language': 'en-US,en;q=0.9',
            }

            response = requests.get(url, headers=headers, timeout=15)

            # Check if the response is HTML
            if not response.ok or 'text/html' not in response.headers.get('Content-Type', ''):
                logging.warning(f"Skipping non-HTML or failed request for {url} (Status: {response.status_code})")
                return [] # Return empty list on non-HTML or error

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Create a filename based on the URL
            filename = clean_filename(url)
            full_path = os.path.join(output_dir, filename)

            # Ensure subdirectories exist
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            # Save the HTML with the source URL comment for your extractor
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(f'<!-- Original URL: {url} -->\n') # Corrected escape sequence
                f.write(str(soup))

            logging.info(f"Saved {url} to {full_path}")

            # Find all links on the page
            links = []
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                absolute_url = urljoin(url, href)
                links.append(absolute_url)

            return links # Success, return links

        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt + 1}/{retries} failed for {url}: {e}")
            if attempt + 1 == retries:
                logging.error(f"Final attempt failed for {url}. Skipping.")
                return [] # Return empty list after final retry fails
            time.sleep(delay * (2 ** attempt)) # Exponential backoff

        except Exception as e:
            logging.error(f"Non-request error scraping {url}: {e}")
            return [] # Return empty list for other errors

    return [] # Should not be reached, but ensures a return value

def fetch_sitemap_urls(base_url):
    """Fetch and parse sitemap.xml to extract URLs."""
    sitemap_url = urljoin(base_url, "/sitemap.xml")
    urls = []
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; SitemapFetcher/1.0)',
            'Accept': 'application/xml, text/xml',
        }
        response = requests.get(sitemap_url, headers=headers, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Check content type
        if 'xml' not in response.headers.get('Content-Type', '').lower():
            logging.warning(f"Sitemap at {sitemap_url} is not XML. Content-Type: {response.headers.get('Content-Type')}")
            return []

        # Parse XML
        root = ET.fromstring(response.content)
        # Namespace is common in sitemaps
        namespace = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        for url_element in root.findall('sm:url', namespace):
            loc_element = url_element.find('sm:loc', namespace)
            if loc_element is not None and loc_element.text:
                urls.append(loc_element.text.strip())
        logging.info(f"Found {len(urls)} URLs in sitemap: {sitemap_url}")
        return urls

    except requests.exceptions.RequestException as e:
        logging.warning(f"Could not fetch or parse sitemap at {sitemap_url}: {e}")
        return []
    except ET.ParseError as e:
        logging.warning(f"Error parsing XML sitemap at {sitemap_url}: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error processing sitemap {sitemap_url}: {e}")
        return []


def scrape_website(start_urls, output_dir, max_pages=100, num_workers=5):
    """Recursively scrape a website and save HTML files."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract domain for filtering
    if not start_urls:
        logging.error("No start URLs provided.")
        return 0
    domain = urlparse(start_urls[0]).netloc
    base_url = f"{urlparse(start_urls[0]).scheme}://{domain}"

    # Track visited and queued URLs
    visited_urls = set()
    urls_to_visit = start_urls.copy()  # Start with all provided URLs

    # Attempt to fetch sitemap URLs
    sitemap_urls = fetch_sitemap_urls(base_url)
    for url in sitemap_urls:
        parsed_link = urlparse(url)
        # Add sitemap URLs if they are on the same domain and not already queued
        if parsed_link.netloc == domain and url not in urls_to_visit:
            urls_to_visit.append(url)
            logging.info(f"Added sitemap URL to queue: {url}")

    # Count of saved pages
    saved_pages = 0

    logging.info(f"Starting to scrape from {len(urls_to_visit)} initial URLs (including sitemap)")
    logging.info(f"Target domain: {domain}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Max pages: {max_pages}")

    with tqdm(total=max_pages, desc="Scraping pages") as progress_bar:
        # Using ThreadPoolExecutor for concurrent scraping
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            while urls_to_visit and saved_pages < max_pages:
                # Get a batch of URLs to process
                current_batch = []
                processed_in_batch = set() # Track URLs added to this batch to avoid duplicates
                while urls_to_visit and len(current_batch) < num_workers * 2: # Fill batch faster
                    next_url = urls_to_visit.pop(0)
                    # Normalize URL slightly (remove fragment)
                    next_url = urljoin(next_url, urlparse(next_url).path)
                    if next_url not in visited_urls and next_url not in processed_in_batch:
                        # Only add URLs matching the target domain
                        if urlparse(next_url).netloc == domain:
                            current_batch.append(next_url)
                            processed_in_batch.add(next_url)
                        else:
                            logging.debug(f"Skipping off-domain URL: {next_url}")


                if not current_batch:
                    if not urls_to_visit:
                        logging.info("No more URLs to visit.")
                    break

                # Submit all URLs in the current batch to the thread pool
                future_to_url = {
                    executor.submit(scrape_page, url, output_dir): url
                    for url in current_batch
                }

                # Process completed futures
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    visited_urls.add(url) # Mark as visited once processed

                    try:
                        new_links = future.result()

                        # If page was scraped successfully (result is not None), increment counter
                        # Check if the result is a list (success) vs None (error during scrape)
                        if isinstance(new_links, list):
                            saved_pages += 1
                            progress_bar.update(1)

                            # Filter and queue new URLs
                            for link in new_links:
                                parsed_link = urlparse(link)
                                # Normalize link
                                normalized_link = urljoin(link, parsed_link.path)
                                # Only follow links on the same domain and not visited/queued yet
                                if (parsed_link.netloc == domain and
                                    normalized_link not in visited_urls and
                                    normalized_link not in urls_to_visit):
                                    # Basic check to avoid obvious non-HTML links
                                    if not any(normalized_link.lower().endswith(ext) for ext in ['.pdf', '.zip', '.jpg', '.png', '.gif', '.css', '.js']):
                                        urls_to_visit.append(normalized_link)

                    except Exception as e:
                        logging.error(f"Error processing future for {url}: {e}")

                    # Check if we've reached the limit
                    if saved_pages >= max_pages:
                        logging.info(f"Reached max pages limit ({max_pages}). Stopping.")
                        # Cancel remaining futures (optional, helps stop faster)
                        for fut in future_to_url:
                            if not fut.done():
                                fut.cancel()
                        urls_to_visit = [] # Clear queue
                        break # Exit the inner loop

    logging.info(f"Scraping complete. Saved {saved_pages} pages from {domain}.")
    return saved_pages

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape the CHPC Utah website")
    parser.add_argument("--output", type=str, default="source_documents/chpc_utah", 
                        help="Output directory for HTML files")
    parser.add_argument("--max-pages", type=int, default=300, 
                        help="Maximum number of pages to scrape")
    parser.add_argument("--workers", type=int, default=5, 
                        help="Number of concurrent workers")
    args = parser.parse_args()
    

    start_urls = [
        "https://www.chpc.utah.edu/",
        "https://chpc.utah.edu/documentation/software/",
        "https://chpc.utah.edu/resources/",
        "https://chpc.utah.edu/userservices/"
    ]
    
    scrape_website(start_urls, args.output, args.max_pages, args.workers)
