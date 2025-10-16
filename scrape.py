import requests
from bs4 import BeautifulSoup
import os
import time
import re
import json
import hashlib
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
import logging
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import xml.etree.ElementTree as ET  # Sitemap parsing
from typing import List, Tuple, Optional, Set, Dict

# Configure logging
logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO
)

def clean_filename(url: str) -> str:
    """Create a reproducible safe filename from URL path and query (lossless via hashing)."""
    parsed = urlparse(url)
    # Build a base name from path components (limit length)
    path = parsed.path or '/'
    if path.endswith('/'):
        path += 'index'
    name = path.strip('/')[:120].replace('/', '_')
    if not name:
        name = 'index'
    # Hash full URL (without fragment) for uniqueness
    h = hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]
    # Ensure extension
    if not re.search(r'\.html?$', name, re.IGNORECASE):
        name = name + '.html'
    return f"{h}_{name}"


def normalize_url(url: str) -> str:
    """Normalize URL for frontier comparison: strip fragment, remove default ports, collapse slashes."""
    url, _frag = urldefrag(url)
    parsed = urlparse(url)
    netloc = parsed.netloc
    if netloc.endswith(':80') and parsed.scheme == 'http':
        netloc = netloc[:-3]
    if netloc.endswith(':443') and parsed.scheme == 'https':
        netloc = netloc[:-4]
    path = re.sub(r'/+', '/', parsed.path) or '/'
    # Remove meaningless trailing slash except root
    if path != '/' and path.endswith('/'):
        path = path[:-1]
    normalized = parsed._replace(netloc=netloc, path=path, params='', query=parsed.query, fragment='').geturl()
    return normalized


class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._next = 0.0

    def wait(self):
        if self.min_interval <= 0:
            return
        now = time.time()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = time.time() + self.min_interval


class RobotsRules:
    def __init__(self, allowed_all: bool = True):
        self.allowed_all = allowed_all
        self.disallow: List[str] = []

    def allows(self, path: str) -> bool:
        if self.allowed_all:
            return True
        for rule in self.disallow:
            if path.startswith(rule):
                return False
        return True


def fetch_robots(base: str, user_agent: str, timeout: int = 10) -> RobotsRules:
    robots_url = urljoin(base, '/robots.txt')
    try:
        r = requests.get(robots_url, timeout=timeout, headers={'User-Agent': user_agent})
        if r.status_code != 200:
            return RobotsRules(True)
        lines = r.text.splitlines()
        ua_section = False
        disallow: List[str] = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.lower().startswith('user-agent:'):
                agent = line.split(':', 1)[1].strip()
                ua_section = agent == '*'  # Simple logic: only honor * rules
            elif ua_section and line.lower().startswith('disallow:'):
                path = line.split(':', 1)[1].strip()
                if path:
                    disallow.append(path)
        rules = RobotsRules(False)
        rules.disallow = disallow
        return rules
    except Exception:
        return RobotsRules(True)

class PageResult:
    def __init__(self, url: str, status: str, links: Optional[List[str]] = None, meta: Optional[Dict] = None):
        self.url = url
        self.status = status  # 'ok', 'non_html', 'error', 'too_large', 'skipped'
        self.links = links or []
        self.meta = meta or {}


def scrape_page(url: str, output_dir: str, user_agent: str, max_bytes: int, retries: int = 3, base_backoff: float = 1.0) -> PageResult:
    headers = {
        'User-Agent': user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    for attempt in range(retries):
        try:
            # Perform request
            resp = requests.get(url, headers=headers, timeout=20, stream=True)
            ctype = resp.headers.get('Content-Type', '')
            clen = int(resp.headers.get('Content-Length', '0') or 0)
            if max_bytes and clen and clen > max_bytes:
                return PageResult(url, 'too_large', meta={'status_code': resp.status_code, 'content_type': ctype, 'content_length': clen})
            if not resp.ok:
                return PageResult(url, 'error', meta={'status_code': resp.status_code, 'content_type': ctype})
            if 'html' not in ctype.lower():
                return PageResult(url, 'non_html', meta={'status_code': resp.status_code, 'content_type': ctype})
            # Read (respect max_bytes)
            content_bytes = b''
            for chunk in resp.iter_content(4096):
                content_bytes += chunk
                if max_bytes and len(content_bytes) > max_bytes:
                    return PageResult(url, 'too_large', meta={'status_code': resp.status_code, 'content_type': ctype, 'content_length': len(content_bytes)})
            text = content_bytes.decode(resp.encoding or 'utf-8', errors='replace')
            soup = BeautifulSoup(text, 'html.parser')
            filename = clean_filename(url)
            full_path = os.path.join(output_dir, filename)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(f'<!-- Original URL: {url} -->\n')
                f.write(text)
            meta = {
                'url': url,
                'saved_path': full_path,
                'status_code': resp.status_code,
                'content_type': ctype,
                'fetched_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'content_length': len(content_bytes),
                'sha256': hashlib.sha256(content_bytes).hexdigest()
            }
            with open(full_path + '.json', 'w', encoding='utf-8') as mf:
                json.dump(meta, mf, indent=2)
            links = []
            for a in soup.find_all('a', href=True):
                href = a['href'].strip()
                if href.lower().startswith('javascript:') or href.lower().startswith('mailto:'):
                    continue
                abs_url = urljoin(url, href)
                links.append(abs_url)
            return PageResult(url, 'ok', links=links, meta=meta)
        except requests.exceptions.RequestException as e:
            if attempt + 1 == retries:
                return PageResult(url, 'error', meta={'error': str(e)})
            time.sleep(base_backoff * (2 ** attempt) + random.uniform(0, 0.2))
        except Exception as e:
            return PageResult(url, 'error', meta={'error': str(e)})
    return PageResult(url, 'error')

def fetch_sitemap_urls(base_url: str, user_agent: str, max_urls: int = 5000) -> List[str]:
    sitemap_url = urljoin(base_url, '/sitemap.xml')
    collected: Set[str] = set()

    def fetch(url: str):
        try:
            r = requests.get(url, timeout=15, headers={'User-Agent': user_agent, 'Accept': 'application/xml,text/xml'})
            if r.status_code != 200 or 'xml' not in r.headers.get('Content-Type', '').lower():
                return None
            return r.content
        except Exception:
            return None

    def parse(content: bytes):
        try:
            return ET.fromstring(content)
        except ET.ParseError:
            return None

    queue = [sitemap_url]
    while queue and len(collected) < max_urls:
        cur = queue.pop(0)
        data = fetch(cur)
        if not data:
            continue
        root = parse(data)
        if root is None:
            continue
        tag_lower = root.tag.lower()
        if 'sitemapindex' in tag_lower:
            for sm in root.findall('.//{*}sitemap/{*}loc'):
                if sm.text:
                    queue.append(sm.text.strip())
            continue
        for loc in root.findall('.//{*}url/{*}loc'):
            if loc.text:
                collected.add(loc.text.strip())
                if len(collected) >= max_urls:
                    break
    logging.info(f"Collected {len(collected)} URLs from sitemap(s).")
    return list(collected)


def scrape_website(start_urls: List[str],
                   output_dir: str,
                   max_pages: int = 100,
                   num_workers: int = 5,
                   respect_robots: bool = False,
                   min_delay: float = 0.2,
                   max_bytes: int = 2_000_000,
                   allow_regex: Optional[str] = None,
                   deny_regex: Optional[str] = None,
                   depth_limit: Optional[int] = None,
                   user_agent: str = 'CHPC-Scraper/1.0 (+https://example.org)') -> int:
    """Recursively scrape website and save HTML content + metadata sidecars."""
    if not start_urls:
        logging.error("No start URLs provided.")
        return 0
    os.makedirs(output_dir, exist_ok=True)
    # Domain + base
    first = urlparse(start_urls[0])
    domain = first.netloc
    base_url = f"{first.scheme}://{domain}"

    visited: Set[str] = set()
    enqueued: Set[str] = set()
    frontier: deque[Tuple[str, int]] = deque()
    for u in start_urls:
        nu = normalize_url(u)
        frontier.append((nu, 0))
        enqueued.add(nu)

    # Robots
    robots = fetch_robots(base_url, user_agent) if respect_robots else RobotsRules(True)
    if respect_robots:
        logging.info(f"Robots disallow rules: {len(robots.disallow)}")

    # Compile filters
    allow_pat = re.compile(allow_regex) if allow_regex else None
    deny_pat = re.compile(deny_regex) if deny_regex else None

    # Sitemap augmentation
    sitemap_added = 0
    sitemap_urls = fetch_sitemap_urls(base_url, user_agent) if max_pages > 0 else []
    for su in sitemap_urls:
        nsu = normalize_url(su)
        if urlparse(nsu).netloc == domain and nsu not in enqueued:
            frontier.append((nsu, 0))
            enqueued.add(nsu)
            sitemap_added += 1
    if sitemap_added:
        logging.info(f"Added {sitemap_added} sitemap URLs to frontier.")

    limiter = RateLimiter(min_interval=min_delay)
    saved_pages = 0
    counters = {k: 0 for k in ['ok', 'non_html', 'error', 'too_large', 'skipped']}

    def submit(executor, batch):
        futures = {}
        for url, depth in batch:
            limiter.wait()
            futures[executor.submit(scrape_page, url, output_dir, user_agent, max_bytes)] = (url, depth)
        return futures

    with tqdm(total=max_pages, desc='Scraping') as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            while frontier and saved_pages < max_pages:
                # Prepare batch
                batch = []
                while frontier and len(batch) < num_workers * 2 and saved_pages + len(batch) < max_pages:
                    url, depth = frontier.popleft()
                    if url in visited:
                        continue
                    # Depth limit
                    if depth_limit is not None and depth > depth_limit:
                        counters['skipped'] += 1
                        continue
                    # Domain filter
                    if urlparse(url).netloc != domain:
                        counters['skipped'] += 1
                        continue
                    # Robots
                    if not robots.allows(urlparse(url).path):
                        counters['skipped'] += 1
                        continue
                    # Path regex filters
                    pathq = urlparse(url).path
                    if deny_pat and deny_pat.search(pathq):
                        counters['skipped'] += 1
                        continue
                    if allow_pat and not allow_pat.search(pathq):
                        counters['skipped'] += 1
                        continue
                    batch.append((url, depth))

                if not batch:
                    break
                futures = submit(executor, batch)
                for fut in as_completed(futures):
                    url, depth = futures[fut]
                    visited.add(url)
                    try:
                        res: PageResult = fut.result()
                    except Exception as e:
                        counters['error'] += 1
                        continue
                    counters[res.status] = counters.get(res.status, 0) + 1
                    if res.status == 'ok':
                        saved_pages += 1
                        pbar.update(1)
                        # Enqueue links
                        for link in res.links:
                            nlink = normalize_url(link)
                            if nlink in enqueued or nlink in visited:
                                continue
                            if urlparse(nlink).netloc != domain:
                                continue
                            enqueued.add(nlink)
                            frontier.append((nlink, depth + 1))
                    if saved_pages >= max_pages:
                        break

    logging.info(f"Scraping complete. Saved {saved_pages} pages from {domain}.")
    logging.info("Counters: " + ', '.join(f"{k}={v}" for k, v in counters.items()))
    return saved_pages

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursive website scraper with metadata & politeness features")
    parser.add_argument("--output", default="source_documents/chpc_utah", help="Directory for saved HTML")
    parser.add_argument("--max-pages", type=int, default=600)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--respect-robots", action='store_true')
    parser.add_argument("--delay", type=float, default=0.25, help="Minimum seconds between requests (global)")
    parser.add_argument("--max-bytes", type=int, default=2_000_000, help="Max page size to download")
    parser.add_argument("--allow-path-regex", type=str, default=None, help="Only crawl paths matching this regex")
    parser.add_argument("--deny-path-regex", type=str, default=None, help="Skip paths matching this regex")
    parser.add_argument("--depth-limit", type=int, default=None, help="Max crawl depth (0=start URLs)")
    parser.add_argument("--user-agent", type=str, default='CHPC-Scraper/1.0 (+https://www.chpc.utah.edu)')
    parser.add_argument("start_urls", nargs='*', default=[
        "https://www.chpc.utah.edu/",
        "https://chpc.utah.edu/documentation/software/",
        "https://chpc.utah.edu/resources/",
        "https://chpc.utah.edu/userservices/"
    ])
    a = parser.parse_args()
    scrape_website(
        a.start_urls,
        a.output,
        max_pages=a.max_pages,
        num_workers=a.workers,
        respect_robots=a.respect_robots,
        min_delay=a.delay,
        max_bytes=a.max_bytes,
        allow_regex=a.allow_path_regex,
        deny_regex=a.deny_path_regex,
        depth_limit=a.depth_limit,
        user_agent=a.user_agent,
    )
