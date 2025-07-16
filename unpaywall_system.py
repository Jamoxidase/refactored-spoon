#!/usr/bin/env python3
"""
Complete Enhanced Unpaywall System
Using your proven Selenium PDF downloader with enhanced async metadata collection
"""

import asyncio
import aiohttp
import json
import time
import csv
import logging
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
import os
import hashlib

# Import your proven Selenium PDF downloader
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PaperMetadata:
    """Enhanced paper metadata structure with full tracking"""
    # Core identification (required fields)
    doi: str
    title: str
    journal: str
    authors: str
    is_oa: bool
    search_terms: List[str]
    pdf_links: List[Dict]
    
    # Optional fields with defaults
    year: Optional[int] = None
    abstract: Optional[str] = None
    oa_date: Optional[str] = None
    oa_locations: List[Dict] = field(default_factory=list)
    publisher: Optional[str] = None
    genre: Optional[str] = None  # journal-article, book-chapter, etc.
    is_paratext: Optional[bool] = None  # editorial vs research content
    classification_status: str = "pending"
    classification_reason: Optional[str] = None
    download_status: str = "not_attempted"  # not_attempted, success, failed
    download_error: Optional[str] = None
    download_filepath: Optional[str] = None
    download_attempts: int = 0
    data_standard: Optional[int] = None  # 1 or 2 (data quality)
    updated: Optional[str] = None  # last update timestamp
    evidence: Optional[str] = None  # how OA status was determined
    
    def get_unique_id(self) -> str:
        return hashlib.md5(self.doi.encode()).hexdigest() if self.doi else None

class MockClassifier:
    """Mock classifier with keyword-based filtering"""
    
    async def classify_paper(self, paper: PaperMetadata) -> Tuple[bool, str]:
        title_lower = paper.title.lower()
        
        # Relevant keywords for tRNA research
        relevant_keywords = [
            'trna', 'transfer rna', 'aminoacyl', 'codon', 'anticodon',
            'ribosome', 'translation', 'protein synthesis', 'genetic code'
        ]
        
        for keyword in relevant_keywords:
            if keyword in title_lower:
                return True, f"Contains relevant keyword: {keyword}"
        
        # Accept by default for testing but mark reason
        return True, "Accepted by default classification"

class EnhancedUnpaywallExtractor:
    """Enhanced async Unpaywall extractor with classification and deduplication"""
    
    def __init__(self, email: str, classifier, max_concurrent: int = 3):
        self.email = email
        self.classifier = classifier
        self.max_concurrent = max_concurrent
        self.base_url = "https://api.unpaywall.org/v2/search/"
        self.session: Optional[aiohttp.ClientSession] = None
        self.seen_dois: Set[str] = set()
        self.rate_limit_delay = 0.2  # 200ms between requests
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_term_async(self, query: str, max_pages: int = 1) -> List[PaperMetadata]:
        """Search for papers with enhanced data extraction"""
        papers = []
        page = 1
        
        logger.info(f"Searching for: '{query}' (max {max_pages} pages)")
        
        while page <= max_pages:
            params = {'query': query, 'email': self.email, 'page': page}
            
            try:
                await asyncio.sleep(self.rate_limit_delay)
                
                async with self.session.get(self.base_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    if not data.get('results'):
                        logger.info(f"No results for '{query}' at page {page}")
                        break
                    
                    page_papers = []
                    for paper_data in data['results'][:8]:  # Limit to 8 per search term
                        paper = self._extract_enhanced_paper_info(paper_data, query)
                        if paper and paper.doi not in self.seen_dois:
                            # Classify paper before storing
                            is_useful, reason = await self.classifier.classify_paper(paper)
                            paper.classification_status = "accepted" if is_useful else "rejected"
                            paper.classification_reason = reason
                            
                            page_papers.append(paper)
                            self.seen_dois.add(paper.doi)
                    
                    papers.extend(page_papers)
                    accepted_count = sum(1 for p in page_papers if p.classification_status == "accepted")
                    logger.info(f"Page {page}: {len(page_papers)} papers, {accepted_count} accepted")
                    
                    page += 1
                    
            except Exception as e:
                logger.error(f"Error for '{query}' page {page}: {e}")
                break
        
        logger.info(f"Search complete for '{query}': {len(papers)} total papers")
        return papers
    
    def _extract_enhanced_paper_info(self, paper_data: Dict, search_term: str) -> Optional[PaperMetadata]:
        """Extract comprehensive paper information including all metadata"""
        response = paper_data.get('response', {})
        
        if not response.get('doi'):
            return None
        
        # Extract enhanced author information
        authors_list = response.get('z_authors', []) or []
        authors = []
        for author in authors_list[:3]:  # Limit to first 3 authors
            if isinstance(author, dict):
                family = author.get('family', '')
                given = author.get('given', '')
                if family:
                    authors.append(f"{family}, {given}".strip(', '))
        
        # Extract all OA locations for comprehensive tracking
        oa_locations = []
        if response.get('oa_locations'):
            for loc in response['oa_locations']:
                oa_locations.append({
                    'url': loc.get('url'),
                    'url_for_pdf': loc.get('url_for_pdf'),
                    'host_type': loc.get('host_type'),
                    'license': loc.get('license'),
                    'repository_institution': loc.get('repository_institution')
                })
        
        # Create enhanced paper metadata with all fields
        paper = PaperMetadata(
            doi=response.get('doi', ''),
            title=response.get('title', ''),
            year=response.get('year'),
            journal=response.get('journal_name', ''),
            authors='; '.join(authors),
            abstract=None,  # Unpaywall doesn't provide abstracts
            is_oa=response.get('is_oa', False),
            oa_date=response.get('oa_date'),
            oa_locations=oa_locations,
            publisher=response.get('publisher'),
            genre=response.get('genre'),
            is_paratext=response.get('is_paratext'),
            search_terms=[search_term],
            pdf_links=[],
            data_standard=response.get('data_standard'),
            updated=response.get('updated'),
            evidence=response.get('evidence')
        )
        
        # Extract enhanced PDF link metadata
        self._extract_enhanced_pdf_links(response, paper)
        
        # Return paper even if no PDF links - we want all metadata
        return paper
    
    def _extract_enhanced_pdf_links(self, response: Dict, paper: PaperMetadata):
        """Extract PDF links with comprehensive metadata"""
        pdf_links = []
        
        # Process all oa_locations for comprehensive data
        if response.get('oa_locations'):
            for location in response['oa_locations']:
                pdf_url = location.get('url_for_pdf')
                if pdf_url:
                    pdf_links.append({
                        'pdf_url': pdf_url,
                        'host_type': location.get('host_type', ''),
                        'license': location.get('license', ''),
                        'repository_institution': location.get('repository_institution', ''),
                        'pmh_id': location.get('pmh_id', ''),
                        'endpoint_id': location.get('endpoint_id', ''),
                        'evidence': location.get('evidence', ''),
                        'updated': location.get('updated', ''),
                        'is_best': False
                    })
        
        # Mark best location
        if response.get('best_oa_location'):
            best_location = response['best_oa_location']
            best_pdf_url = best_location.get('url_for_pdf')
            if best_pdf_url:
                best_exists = any(link['pdf_url'] == best_pdf_url for link in pdf_links)
                if not best_exists:
                    pdf_links.insert(0, {
                        'pdf_url': best_pdf_url,
                        'host_type': best_location.get('host_type', ''),
                        'license': best_location.get('license', ''),
                        'repository_institution': best_location.get('repository_institution', ''),
                        'pmh_id': best_location.get('pmh_id', ''),
                        'endpoint_id': best_location.get('endpoint_id', ''),
                        'evidence': best_location.get('evidence', ''),
                        'updated': best_location.get('updated', ''),
                        'is_best': True
                    })
                else:
                    # Mark existing entry as best
                    for link in pdf_links:
                        if link['pdf_url'] == best_pdf_url:
                            link['is_best'] = True
                            break
        
        paper.pdf_links = pdf_links
    
    def _merge_duplicate_papers(self, papers: List[PaperMetadata]) -> List[PaperMetadata]:
        """Robust deduplication - merge papers with same DOI from different searches"""
        paper_dict = {}
        
        for paper in papers:
            if paper.doi in paper_dict:
                # Merge search terms
                existing = paper_dict[paper.doi]
                existing.search_terms.extend(paper.search_terms)
                existing.search_terms = list(set(existing.search_terms))  # Remove duplicates
                
                # Merge PDF links if different
                existing_urls = {link['pdf_url'] for link in existing.pdf_links}
                for link in paper.pdf_links:
                    if link['pdf_url'] not in existing_urls:
                        existing.pdf_links.append(link)
            else:
                paper_dict[paper.doi] = paper
        
        return list(paper_dict.values())
    
    def _save_papers_to_csv(self, papers: List[PaperMetadata], output_file: str):
        """Save papers with enhanced metadata to CSV"""
        fieldnames = [
            'doi', 'title', 'year', 'journal', 'authors', 'is_oa', 'oa_date',
            'publisher', 'genre', 'is_paratext', 'data_standard', 'updated', 'evidence',
            'search_terms', 'classification_status', 'classification_reason',
            'pdf_count', 'pdf_urls', 'host_types', 'licenses', 'best_pdf_url',
            'download_status', 'download_filepath', 'download_error', 'download_attempts'
        ]
        
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for paper in papers:
                # Find best PDF URL
                best_pdf = next((link['pdf_url'] for link in paper.pdf_links if link.get('is_best')), 
                              paper.pdf_links[0]['pdf_url'] if paper.pdf_links else '')
                
                row = {
                    'doi': paper.doi,
                    'title': paper.title,
                    'year': paper.year,
                    'journal': paper.journal,
                    'authors': paper.authors,
                    'is_oa': paper.is_oa,
                    'oa_date': paper.oa_date,
                    'publisher': paper.publisher,
                    'genre': paper.genre,
                    'is_paratext': paper.is_paratext,
                    'data_standard': paper.data_standard,
                    'updated': paper.updated,
                    'evidence': paper.evidence,
                    'search_terms': '|'.join(paper.search_terms),
                    'classification_status': paper.classification_status,
                    'classification_reason': paper.classification_reason,
                    'pdf_count': len(paper.pdf_links),
                    'pdf_urls': '|'.join([link['pdf_url'] for link in paper.pdf_links]),
                    'host_types': '|'.join([link.get('host_type', '') or '' for link in paper.pdf_links]),
                    'licenses': '|'.join([link.get('license', '') or '' for link in paper.pdf_links]),
                    'best_pdf_url': best_pdf,
                    'download_status': paper.download_status,
                    'download_filepath': paper.download_filepath or '',
                    'download_error': paper.download_error or '',
                    'download_attempts': paper.download_attempts
                }
                writer.writerow(row)

# YOUR PROVEN SELENIUM PDF DOWNLOADER - MINIMAL MODIFICATIONS
class SeleniumPDFDownloader:
    def __init__(self, download_dir, headless=True):
        self.download_dir = os.path.abspath(download_dir)
        os.makedirs(download_dir, exist_ok=True)
        
        # Setup Chrome options
        self.chrome_options = Options()
        if headless:
            self.chrome_options.add_argument("--headless")
        
        # Important: Set download directory
        prefs = {
            "download.default_directory": self.download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "plugins.always_open_pdf_externally": True  # Download PDFs instead of opening in browser
        }
        self.chrome_options.add_experimental_option("prefs", prefs)
        
        # Anti-detection options
        self.chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        self.chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.chrome_options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = None
        self.headless = headless
    
    def start_browser(self):
        """Initialize the browser"""
        self.driver = webdriver.Chrome(options=self.chrome_options)
        
        # Execute script to remove webdriver property
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        print("Browser started successfully")
    
    def close_browser(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def detect_turnstile(self):
        """Check if we're on a Turnstile/Cloudflare challenge page"""
        try:
            page_source_lower = self.driver.page_source.lower()
            current_url = self.driver.current_url.lower()
            
            # Check for Turnstile/Cloudflare indicators
            turnstile_indicators = [
                "turnstile", "cloudflare", "challenge-platform", 
                "cf-challenge", "checking your browser", "please wait"
            ]
            
            return any(indicator in page_source_lower or indicator in current_url 
                      for indicator in turnstile_indicators)
                
        except Exception as e:
            return False
    
    def sanitize_filename(self, filename):
        """Clean filename for filesystem"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename[:200]
    
    def download_pdf_selenium(self, url, expected_filename=None):
        """Download PDF using Selenium (real browser)"""
        try:
            print(f"Visiting: {url}")
            
            # Navigate to URL
            self.driver.get(url)
            
            # For direct PDF URLs, the download should start automatically
            # For landing pages, we might need to find and click download links
            if not url.endswith('.pdf'):
                try:
                    # Look for common PDF download patterns
                    pdf_selectors = [
                        "a[href*='.pdf']",
                        "a[href*='pdf']",
                        "a[title*='PDF']",
                        "a[aria-label*='PDF']",
                        ".pdf-download",
                        ".download-pdf"
                    ]
                    
                    for selector in pdf_selectors:
                        try:
                            pdf_link = WebDriverWait(self.driver, 2).until(
                                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                            )
                            print(f"Found PDF link, clicking...")
                            pdf_link.click()
                            break
                        except:
                            continue
                except:
                    # If no download link found, the page itself might trigger download
                    pass
            
            # Wait adaptively for download to appear
            initial_files = set(os.listdir(self.download_dir))
            
            # In non-headless mode, wait longer if we detect Turnstile
            if not self.headless and self.detect_turnstile():
                print(f"🚫 Turnstile/Cloudflare detected - please solve the challenge in the browser window")
                print(f"The download will begin automatically after you solve it...")
                max_wait = 300  # 5 minutes for manual solving
            else:
                max_wait = 10  # Default 10 seconds
            
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                current_files = set(os.listdir(self.download_dir))
                new_files = current_files - initial_files
                
                # Check for new PDF files
                pdf_files = [f for f in new_files if f.endswith('.pdf')]
                if pdf_files:
                    break
                    
                time.sleep(0.1)  # Check every 100ms
            
            # Check if file was downloaded
            downloads = [f for f in os.listdir(self.download_dir) if f.endswith('.pdf')]
            if downloads:
                # Get the most recently downloaded file
                latest_file = max([os.path.join(self.download_dir, f) for f in downloads], 
                                key=os.path.getctime)
                
                # Rename if we have an expected filename
                if expected_filename:
                    expected_path = os.path.join(self.download_dir, expected_filename)
                    if latest_file != expected_path:
                        os.rename(latest_file, expected_path)
                        latest_file = expected_path
                
                print(f"Downloaded: {os.path.basename(latest_file)}")
                return latest_file
            else:
                print(f"No PDF downloaded from: {url}")
                return None
                
        except Exception as e:
            print(f"Selenium download failed for {url}: {e}")
            return None
    
    def fallback_requests_download(self, url, filename):
        """Fallback to requests method"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            filepath = os.path.join(self.download_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"Fallback download successful: {filename}")
            return filepath
            
        except Exception as e:
            print(f"Fallback download failed: {e}")
            return None

    def download_papers_from_metadata(self, papers: List[PaperMetadata], num_workers: int = 3) -> List[PaperMetadata]:
        """Download PDFs for papers using parallel Selenium workers and update metadata"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        # Prepare download tasks
        download_tasks = []
        for paper in papers:
            if paper.classification_status != "accepted":
                paper.download_status = "not_attempted"
                paper.download_error = "Paper not accepted for download"
                continue
                
            if not paper.pdf_links:
                paper.download_status = "failed"
                paper.download_error = "No PDF links available"
                continue
            
            # Generate filename from paper metadata
            safe_title = self.sanitize_filename(paper.title)[:50]
            safe_doi = paper.doi.replace('/', '_').replace('.', '_')
            filename = f"{safe_title}_{safe_doi}.pdf"
            filepath = os.path.join(self.download_dir, filename)
            
            # Skip if already exists
            if os.path.exists(filepath):
                print(f"Already exists: {filename}")
                paper.download_status = "success"
                paper.download_filepath = filepath
                paper.download_attempts = 0
                continue
            
            download_tasks.append((paper, filename))
        
        print(f"Starting {num_workers} parallel workers for {len(download_tasks)} papers")
        
        successful_downloads = []
        failed_papers = []
        
        def worker_download(worker_id, paper, filename):
            """Worker function for downloading a single paper"""
            # Each worker gets its own driver
            worker_downloader = SeleniumPDFDownloader(self.download_dir, headless=self.headless)
            worker_downloader.start_browser()
            
            try:
                print(f"Worker {worker_id}: Processing {paper.title[:50]}...")
                
                filepath = os.path.join(self.download_dir, filename)
                paper.download_attempts = 0
                
                # Try URLs in order (best first), stop at first success
                for i, link in enumerate(paper.pdf_links):
                    url = link['pdf_url']
                    is_best = link.get('is_best', False)
                    
                    print(f"Worker {worker_id}: Trying URL {i+1}/{len(paper.pdf_links)} {'(BEST)' if is_best else ''}")
                    paper.download_attempts += 1
                    
                    result = worker_downloader.download_pdf_selenium(url, filename)
                    if result:
                        print(f"Worker {worker_id}: SUCCESS - Downloaded {filename}")
                        paper.download_status = "success"
                        paper.download_filepath = filepath
                        return paper
                    else:
                        print(f"Worker {worker_id}: Failed URL {i+1}, trying next...")
                        # Take screenshot on failure for debugging (includes Turnstile pages)
                        try:
                            screenshot_path = os.path.join(self.download_dir, f"failure_{worker_id}_{i}.png")
                            worker_downloader.driver.save_screenshot(screenshot_path)
                            print(f"Worker {worker_id}: Screenshot saved: {screenshot_path}")
                            
                            # Also save page source for debugging
                            source_path = os.path.join(self.download_dir, f"failure_{worker_id}_{i}.html")
                            with open(source_path, 'w', encoding='utf-8') as f:
                                f.write(worker_downloader.driver.page_source)
                            print(f"Worker {worker_id}: Page source saved: {source_path}")
                        except Exception as e:
                            print(f"Worker {worker_id}: Could not save debug files: {e}")
                
                print(f"Worker {worker_id}: FAILED - All URLs failed for {paper.title[:30]}...")
                paper.download_status = "failed"
                paper.download_error = f"All {len(paper.pdf_links)} URLs failed"
                return paper
                
            except Exception as e:
                paper.download_status = "failed"
                paper.download_error = str(e)
                return paper
            finally:
                worker_downloader.close_browser()
        
        # Execute downloads in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for i, (paper, filename) in enumerate(download_tasks):
                worker_id = i % num_workers
                future = executor.submit(worker_download, worker_id, paper, filename)
                future_to_task[future] = (paper, filename)
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                try:
                    updated_paper = future.result()
                    if updated_paper.download_status == 'success':
                        successful_downloads.append(updated_paper.download_filepath)
                    else:
                        failed_papers.append(updated_paper.title)
                except Exception as e:
                    paper, filename = future_to_task[future]
                    print(f"Worker exception for {paper.title[:30]}: {e}")
                    paper.download_status = "failed"
                    paper.download_error = f"Worker exception: {str(e)}"
                    failed_papers.append(paper.title)
        
        print(f"\n=== PARALLEL SELENIUM DOWNLOAD SUMMARY ===")
        print(f"Successfully downloaded: {len(successful_downloads)} PDFs")
        print(f"Failed downloads: {len(failed_papers)}")
        
        if failed_papers:
            print(f"\nFailed papers:")
            for title in failed_papers[:10]:
                print(f"  {title[:60]}...")
        
        return papers  # Return all papers with updated metadata

async def run_complete_test(headless=True):
    """Run complete test: enhanced search -> classification -> Selenium PDF download"""
    EMAIL = "j.martinez.4823@gmail.com"
    
    # Small test parameters
    SEARCH_TERMS = ["tRNA processing", "transfer RNA modification"]
    OUTPUT_FILE = "./complete_test_results.csv"
    DOWNLOAD_DIR = "./complete_test_pdfs"
    
    print("=== COMPLETE ENHANCED SYSTEM TEST ===")
    print(f"Search terms: {SEARCH_TERMS}")
    print(f"Output CSV: {OUTPUT_FILE}")
    print(f"PDF directory: {DOWNLOAD_DIR}")
    
    start_time = time.time()
    classifier = MockClassifier()
    
    # Phase 1: Enhanced async search and classification
    print(f"\n=== PHASE 1: SEARCH & CLASSIFY ===")
    async with EnhancedUnpaywallExtractor(EMAIL, classifier, max_concurrent=2) as extractor:
        all_papers = []
        
        for term in SEARCH_TERMS:
            papers = await extractor.search_term_async(term, max_pages=1)
            all_papers.extend(papers)
        
        # Apply deduplication
        final_papers = extractor._merge_duplicate_papers(all_papers)
        
        # Save enhanced metadata
        extractor._save_papers_to_csv(final_papers, OUTPUT_FILE)
    
    search_time = time.time() - start_time
    accepted_papers = [p for p in final_papers if p.classification_status == "accepted"]
    
    print(f"Search complete: {len(final_papers)} papers, {len(accepted_papers)} accepted")
    print(f"Search time: {search_time:.1f} seconds")
    
    # Phase 2: Your proven Selenium PDF downloading
    print(f"\n=== PHASE 2: SELENIUM PDF DOWNLOAD ===")
    print(f"Running in {'HEADLESS' if headless else 'HEADED'} mode")
    downloader = SeleniumPDFDownloader(DOWNLOAD_DIR, headless=headless)
    downloaded_pdfs = downloader.download_papers_from_metadata(accepted_papers)
    
    total_time = time.time() - start_time
    
    # Final results
    print(f"\n=== COMPLETE SYSTEM RESULTS ===")
    print(f"Total execution time: {total_time:.1f} seconds")
    print(f"Papers found: {len(final_papers)}")
    print(f"Papers classified as accepted: {len(accepted_papers)}")
    print(f"PDFs successfully downloaded: {len(downloaded_pdfs)}")
    print(f"Download success rate: {len(downloaded_pdfs)}/{len(accepted_papers)} ({100*len(downloaded_pdfs)/len(accepted_papers) if accepted_papers else 0:.1f}%)")
    print(f"CSV saved: {OUTPUT_FILE}")
    print(f"PDFs saved: {DOWNLOAD_DIR}")
    
    # Show sample results
    print(f"\n=== SAMPLE PAPERS ===")
    for i, paper in enumerate(final_papers[:3]):
        print(f"{i+1}. {paper.title[:70]}...")
        print(f"   DOI: {paper.doi}")
        print(f"   Classification: {paper.classification_status} - {paper.classification_reason}")
        print(f"   PDF URLs: {len(paper.pdf_links)}")
    
    return final_papers, downloaded_pdfs

def interactive_mode():
    """Interactive mode for simple configuration"""
    from argparse import Namespace
    
    print("="*60)
    print("ENHANCED UNPAYWALL PDF DOWNLOADER - INTERACTIVE MODE")
    print("="*60)
    print("Please provide the following information:")
    print()
    
    # Search keywords
    print("1. SEARCH KEYWORDS")
    print("   Enter search terms separated by commas")
    print("   Example: tRNA processing, transfer RNA, ribosome")
    keywords_input = input("   Keywords: ").strip()
    keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
    
    if not keywords:
        print("   No keywords provided. Exiting.")
        exit(1)
    
    # Max pages
    print("\n2. SEARCH SCOPE")
    print("   How many pages per keyword? (1 page ≈ 20 papers)")
    print("   Recommendation: 1-2 for testing, 5+ for comprehensive search")
    max_pages_input = input("   Max pages [1]: ").strip()
    max_pages = int(max_pages_input) if max_pages_input else 1
    
    # Filter keywords
    print("\n3. PAPER FILTERING")
    print("   Filter papers by keywords in title? (optional)")
    print("   Only papers containing these words will be accepted")
    print("   Example: human, mouse, clinical")
    filter_input = input("   Filter keywords (comma-separated, or press Enter to skip): ").strip()
    filter_keywords = [k.strip() for k in filter_input.split(',') if k.strip()] if filter_input else None
    
    # Reject keywords
    print("   Reject papers containing certain keywords? (optional)")
    print("   Papers with these words will be excluded")
    print("   Example: review, survey, commentary")
    reject_input = input("   Reject keywords (comma-separated, or press Enter to skip): ").strip()
    reject_keywords = [k.strip() for k in reject_input.split(',') if k.strip()] if reject_input else None
    
    # Accept all option
    accept_all = False
    if not filter_keywords and not reject_keywords:
        print("   Accept all papers without filtering? [y/N]: ", end="")
        accept_input = input().strip().lower()
        accept_all = accept_input in ['y', 'yes']
    
    # Download options
    print("\n4. DOWNLOAD OPTIONS")
    print("   Download PDFs or just collect metadata?")
    print("   1) Download PDFs (default)")
    print("   2) Metadata only (faster)")
    download_choice = input("   Choice [1]: ").strip()
    skip_download = download_choice == '2'
    
    if not skip_download:
        print("   Number of parallel download workers?")
        print("   More workers = faster, but may trigger rate limits")
        print("   Recommendation: 3-6 workers")
        workers_input = input("   Workers [4]: ").strip()
        workers = int(workers_input) if workers_input else 4
        
        print("   Browser mode for handling challenges?")
        print("   Headed mode shows browser windows (for Turnstile/Cloudflare)")
        print("   Headless mode is faster but may fail on protected sites")
        print("   1) Headless (faster, default)")
        print("   2) Headed (visible browsers, handles challenges)")
        browser_choice = input("   Choice [1]: ").strip()
        headed = browser_choice == '2'
    else:
        workers = 3
        headed = False
    
    # Output options
    print("\n5. OUTPUT OPTIONS")
    default_output = f"./pdfs_{keywords[0].replace(' ', '_')}"
    output_input = input(f"   PDF output directory [{default_output}]: ").strip()
    output_dir = output_input if output_input and output_input.lower() not in ['yes', 'y'] else default_output
    
    default_csv = f"./results_{keywords[0].replace(' ', '_')}.csv"
    csv_input = input(f"   CSV results file [{default_csv}]: ").strip()
    csv_file = csv_input if csv_input and csv_input.lower() not in ['yes', 'y'] else default_csv
    
    # Summary
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Keywords: {', '.join(keywords)}")
    print(f"Max pages per keyword: {max_pages}")
    if filter_keywords:
        print(f"Filter keywords: {', '.join(filter_keywords)}")
    if reject_keywords:
        print(f"Reject keywords: {', '.join(reject_keywords)}")
    if accept_all:
        print("Accept all papers: Yes")
    print(f"Download PDFs: {'No (metadata only)' if skip_download else 'Yes'}")
    if not skip_download:
        print(f"Workers: {workers}")
        print(f"Browser mode: {'Headed (visible)' if headed else 'Headless'}")
    print(f"Output directory: {output_dir}")
    print(f"CSV file: {csv_file}")
    print("="*60)
    
    print("\nProceed with this configuration? [Y/n]: ", end="")
    confirm = input().strip().lower()
    if confirm in ['n', 'no']:
        print("Cancelled.")
        exit(0)
    
    # Create namespace object
    args = Namespace()
    args.keywords = keywords
    args.max_pages = max_pages
    args.email = "j.martinez.4823@gmail.com"
    args.filter_keywords = filter_keywords
    args.reject_keywords = reject_keywords
    args.accept_all = accept_all
    args.workers = workers
    args.headed = headed
    args.skip_download = skip_download
    args.output_dir = output_dir
    args.csv_file = csv_file
    args.concurrent_api = 3
    args.no_dedup = False
    
    return args

def main():
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced Unpaywall PDF Downloader - Search, classify, and download academic papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic search for tRNA papers
  %(prog)s --keywords "tRNA processing" "transfer RNA"
  
  # Visible browser mode for Turnstile challenges
  %(prog)s --keywords "ribosome" --headed
  
  # Large scale with custom settings
  %(prog)s --keywords "tRNA" "aminoacyl" --max-pages 5 --workers 8 --output-dir ./my_pdfs
  
  # Accept only papers with specific keywords
  %(prog)s --keywords "tRNA synthesis" --filter-keywords "human" "mouse"
        """
    )
    
    # Search parameters
    parser.add_argument('-k', '--keywords', nargs='+', 
                        help='Search keywords (space-separated, e.g., -k "tRNA processing" "ribosome")')
    parser.add_argument('-p', '--max-pages', type=int, default=1,
                        help='Maximum pages per search term (default: 1, ~20 papers per page)')
    parser.add_argument('-e', '--email', default="j.martinez.4823@gmail.com",
                        help='Email for Unpaywall API (default: j.martinez.4823@gmail.com)')
    
    # Classification parameters
    parser.add_argument('-f', '--filter-keywords', nargs='*',
                        help='Only accept papers containing these keywords in title')
    parser.add_argument('--reject-keywords', nargs='*',
                        help='Reject papers containing these keywords in title')
    parser.add_argument('--accept-all', action='store_true',
                        help='Accept all papers without classification')
    
    # Download parameters
    parser.add_argument('-w', '--workers', type=int, default=3,
                        help='Number of parallel download workers (default: 3)')
    parser.add_argument('--headed', action='store_true',
                        help='Run browsers in visible mode (for Turnstile challenges)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Only search and classify, skip PDF downloads')
    parser.add_argument('--download-pdfs', action='store_true',
                        help='Download PDFs (default is metadata only)')
    
    # Output parameters
    parser.add_argument('-o', '--output-dir', default='./pdfs',
                        help='Directory for downloaded PDFs (default: ./pdfs)')
    parser.add_argument('-c', '--csv-file', default='./results.csv',
                        help='Output CSV file path (default: ./results.csv)')
    
    # Performance parameters
    parser.add_argument('--concurrent-api', type=int, default=3,
                        help='Max concurrent API requests (default: 3)')
    parser.add_argument('--no-dedup', action='store_true',
                        help='Disable deduplication (download duplicates)')
    
    # Interactive mode
    parser.add_argument('--simple', action='store_true',
                        help='Interactive mode - asks for each option')
    
    args = parser.parse_args()
    
    # If simple mode, override args with interactive input
    if args.simple:
        args = interactive_mode()
    elif not args.keywords:
        parser.error("Keywords are required unless using --simple mode")
    
    # Create custom classifier based on filter keywords
    class CustomClassifier:
        def __init__(self, filter_keywords=None, reject_keywords=None, accept_all=False):
            self.filter_keywords = [k.lower() for k in (filter_keywords or [])]
            self.reject_keywords = [k.lower() for k in (reject_keywords or [])]
            self.accept_all = accept_all
            
        async def classify_paper(self, paper):
            if self.accept_all:
                return True, "Accepted all papers mode"
                
            title_lower = paper.title.lower()
            
            # Check reject keywords first
            for keyword in self.reject_keywords:
                if keyword in title_lower:
                    return False, f"Rejected - contains: {keyword}"
            
            # If filter keywords specified, must contain at least one
            if self.filter_keywords:
                for keyword in self.filter_keywords:
                    if keyword in title_lower:
                        return True, f"Accepted - contains: {keyword}"
                return False, "Does not contain required keywords"
            
            # Default: accept all if no filters
            return True, "No filters specified"
    
    # Run the system with CLI parameters
    asyncio.run(run_cli(args, CustomClassifier))

async def run_cli(args, CustomClassifier):
    """Run the system with CLI arguments"""
    print("="*60)
    print("ENHANCED UNPAYWALL PDF DOWNLOADER")
    print("="*60)
    print(f"Search keywords: {', '.join(args.keywords)}")
    print(f"Max pages per keyword: {args.max_pages}")
    print(f"Output directory: {args.output_dir}")
    print(f"CSV file: {args.csv_file}")
    print(f"Download workers: {args.workers}")
    print(f"Mode: {'HEADED (visible)' if args.headed else 'HEADLESS'}")
    
    if args.filter_keywords:
        print(f"Filter keywords: {', '.join(args.filter_keywords)}")
    if args.reject_keywords:
        print(f"Reject keywords: {', '.join(args.reject_keywords)}")
    print("="*60)
    
    # Initialize classifier
    classifier = CustomClassifier(
        filter_keywords=args.filter_keywords,
        reject_keywords=args.reject_keywords,
        accept_all=args.accept_all
    )
    
    start_time = time.time()
    
    # Phase 1: Search and classify
    print(f"\n=== PHASE 1: SEARCH & CLASSIFY ===")
    async with EnhancedUnpaywallExtractor(args.email, classifier, max_concurrent=args.concurrent_api) as extractor:
        all_papers = []
        
        for keyword in args.keywords:
            papers = await extractor.search_term_async(keyword, max_pages=args.max_pages)
            all_papers.extend(papers)
        
        # Apply deduplication unless disabled
        if not args.no_dedup:
            final_papers = extractor._merge_duplicate_papers(all_papers)
            print(f"Deduplication: {len(all_papers)} → {len(final_papers)} unique papers")
        else:
            final_papers = all_papers
        
        # Save metadata
        extractor._save_papers_to_csv(final_papers, args.csv_file)
    
    search_time = time.time() - start_time
    accepted_papers = [p for p in final_papers if p.classification_status == "accepted"]
    rejected_papers = [p for p in final_papers if p.classification_status == "rejected"]
    
    print(f"\nSearch complete in {search_time:.1f} seconds")
    print(f"Total papers: {len(final_papers)}")
    print(f"Accepted: {len(accepted_papers)}")
    print(f"Rejected: {len(rejected_papers)}")
    print(f"CSV saved: {args.csv_file}")
    
    # Phase 2: Download PDFs (if requested and not skipped)
    if args.download_pdfs and not args.skip_download and accepted_papers:
        print(f"\n=== PHASE 2: PDF DOWNLOAD ===")
        print(f"Downloading {len(accepted_papers)} papers with {args.workers} workers")
        
        if args.headed:
            print("\n⚠️  Running in HEADED mode - browser windows will be visible")
            print("When Turnstile appears, solve it in the browser window")
            print("The system will wait up to 5 minutes for you to solve challenges\n")
        
        downloader = SeleniumPDFDownloader(args.output_dir, headless=not args.headed)
        updated_papers = downloader.download_papers_from_metadata(final_papers, num_workers=args.workers)
        
        # Re-save CSV with download results
        extractor._save_papers_to_csv(updated_papers, args.csv_file)
        
        # Count successes
        successful_downloads = [p for p in updated_papers if p.download_status == "success"]
        failed_downloads = [p for p in updated_papers if p.download_status == "failed"]
        
        print(f"\nDownload success rate: {len(successful_downloads)}/{len(accepted_papers)} ({100*len(successful_downloads)/len(accepted_papers):.1f}%)")
    else:
        if args.skip_download:
            print("\n=== SKIPPING PDF DOWNLOAD (--skip-download) ===")
        else:
            print("\n=== NO ACCEPTED PAPERS TO DOWNLOAD ===")
    
    total_time = time.time() - start_time
    print(f"\n=== COMPLETE ===")
    print(f"Total execution time: {total_time:.1f} seconds")
    
    # Show sample results
    if final_papers:
        print(f"\n=== SAMPLE PAPERS ===")
        for i, paper in enumerate(final_papers[:5]):
            print(f"{i+1}. {paper.title[:70]}...")
            print(f"   DOI: {paper.doi}")
            print(f"   Status: {paper.classification_status} - {paper.classification_reason}")
            if paper.pdf_links:
                print(f"   PDF URLs: {len(paper.pdf_links)}")
            print(f"   Download: {paper.download_status}", end="")
            if paper.download_error:
                print(f" - {paper.download_error}")
            else:
                print()

if __name__ == "__main__":
    main()