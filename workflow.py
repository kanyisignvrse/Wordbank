import json
import re
import requests
import os
import time
import logging
from typing import List, Set, Dict, Tuple, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from collections import Counter
from dataclasses import dataclass
import google.generativeai as genai
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('word_collector.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class CrawlStats:
    """Statistics tracking for the crawling process."""
    pages_crawled: int = 0
    pages_failed: int = 0
    total_words_found: int = 0
    unique_words_found: int = 0
    links_discovered: int = 0

class Config:
    """Enhanced configuration class with more options and validation."""
    
    # Input Configuration
    SEED_URLS = ['[SEED_URL_HERE]']
    URL_FILE = None
    
    # Crawling Configuration
    MIN_UNIQUE_LINKS = 50
    MAX_CRAWL_DEPTH = 3
    STAY_ON_DOMAIN = True
    RESPECT_ROBOTS_TXT = True
    MAX_PAGES_PER_DOMAIN = 100  # Prevent overwhelming any single domain
    
    # Request Configuration
    REQUEST_TIMEOUT = 15
    MAX_RETRIES = 3
    RETRY_BACKOFF_FACTOR = 1
    USER_AGENT = 'WordCollector/1.0 (+https://example.com/contact)'
    
    # Content Filtering
    WORD_FREQUENCY_THRESHOLD = 2
    MIN_WORD_LENGTH = 2
    MAX_WORD_LENGTH = 50  # Exclude extremely long strings
    
    # Enhanced web noise terms
    WEB_NOISE_TERMS = {
        'menu', 'login', 'signup', 'copyright', 'home', 'click', 'search', 
        'page', 'content', 'privacy', 'terms', 'cookies', 'accept', 'close',
        'navigation', 'skip', 'main', 'footer', 'header', 'sidebar', 'breadcrumb',
        'toggle', 'button', 'link', 'image', 'video', 'audio', 'download',
        'share', 'social', 'follow', 'subscribe', 'newsletter', 'email',
        'phone', 'contact', 'address', 'location', 'map', 'directions'
    }
    
    # Common English stopwords (basic set)
    STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 
        'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 
        'the', 'to', 'was', 'were', 'will', 'with', 'the', 'i', 'you', 
        'they', 'we', 'she', 'my', 'your', 'his', 'her', 'our', 'their',
        'this', 'these', 'those', 'there', 'here', 'where', 'when', 'what',
        'who', 'why', 'how', 'all', 'any', 'some', 'can', 'could', 'would',
        'should', 'may', 'might', 'must', 'shall', 'do', 'does', 'did',
        'have', 'had', 'having', 'am', 'being', 'but', 'or', 'not', 'no',
        'if', 'then', 'than', 'so', 'very', 'just', 'now', 'only', 'also',
        'about', 'into', 'through', 'over', 'under', 'up', 'down', 'out',
        'off', 'above', 'below', 'between', 'among', 'during', 'before',
        'after', 'while', 'since', 'until', 'because', 'although', 'though'
    }
    
    # LLM Configuration
    LLM_FILTERING_ENABLED = True
    LLM_BATCH_SIZE = 1000  # Slightly smaller for better reliability
    GEMINI_API_KEY = '[API_KEY_HERE]'
    LLM_RATE_LIMIT_DELAY = 1  # Seconds between API calls
    
    # Output Configuration
    OUTPUT_FILE = 'cleaned_word_list.json'
    INCLUDE_FREQUENCY_COUNTS = True
    SAVE_RAW_WORDS = False  # Option to save pre-LLM filtered words
    
    # Performance
    USE_CACHING = True
    CACHE_SIZE_LIMIT = 1000  # Maximum cached pages
    
    @classmethod
    def validate(cls):
        """Validate configuration settings."""
        if cls.MIN_UNIQUE_LINKS <= 0:
            raise ValueError("MIN_UNIQUE_LINKS must be positive")
        if cls.MAX_CRAWL_DEPTH < 0:
            raise ValueError("MAX_CRAWL_DEPTH cannot be negative")
        if cls.LLM_FILTERING_ENABLED and not cls.GEMINI_API_KEY:
            logging.warning("LLM filtering enabled but no API key found. Disabling LLM filtering.")
            cls.LLM_FILTERING_ENABLED = False

class EnhancedWordCollector:
    """
    Enhanced version with better error handling, robots.txt support,
    improved filtering, and comprehensive logging.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.config.validate()
        
        # State tracking
        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.url_cache: Dict[str, str] = {}
        self.robots_cache: Dict[str, RobotFileParser] = {}
        self.domain_page_counts: Dict[str, int] = Counter()
        self.word_counts: Counter = Counter()
        self.stats = CrawlStats()
        
        # Setup components
        self.session = self._create_session()
        self.llm_model = None
        
        if self.config.LLM_FILTERING_ENABLED:
            self._setup_gemini()
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=self.config.MAX_RETRIES,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=self.config.RETRY_BACKOFF_FACTOR
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers
        session.headers.update({
            'User-Agent': self.config.USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive'
        })
        
        return session
    
    def _setup_gemini(self):
        """Configure Gemini API with error handling."""
        try:
            genai.configure(api_key=self.config.GEMINI_API_KEY)
            self.llm_model = genai.GenerativeModel('gemini-2.0-flash')
            logging.info("Gemini API configured successfully")
        except Exception as e:
            logging.error(f"Failed to configure Gemini API: {e}")
            self.config.LLM_FILTERING_ENABLED = False
    
    def load_seed_urls(self) -> List[str]:
        """Load and validate seed URLs."""
        if self.config.URL_FILE:
            try:
                with open(self.config.URL_FILE, 'r') as f:
                    if self.config.URL_FILE.endswith('.json'):
                        urls = json.load(f)
                    else:
                        urls = [line.strip() for line in f if line.strip()]
                logging.info(f"Loaded {len(urls)} URLs from {self.config.URL_FILE}")
                return urls
            except FileNotFoundError:
                logging.error(f"URL file not found: {self.config.URL_FILE}")
                return []
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON in URL file: {self.config.URL_FILE}")
                return []
        
        return self.config.SEED_URLS
    
    def _can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        if not self.config.RESPECT_ROBOTS_TXT:
            return True
        
        try:
            parsed_url = urlparse(url)
            domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            if domain not in self.robots_cache:
                rp = RobotFileParser()
                rp.set_url(urljoin(domain, '/robots.txt'))
                try:
                    rp.read()
                    self.robots_cache[domain] = rp
                except Exception:
                    # If robots.txt can't be read, assume crawling is allowed
                    logging.debug(f"Could not read robots.txt for {domain}")
                    return True
            
            return self.robots_cache[domain].can_fetch(self.config.USER_AGENT, url)
        except Exception as e:
            logging.debug(f"Error checking robots.txt for {url}: {e}")
            return True
    
    def crawl_and_extract(self):
        """Enhanced crawling with better error handling and domain limits."""
        seed_urls = self.load_seed_urls()
        if not seed_urls:
            logging.error("No valid seed URLs found")
            return
        
        # Initialize queue with (url, depth) tuples
        urls_to_visit = [(url, 0) for url in seed_urls]
        processed_count = 0
        
        logging.info(f"Starting crawl with {len(seed_urls)} seed URLs")
        
        while urls_to_visit and processed_count < self.config.MIN_UNIQUE_LINKS:
            url, depth = urls_to_visit.pop(0)
            
            # Skip if already processed or depth exceeded
            if (url in self.visited_urls or 
                url in self.failed_urls or 
                depth > self.config.MAX_CRAWL_DEPTH):
                continue
            
            # Check domain page limit
            domain = urlparse(url).netloc
            if self.domain_page_counts[domain] >= self.config.MAX_PAGES_PER_DOMAIN:
                logging.debug(f"Domain page limit reached for {domain}")
                continue
            
            # Check robots.txt
            if not self._can_fetch(url):
                logging.debug(f"Robots.txt disallows crawling {url}")
                continue
            
            # Fetch and process page
            if self._process_page(url, depth, urls_to_visit):
                processed_count += 1
                self.domain_page_counts[domain] += 1
                
                if processed_count % 10 == 0:
                    logging.info(f"Processed {processed_count} pages, found {len(self.word_counts)} unique words")
        
        logging.info(f"Crawling complete. Stats: {self._get_stats_summary()}")
    
    def _process_page(self, url: str, depth: int, urls_to_visit: List[Tuple[str, int]]) -> bool:
        """Process a single page and return success status."""
        try:
            html_content = self._fetch_html(url)
            if not html_content:
                return False
            
            self.visited_urls.add(url)
            self.stats.pages_crawled += 1
            
            # Extract and process text
            text = self._extract_text(html_content)
            words = self._clean_and_tokenize(text)
            filtered_words = self._initial_filter(words)
            
            self.word_counts.update(filtered_words)
            self.stats.total_words_found += len(words)
            self.stats.unique_words_found = len(self.word_counts)
            
            # Extract links for further crawling
            if depth < self.config.MAX_CRAWL_DEPTH:
                new_links = self._extract_links(html_content, url)
                for link in new_links:
                    if link not in self.visited_urls and link not in self.failed_urls:
                        urls_to_visit.append((link, depth + 1))
                        self.stats.links_discovered += 1
            
            return True
            
        except Exception as e:
            logging.error(f"Error processing {url}: {e}")
            self.failed_urls.add(url)
            self.stats.pages_failed += 1
            return False
    
    def _fetch_html(self, url: str) -> Optional[str]:
        """Fetch HTML with caching and comprehensive error handling."""
        # Check cache first
        if self.config.USE_CACHING and url in self.url_cache:
            return self.url_cache[url]
        
        try:
            response = self.session.get(url, timeout=self.config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                logging.debug(f"Skipping non-HTML content: {url}")
                return None
            
            # Check content length
            if len(response.text) < 100:  # Skip very short pages
                logging.debug(f"Skipping short content: {url}")
                return None
            
            # Cache if enabled and within limits
            if self.config.USE_CACHING and len(self.url_cache) < self.config.CACHE_SIZE_LIMIT:
                self.url_cache[url] = response.text
            
            return response.text
            
        except requests.exceptions.RequestException as e:
            logging.debug(f"Request failed for {url}: {e}")
            return None
    
    def _extract_links(self, html_content: str, base_url: str) -> List[str]:
        """Extract and filter links with better validation."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            links = set()
            base_domain = urlparse(base_url).netloc
            
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href'].strip()
                
                # Skip empty, javascript, or anchor links
                if not href or href.startswith(('javascript:', 'mailto:', 'tel:', '#')):
                    continue
                
                # Resolve relative URLs
                full_url = urljoin(base_url, href)
                
                # Remove fragment
                full_url = full_url.split('#')[0]
                
                # Validate URL structure
                parsed = urlparse(full_url)
                if not parsed.scheme or not parsed.netloc:
                    continue
                
                # Apply domain restrictions
                if self.config.STAY_ON_DOMAIN and parsed.netloc != base_domain:
                    continue
                
                # Skip common non-content URLs
                if any(skip in full_url.lower() for skip in [
                    '/login', '/signup', '/register', '/logout', '/admin',
                    '.pdf', '.doc', '.zip', '.exe', '/download',
                    '/api/', '/rss', '/feed'
                ]):
                    continue
                
                links.add(full_url)
            
            return list(links)
            
        except Exception as e:
            logging.debug(f"Error extracting links from {base_url}: {e}")
            return []
    
    def _extract_text(self, html_content: str) -> str:
        """Enhanced text extraction with better content filtering."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                               'aside', 'noscript', 'iframe', 'form']):
                element.decompose()
            
            # Focus on main content areas
            main_content = (
                soup.find('main') or 
                soup.find('article') or 
                soup.find('div', {'class': re.compile(r'content|main|article', re.I)}) or
                soup.find('body') or
                soup
            )
            
            text = main_content.get_text(separator=' ', strip=True)
            
            # Basic cleaning
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            text = re.sub(r'[^\w\s-]', ' ', text)  # Keep only alphanumeric, spaces, hyphens
            
            return text
            
        except Exception as e:
            logging.debug(f"Error extracting text: {e}")
            return ""
    
    def _clean_and_tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization with better cleaning."""
        if not text:
            return []
        
        # Convert to lowercase and split
        words = text.lower().split()
        
        # Clean individual words
        cleaned_words = []
        for word in words:
            # Remove leading/trailing non-alphanumeric characters
            word = re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', word)
            
            # Skip if empty after cleaning
            if not word:
                continue
                
            # Skip if contains numbers or special characters
            if re.search(r'[0-9]', word) or not word.isalpha():
                continue
                
            # Check length constraints
            if (self.config.MIN_WORD_LENGTH <= len(word) <= self.config.MAX_WORD_LENGTH):
                cleaned_words.append(word)
        
        return cleaned_words
    
    def _initial_filter(self, words: List[str]) -> List[str]:
        """Enhanced initial filtering with stopwords and web noise removal."""
        return [
            word for word in words
            if (word not in self.config.STOPWORDS and
                word not in self.config.WEB_NOISE_TERMS and
                len(set(word)) > 1 and  # Skip words with repeated single character
                not word.startswith(('www', 'http')))  # Skip URL fragments
        ]
    
    def _llm_filter_batch(self, word_batch: List[str]) -> List[str]:
        """Filter a batch of words using LLM with improved prompt."""
        if not self.llm_model:
            return word_batch
        
        # Enhanced prompt for better filtering
        prompt = f"""
You are an expert English vocabulary curator. From the following word list, return ONLY common English vocabulary words that meet ALL these criteria:

INCLUDE:
- Common nouns, adjectives, adverbs used in everyday English
- Words that would appear in a standard English dictionary
- Words appropriate for general conversation and writing
- Words relevant to the Kenyan context
- Lemmatized words for verbs

EXCLUDE:
- Function words (the, a, is, in, of, for, with, etc.)
- Proper nouns (names of people, places, brands)
- Technical jargon, specialized terminology
- Archaic or obsolete words
- Internet slang, abbreviations, acronyms
- Non-English words
- Misspelled words
- Words shorter than 3 characters
- Words irrelevant to the Kenyan context

Return your response as a single line of space-separated words with no other text or explanation.

Word list to filter:
{' '.join(word_batch)}
"""
        
        try:
            response = self.llm_model.generate_content(prompt)
            if response.text:
                filtered_words = response.text.strip().split()
                # Additional validation of returned words
                return [w for w in filtered_words if w.isalpha() and len(w) >= 3]
            return []
            
        except Exception as e:
            logging.error(f"LLM filtering error: {e}")
            return word_batch  # Return original batch on error
    
    def llm_based_filtering(self, word_list: List[str]) -> List[str]:
        """Enhanced LLM filtering with rate limiting and progress tracking."""
        if not self.config.LLM_FILTERING_ENABLED or not self.llm_model:
            logging.info("Skipping LLM-based filtering")
            return word_list
        
        logging.info(f"Starting LLM filtering for {len(word_list)} words...")
        final_words = []
        batch_count = (len(word_list) + self.config.LLM_BATCH_SIZE - 1) // self.config.LLM_BATCH_SIZE
        
        for i in range(0, len(word_list), self.config.LLM_BATCH_SIZE):
            batch_num = i // self.config.LLM_BATCH_SIZE + 1
            batch = word_list[i:i + self.config.LLM_BATCH_SIZE]
            
            logging.info(f"Processing batch {batch_num}/{batch_count} ({len(batch)} words)")
            
            filtered_batch = self._llm_filter_batch(batch)
            final_words.extend(filtered_batch)
            
            logging.info(f"Batch {batch_num} complete: {len(filtered_batch)} words retained")
            
            # Rate limiting
            if i + self.config.LLM_BATCH_SIZE < len(word_list):
                time.sleep(self.config.LLM_RATE_LIMIT_DELAY)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_final_words = []
        for word in final_words:
            if word not in seen:
                seen.add(word)
                unique_final_words.append(word)
        
        logging.info(f"LLM filtering complete: {len(unique_final_words)} final words")
        return unique_final_words
    
    def _get_stats_summary(self) -> str:
        """Generate a summary of crawling statistics."""
        return (f"Pages crawled: {self.stats.pages_crawled}, "
                f"Failed: {self.stats.pages_failed}, "
                f"Total words: {self.stats.total_words_found}, "
                f"Unique words: {self.stats.unique_words_found}")
    
    def run(self) -> List[str]:
        """Execute the complete workflow with comprehensive logging."""
        start_time = time.time()
        logging.info("Starting word collection workflow")
        
        try:
            # Step 1: Crawl and extract
            self.crawl_and_extract()
            
            if not self.word_counts:
                logging.error("No words collected from crawling")
                return []
            
            # Step 2: Apply frequency threshold
            if self.config.WORD_FREQUENCY_THRESHOLD > 1:
                pre_llm_words = [
                    word for word, count in self.word_counts.items()
                    if count >= self.config.WORD_FREQUENCY_THRESHOLD
                ]
                logging.info(f"Applied frequency threshold: {len(pre_llm_words)} words remain")
            else:
                pre_llm_words = list(self.word_counts.keys())
            
            pre_llm_words.sort()
            
            # Save raw words if requested
            if self.config.SAVE_RAW_WORDS:
                raw_output_file = self.config.OUTPUT_FILE.replace('.json', '_raw.json')
                self._save_word_list(pre_llm_words, raw_output_file, include_counts=True)
            
            # Step 3: LLM filtering
            final_word_list = self.llm_based_filtering(pre_llm_words)
            
            # Step 4: Save results
            self.save_output(final_word_list)
            
            elapsed_time = time.time() - start_time
            logging.info(f"Workflow completed in {elapsed_time:.1f} seconds")
            logging.info(f"Final word count: {len(final_word_list)}")
            
            return final_word_list
            
        except Exception as e:
            logging.error(f"Workflow failed: {e}")
            raise
    
    def _save_word_list(self, word_list: List[str], filename: str, include_counts: bool = False):
        """Helper method to save word lists."""
        if include_counts:
            output_data = {
                word: self.word_counts.get(word, 0) 
                for word in word_list
            }
        else:
            output_data = word_list
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    def save_output(self, word_list: List[str]):
        """Save the final results with metadata."""
        try:
            if self.config.INCLUDE_FREQUENCY_COUNTS:
                # Create detailed output with metadata
                output_data = {
                    "metadata": {
                        "total_words": len(word_list),
                        "pages_crawled": self.stats.pages_crawled,
                        "pages_failed": self.stats.pages_failed,
                        "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "config": {
                            "min_unique_links": self.config.MIN_UNIQUE_LINKS,
                            "max_crawl_depth": self.config.MAX_CRAWL_DEPTH,
                            "word_frequency_threshold": self.config.WORD_FREQUENCY_THRESHOLD,
                            "llm_filtering_enabled": self.config.LLM_FILTERING_ENABLED
                        }
                    },
                    "words": {
                        word: self.word_counts.get(word, 0) 
                        for word in word_list
                    }
                }
            else:
                output_data = {
                    "metadata": {
                        "total_words": len(word_list),
                        "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    },
                    "words": word_list
                }
            
            with open(self.config.OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
            logging.info(f"✅ Main results saved to: {self.config.OUTPUT_FILE}")
            print(f"✅ Results saved to: {self.config.OUTPUT_FILE}")
            
        except IOError as e:
            logging.error(f"❌ Error saving output file: {e}")
            raise

def main():
    """Main execution function with example usage."""
    # Check for required API key
    if not os.getenv('GEMINI_API_KEY'):
        logging.error("GEMINI_API_KEY environment variable not set")
        logging.info("Set it with: export GEMINI_API_KEY='your-api-key-here'")
        return
    
    # Example configuration
    Config.SEED_URLS = [
        'https://www.bbc.com/news',
        'https://www.reuters.com',
        'https://en.wikipedia.org/wiki/Main_Page',
        'https://kenyanwallstreet.com/',
        'https://www.nation.co.ke/',
        'https://www.businessdailyafrica.com/',
        'https://www.standardmedia.co.ke/',
        'https://www.the-star.co.ke/',
        'https://www.kenyans.co.ke/',
        'https://www.ksldictionary.com/',
        'https://doorinternational.org/'
    ]
    Config.MIN_UNIQUE_LINKS = 100  # Reduced for demonstration
    Config.MAX_CRAWL_DEPTH = 4
    Config.STAY_ON_DOMAIN = False  # Allow cross-domain crawling
    Config.LLM_FILTERING_ENABLED = True
    Config.WORD_FREQUENCY_THRESHOLD = 1
    Config.OUTPUT_FILE = 'enhanced_word_list.json'
    Config.SAVE_RAW_WORDS = True  # Save intermediate results
    
    try:
        collector = EnhancedWordCollector(Config)
        final_words = collector.run()
        
        if final_words:
            print(f"\n{'='*50}")
            print("WORD COLLECTION COMPLETE")
            print(f"{'='*50}")
            print(f"Final word count: {len(final_words)}")
            print(f"Sample words: {final_words[:20]}")
            print(f"Results saved to: {Config.OUTPUT_FILE}")
        else:
            print("No words were collected. Check the logs for issues.")
            
    except Exception as e:
        logging.error(f"Application failed: {e}")
        raise

if __name__ == '__main__':
    main()




# import json
# import re
# import requests
# import os
# import google.generativeai as genai
# from bs4 import BeautifulSoup
# from collections import Counter
# from urllib.parse import urljoin, urlparse

# # --- Configuration ---
# class Config:
#     """Configuration class for the web crawler and word filter."""
#     # Input: Can be a list of URLs or a path to a JSON/text file.
#     SEED_URLS = ['http://example.com']
#     # Set to a file path to load URLs from a file (e.g., 'urls.json' or 'urls.txt').
#     URL_FILE = None

#     # Crawling Configuration
#     MIN_UNIQUE_LINKS = 50
#     MAX_CRAWL_DEPTH = 3
#     # To respect domain boundaries. If False, it can crawl external links.
#     STAY_ON_DOMAIN = True

#     # Filtering Configuration
#     WORD_FREQUENCY_THRESHOLD = 5  # Optional: Minimum number of times a word must appear.
#     MIN_WORD_LENGTH = 3 # Increased slightly to pre-filter more noise
#     # A custom list of high-frequency web noise to remove.
#     WEB_NOISE_TERMS = {'menu', 'login', 'signup', 'copyright', 'home', 'click', 'search', 'page', 'content', 'privacy', 'terms'}

#     # LLM Filtering (Gemini)
#     LLM_FILTERING_ENABLED = True
#     LLM_BATCH_SIZE = 200 # Gemini can handle larger batches
#     # Securely get the API key from an environment variable
#     GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

#     # Output Configuration
#     OUTPUT_FILE = 'cleaned_word_list_gemini_no_nltk.json'
#     INCLUDE_FREQUENCY_COUNTS = True

#     # Optimization
#     REQUEST_TIMEOUT = 10  # seconds
#     USE_CACHING = True

# class WordCollector:
#     """
#     An end-to-end workflow for collecting and filtering commonly used English words
#     using an LLM for all advanced filtering tasks.
#     """
#     def __init__(self, config):
#         self.config = config
#         self.visited_urls = set()
#         self.url_cache = {}
#         self.word_counts = Counter()
#         self.llm_model = None

#         if self.config.LLM_FILTERING_ENABLED:
#             self.setup_gemini()

#     def setup_gemini(self):
#         """Configures the Gemini API client."""
#         if not self.config.GEMINI_API_KEY:
#             raise ValueError("GEMINI_API_KEY environment variable not set.")
#         try:
#             genai.configure(api_key=self.config.GEMINI_API_KEY)
#             self.llm_model = genai.GenerativeModel('gemini-pro')
#             print("Gemini API configured successfully.")
#         except Exception as e:
#             print(f"Error configuring Gemini API: {e}")
#             self.config.LLM_FILTERING_ENABLED = False

#     def load_seed_urls(self):
#         """Loads seed URLs from the config file or list."""
#         if self.config.URL_FILE:
#             try:
#                 with open(self.config.URL_FILE, 'r') as f:
#                     if self.config.URL_FILE.endswith('.json'):
#                         return json.load(f)
#                     else:
#                         return [line.strip() for line in f if line.strip()]
#             except FileNotFoundError:
#                 print(f"Error: URL file not found at {self.config.URL_FILE}")
#                 return []
#         return self.config.SEED_URLS

#     def crawl_and_extract(self):
#         """
#         Manages the crawling and text extraction process.
#         """
#         seed_urls = self.load_seed_urls()
#         urls_to_visit = [(url, 0) for url in seed_urls]
#         processed_links = 0

#         while urls_to_visit and processed_links < self.config.MIN_UNIQUE_LINKS:
#             url, depth = urls_to_visit.pop(0)

#             if url in self.visited_urls or depth > self.config.MAX_CRAWL_DEPTH:
#                 continue

#             html_content = self.fetch_html(url)
#             if html_content:
#                 self.visited_urls.add(url)
#                 processed_links += 1

#                 # Text Extraction and Initial Filtering
#                 text = self.extract_text(html_content)
#                 words = self.clean_and_tokenize(text)
#                 filtered_words = self.initial_filter(words)
#                 self.word_counts.update(filtered_words)

#                 # Extract and add new links to the queue
#                 new_links = self.extract_links(html_content, url)
#                 for link in new_links:
#                     if link not in self.visited_urls:
#                         urls_to_visit.append((link, depth + 1))
        
#         print(f"Crawled {len(self.visited_urls)} unique pages.")

#     def fetch_html(self, url):
#         """
#         Fetches HTML content from a URL with caching and error handling.
#         """
#         if self.config.USE_CACHING and url in self.url_cache:
#             return self.url_cache[url]

#         try:
#             response = requests.get(url, timeout=self.config.REQUEST_TIMEOUT)
#             response.raise_for_status()
#             if 'text/html' in response.headers.get('Content-Type', ''):
#                 if self.config.USE_CACHING:
#                     self.url_cache[url] = response.text
#                 return response.text
#             return None
#         except requests.exceptions.RequestException as e:
#             print(f"Error fetching {url}: {e}")
#             return None

#     def extract_links(self, html_content, base_url):
#         """
#         Extracts and normalizes links from HTML content.
#         """
#         soup = BeautifulSoup(html_content, 'html.parser')
#         links = set()
#         base_domain = urlparse(base_url).netloc

#         for a_tag in soup.find_all('a', href=True):
#             href = a_tag['href']
#             full_url = urljoin(base_url, href).split('#')[0]

#             if self.config.STAY_ON_DOMAIN:
#                 if urlparse(full_url).netloc == base_domain:
#                     links.add(full_url)
#             else:
#                 links.add(full_url)
#         return list(links)

#     def extract_text(self, html_content):
#         """
#         Extracts visible text from HTML, ignoring scripts and styles.
#         """
#         soup = BeautifulSoup(html_content, 'html.parser')
#         for script_or_style in soup(['script', 'style']):
#             script_or_style.decompose()
#         return soup.get_text()

#     def clean_and_tokenize(self, text):
#         """
#         Cleans text by removing punctuation, numbers, and converting to lowercase.
#         """
#         text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
#         return text.split()

#     def initial_filter(self, words):
#         """
#         Performs minimal initial filtering. LLM will handle the rest.
#         """
#         return [
#             word for word in words
#             if len(word) >= self.config.MIN_WORD_LENGTH and
#                word not in self.config.WEB_NOISE_TERMS
#         ]

#     def llm_based_filtering(self, word_list):
#         """
#         Sends a list of words to the Gemini API for all advanced filtering.
#         """
#         if not self.config.LLM_FILTERING_ENABLED or not self.llm_model:
#             print("Skipping LLM-based filtering.")
#             return word_list

#         print(f"Starting LLM-based filtering with Gemini for {len(word_list)} words...")
#         final_words = []
#         for i in range(0, len(word_list), self.config.LLM_BATCH_SIZE):
#             batch = word_list[i:i + self.config.LLM_BATCH_SIZE]
            
#             # Enhanced prompt to explicitly handle stopwords
#             prompt = (
#                 "You are an English vocabulary expert. Your task is to filter the following list "
#                 "and return ONLY the common English vocabulary words (nouns, verbs, adjectives, adverbs). "
#                 "Exclude all of the following: "
#                 "1. Function words (e.g., 'the', 'a', 'is', 'in', 'of', 'for'). "
#                 "2. Proper nouns (e.g., 'Google', 'London'). "
#                 "3. Non-English words. "
#                 "4. Highly technical, archaic, or rare words. "
#                 "5. Internet slang or acronyms. "
#                 "Return the result as a single line of space-separated words, and nothing else. "
#                 "For example, if the input is 'apple run beautiful quickly the schadenfreude wikipedia', "
#                 "the output should be 'apple run beautiful quickly'.\n\n"
#                 "Here is the word list:\n"
#                 + " ".join(batch)
#             )
            
#             try:
#                 response = self.llm_model.generate_content(prompt)
#                 cleaned_batch = response.text.strip().split()
#                 final_words.extend(cleaned_batch)
#                 print(f"  Processed batch {i // self.config.LLM_BATCH_SIZE + 1}, received {len(cleaned_batch)} words.")
#             except Exception as e:
#                 print(f"  An error occurred with the Gemini API on a batch: {e}")
#                 final_words.extend(batch) # Fallback

#         print("LLM-based filtering complete.")
#         return final_words

#     def run(self):
#         """
#         Executes the entire workflow.
#         """
#         self.crawl_and_extract()

#         if self.config.WORD_FREQUENCY_THRESHOLD > 1:
#             pre_llm_words = [
#                 word for word, count in self.word_counts.items()
#                 if count >= self.config.WORD_FREQUENCY_THRESHOLD
#             ]
#         else:
#             pre_llm_words = list(self.word_counts.keys())
        
#         pre_llm_words.sort()

#         final_word_list = self.llm_based_filtering(pre_llm_words)
#         self.save_output(final_word_list)
#         return final_word_list

#     def save_output(self, word_list):
#         """
#         Saves the final word list to a file.
#         """
#         if self.config.INCLUDE_FREQUENCY_COUNTS:
#             output_data = {word: self.word_counts[word] for word in word_list if word in self.word_counts}
#         else:
#             output_data = word_list

#         try:
#             with open(self.config.OUTPUT_FILE, 'w') as f:
#                 json.dump(output_data, f, indent=4)
#             print(f"Successfully saved {len(word_list)} words to {self.config.OUTPUT_FILE}")
#         except IOError as e:
#             print(f"Error saving output file: {e}")

# # --- Execution ---
# if __name__ == '__main__':
#     # Ensure you have set the GEMINI_API_KEY environment variable first.
#     if not os.getenv('GEMINI_API_KEY'):
#         print("FATAL: The 'GEMINI_API_KEY' environment variable is not set.")
#         print("Please set it before running the script.")
#     else:
#         # Configuration for the run
#         Config.SEED_URLS = [
#             'https://www.citizen.digital/news',
#             'https://www.tuko.co.ke/'
#         ]
#         Config.MIN_UNIQUE_LINKS = 20  # For a quicker demonstration
#         Config.STAY_ON_DOMAIN = True # Keeps the crawl focused on wikipedia.org
#         Config.LLM_FILTERING_ENABLED = True

#         collector = WordCollector(Config)
#         final_words = collector.run()
#         print("\n--- Final Cleaned Word List (Sample) ---")
#         print(final_words[:100])