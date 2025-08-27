#!/usr/bin/env python3
"""
Website Content Scraper
A script to scrape content from websites and extract word frequency data.
"""

import requests
from bs4 import BeautifulSoup
import re
from collections import Counter
import argparse
import sys
from urllib.parse import urlparse
import time
import csv
from datetime import datetime

class WebsiteScraper:
    def __init__(self, min_word_length=3, exclude_common_words=True):
        """
        Initialize the scraper with configuration options.
        
        Args:
            min_word_length (int): Minimum length of words to include
            exclude_common_words (bool): Whether to exclude common stop words
        """
        self.min_word_length = min_word_length
        self.exclude_common_words = exclude_common_words
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Common stop words to exclude (if enabled)
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'what', 'where',
            'when', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'just', 'now', 'here', 'there', 'then', 'also', 'get', 'go', 'come', 'see',
            'know', 'take', 'give', 'use', 'find', 'tell', 'ask', 'work', 'seem', 'feel', 'try',
            'leave', 'call'
        }

    def is_valid_url(self, url):
        """Check if URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    def scrape_website(self, url):
        """
        Scrape content from a single website.
        
        Args:
            url (str): The URL to scrape
            
        Returns:
            str: Extracted text content or None if failed
        """
        if not self.is_valid_url(url):
            print(f"Invalid URL: {url}")
            return None
            
        try:
            print(f"Scraping: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except requests.RequestException as e:
            print(f"Error scraping {url}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error scraping {url}: {e}")
            return None

    def extract_words(self, text):
        """
        Extract and clean words from text.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of cleaned words
        """
        if not text:
            return []
        
        # Convert to lowercase and extract words using regex
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter words based on criteria
        filtered_words = []
        for word in words:
            # Check minimum length
            if len(word) < self.min_word_length:
                continue
                
            # Check if should exclude common words
            if self.exclude_common_words and word in self.stop_words:
                continue
                
            filtered_words.append(word)
        
        return filtered_words

    def analyze_websites(self, urls, top_n=50):
        """
        Analyze multiple websites and return word frequency data.
        
        Args:
            urls (list): List of URLs to analyze
            top_n (int): Number of top words to return
            
        Returns:
            dict: Analysis results
        """
        all_words = []
        successful_urls = []
        failed_urls = []
        
        for url in urls:
            # Add delay between requests to be respectful
            if successful_urls:
                time.sleep(1)
                
            text = self.scrape_website(url)
            if text:
                words = self.extract_words(text)
                all_words.extend(words)
                successful_urls.append(url)
                print(f"✓ Successfully scraped {url} - {len(words)} words extracted")
            else:
                failed_urls.append(url)
                print(f"✗ Failed to scrape {url}")
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        return {
            'total_words': len(all_words),
            'unique_words': len(word_counts),
            'successful_urls': successful_urls,
            'failed_urls': failed_urls,
            'top_words': word_counts.most_common(top_n),
            'all_word_counts': dict(word_counts)
        }

    def print_results(self, results):
        """Print analysis results in a formatted way."""
        print("\n" + "="*60)
        print("WEBSITE SCRAPING RESULTS")
        print("="*60)
        
        print(f"\nSUMMARY:")
        print(f"Total words extracted: {results['total_words']:,}")
        print(f"Unique words: {results['unique_words']:,}")
        print(f"Successful URLs: {len(results['successful_urls'])}")
        print(f"Failed URLs: {len(results['failed_urls'])}")
        
        if results['failed_urls']:
            print(f"\nFailed URLs:")
            for url in results['failed_urls']:
                print(f"  - {url}")
        
        print(f"\nTOP {len(results['top_words'])} MOST FREQUENT WORDS:")
        print("-" * 40)
        for i, (word, count) in enumerate(results['top_words'], 1):
            print(f"{i:2d}. {word:<20} ({count:,} times)")

    def save_to_csv(self, results, filename, include_all_words=True):
        """
        Save results to CSV file with multiple sheets of data.
        
        Args:
            results (dict): Analysis results
            filename (str): Output CSV filename
            include_all_words (bool): Whether to include all words or just top words
        """
        try:
            # Ensure filename ends with .csv
            if not filename.lower().endswith('.csv'):
                filename += '.csv'
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header information
                writer.writerow(['Website Scraping Results'])
                writer.writerow(['Generated on:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow([])  # Empty row
                
                # Write summary statistics
                writer.writerow(['SUMMARY STATISTICS'])
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Total Words Extracted', results['total_words']])
                writer.writerow(['Unique Words', results['unique_words']])
                writer.writerow(['Successful URLs', len(results['successful_urls'])])
                writer.writerow(['Failed URLs', len(results['failed_urls'])])
                writer.writerow([])  # Empty row
                
                # Write successful URLs
                if results['successful_urls']:
                    writer.writerow(['SUCCESSFUL URLs'])
                    writer.writerow(['URL'])
                    for url in results['successful_urls']:
                        writer.writerow([url])
                    writer.writerow([])  # Empty row
                
                # Write failed URLs
                if results['failed_urls']:
                    writer.writerow(['FAILED URLs'])
                    writer.writerow(['URL'])
                    for url in results['failed_urls']:
                        writer.writerow([url])
                    writer.writerow([])  # Empty row
                
                # Write word frequency data
                word_data = results['all_word_counts'] if include_all_words else dict(results['top_words'])
                writer.writerow(['WORD FREQUENCY DATA'])
                writer.writerow(['Rank', 'Word', 'Frequency', 'Percentage'])
                
                total_words = results['total_words']
                for i, (word, count) in enumerate(sorted(word_data.items(), key=lambda x: x[1], reverse=True), 1):
                    percentage = (count / total_words) * 100 if total_words > 0 else 0
                    writer.writerow([i, word, count, f'{percentage:.2f}%'])
            
            print(f"✓ Results saved to CSV: {filename}")
            return True
            
        except Exception as e:
            print(f"✗ Error saving to CSV: {e}")
            return False

    def save_words_only_csv(self, results, filename, include_all_words=True):
        """
        Save just the words and frequencies to a simple CSV file.
        
        Args:
            results (dict): Analysis results
            filename (str): Output CSV filename
            include_all_words (bool): Whether to include all words or just top words
        """
        try:
            # Ensure filename ends with .csv
            if not filename.lower().endswith('.csv'):
                filename += '.csv'
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(['word', 'frequency', 'percentage'])
                
                # Write word data
                word_data = results['all_word_counts'] if include_all_words else dict(results['top_words'])
                total_words = results['total_words']
                
                for word, count in sorted(word_data.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_words) * 100 if total_words > 0 else 0
                    writer.writerow([word, count, f'{percentage:.2f}'])
            
            print(f"✓ Word data saved to CSV: {filename}")
            return True
            
        except Exception as e:
            print(f"✗ Error saving word data to CSV: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Scrape websites and analyze word frequency')
    parser.add_argument('urls', nargs='+', help='URLs to scrape')
    parser.add_argument('--min-length', type=int, default=3, 
                        help='Minimum word length (default: 3)')
    parser.add_argument('--include-common', action='store_true', 
                        help='Include common stop words in results')
    parser.add_argument('--top-n', type=int, default=50, 
                        help='Number of top words to display (default: 50)')
    parser.add_argument('--output', '-o', type=str, 
                        help='Output CSV file to save results (default: results.csv)')
    parser.add_argument('--words-only', action='store_true',
                        help='Save only words and frequencies (simpler CSV format)')
    parser.add_argument('--top-only', action='store_true',
                        help='Save only top N words to CSV (default saves all words)')
    parser.add_argument('--all-words', action='store_true',
                        help='Explicitly save all words to CSV (this is now the default)')
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = WebsiteScraper(
        min_word_length=args.min_length,
        exclude_common_words=not args.include_common
    )
    
    # Analyze websites
    print("Starting website analysis...")
    results = scraper.analyze_websites(args.urls, args.top_n)
    
    # Print results
    scraper.print_results(results)
    
    # Save to CSV file
    output_file = args.output if args.output else 'results.csv'
    
    # Determine whether to save all words or just top N
    save_all_words = not args.top_only  # By default, save all words unless --top-only is specified
    
    if args.words_only:
        # Save simple words and frequencies CSV
        scraper.save_words_only_csv(results, output_file, save_all_words)
    else:
        # Save comprehensive CSV with all data
        scraper.save_to_csv(results, output_file, save_all_words)

if __name__ == "__main__":
    # Example usage when run directly
    if len(sys.argv) == 1:
        print("Example usage:")
        print("python scraper.py https://example.com https://another-site.com")
        print("python scraper.py --min-length 4 --top-n 30 https://example.com")
        print("python scraper.py --output my_results.csv https://example.com")
        print("python scraper.py --words-only --top-only https://example.com")
        print("python scraper.py --words-only -o simple_words.csv https://example.com")
        sys.exit(1)
    
    main()