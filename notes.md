### Website Content Scraper — Notes

This repository includes `scraper.py`, a command-line tool that scrapes webpage text and extracts word frequencies.

### What it does
- **Fetches pages**: Downloads HTML and removes scripts, styles, and common layout elements.
- **Extracts words**: Lowercases text and keeps only alphabetic words via regex.
- **Filters**: Drops words shorter than a chosen length and, by default, removes common stop words.
- **Analyzes**: Counts frequencies and reports top words, totals, and per-word percentages.
- **Saves CSV**: Writes detailed results or a simplified words-only CSV.

### Requirements
- Python 3.8+
- Packages: `requests`, `beautifulsoup4`
  - Install: `pip install requests beautifulsoup4`

### Quick start
```bash
python scraper.py https://example.com https://another-site.com
```
- Prints a summary and top words to the console.

### Key options
- `--min-length INT`: Minimum word length to include (default: 3).
- `--include-common`: Include common stop words (off by default).
- `--top-n INT`: Number of top words shown in the console summary (default: 50).
- `--output, -o FILE.csv`: Path for the CSV output (default: results.csv).
- `--words-only`: Save a simple CSV of `word, frequency, percentage`.
- `--top-only`: When saving CSV, include only the top N words.
- `--all-words`: Explicitly save all words to CSV (this is the default unless `--top-only` is used).

### Usage examples
- Basic scrape with defaults:
```bash
python scraper.py https://example.com
```

- Keep words with 4+ letters and show top 30 in console:
```bash
python scraper.py --min-length 4 --top-n 30 https://example.com
```

- Save comprehensive CSV with all sections to a custom file:
```bash
python scraper.py -o my_results.csv https://example.com https://another.com
```

- Save a simple words-only CSV (no summary sheets), top 100 only:
```bash
python scraper.py --words-only --top-only --top-n 100 -o simple_words.csv https://example.com
```

### What gets saved
- Comprehensive CSV (default):
  - Summary stats, successful/failed URLs, and a ranked word-frequency table with percentages.
- Words-only CSV (`--words-only`):
  - Columns: `word, frequency, percentage` sorted by frequency.

### Tips for better results
- Provide multiple, content-rich URLs to get a broader vocabulary.
- Increase `--min-length` to avoid short noise words.
- Use `--include-common` only if you specifically want function words included.
- Be respectful: the scraper sleeps briefly between successful requests; avoid hammering sites.
- Always check a site’s `robots.txt` and terms of service before scraping.

### Common troubleshooting
- Network errors: ensure the URL is valid and reachable; try again later.
- Empty output: site may use heavy client-side rendering; consider alternative pages.
- Encoding issues: the script reads server responses via `requests`; most UTF-8 pages work out of the box.

### Integrating with your word lists
- You can merge scraped words with curated lists using your existing pipeline (e.g., `combine_lists.py`) to build enriched vocabulary sets.

---

### Cleaner — Notes (`cleaner.py`)

`cleaner.py` removes English articles from the NGSL CSV and writes a cleaned file.

- **Input**: `NGSL_1.2_stats.csv` with a `Lemma` column.
- **Removes**: Articles `a`, `an`, `the` (case-insensitive; whitespace-trimmed).
- **Output**: `NGSL_1.2_stats_no_articles.csv` with the same columns as input, minus article rows.

#### How it works
- Reads the CSV into a DataFrame.
- Normalizes `Lemma` by trimming and lowercasing.
- Filters out rows whose normalized lemma is in `{"a", "an", "the"}`.
- Saves the remaining rows to a new CSV.

#### Run
```bash
python cleaner.py
```

#### Use with the word-combiner
- After running `cleaner.py`, `combine_lists.py` will prefer `NGSL_1.2_stats_no_articles.csv` and merge it with other lists (Oxford 5000, cities, general), deduplicate, remove compound words with spaces, and export `combined_unique_words.csv`.
