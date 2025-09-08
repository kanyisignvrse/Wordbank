## WordBank Crawler and Word Collector

This project crawls the web starting from a set of seed URLs, extracts visible text, cleans and filters words, and optionally uses an LLM (Gemini) to curate a final English word list. The main entry point is `workflow.py` which implements an enhanced, configurable crawler and processing pipeline.

### Key Features
- Crawl with depth and per-domain limits, respecting `robots.txt` (configurable)
- Extract and clean visible page text, remove web-noise and stopwords
- Deduplicate and count frequencies
- Optional LLM-based filtering via Gemini to curate general English vocabulary
- Comprehensive logging and structured JSON outputs

### Project Structure
- `workflow.py`: Main crawler and processing pipeline
- `enhanced_word_list.json`: Final output (words + metadata by default)
- `enhanced_word_list_raw.json`: Optional intermediate words (pre-LLM) if enabled
- `word_collector.log`: Run-time logs
- Other data files: Source wordlists and samples used by the project

### Requirements
- Python 3.9+
- Internet access for crawling and (optionally) Gemini API

Install dependencies (standard library plus `requests`, `beautifulsoup4`, `google-generativeai`, `urllib3`):
```bash
pip install requests beautifulsoup4 google-generativeai urllib3
```

### Running the workflow
From the project directory:
```bash
python workflow.py
```

On Windows PowerShell, if you plan to use LLM filtering, set your Gemini API key first:
```powershell
# Persist for future sessions
setx GEMINI_API_KEY "YOUR_API_KEY"

# Or set for the current session only
$Env:GEMINI_API_KEY = "YOUR_API_KEY"
```

Notes:
- `workflow.py` logs to `word_collector.log` and console.
- The script defines seed URLs and config inside `main()`; adjust as needed (see Configuration). 

### Configuration Overview
Configuration lives in the `Config` class in `workflow.py`. You can edit values directly or set them in `main()` before creating the `EnhancedWordCollector` instance.

- Seed URLs
  - `SEED_URLS`: List of starting pages.
  - `URL_FILE`: Optional path to `.json` (array of URLs) or `.txt` (one URL per line). If set, it overrides `SEED_URLS`.

- Crawl behavior
  - `MIN_UNIQUE_LINKS` (int): Target number of successfully processed pages before stopping.
  - `MAX_CRAWL_DEPTH` (int): Maximum link depth from a seed URL.
  - `STAY_ON_DOMAIN` (bool): If true, only crawl within the current page’s domain.
  - `RESPECT_ROBOTS_TXT` (bool): If true, consult `robots.txt` per domain.
  - `MAX_PAGES_PER_DOMAIN` (int): Hard cap per domain to avoid over-focusing. When cross-domain crawling is enabled, the crawler prioritizes external links first to diversify domains.

- Request settings
  - `REQUEST_TIMEOUT`, `MAX_RETRIES`, `RETRY_BACKOFF_FACTOR`, `USER_AGENT`

- Content filtering (pre-LLM)
  - `WORD_FREQUENCY_THRESHOLD`: Keep words appearing at least this many times
  - `MIN_WORD_LENGTH`, `MAX_WORD_LENGTH`
  - `WEB_NOISE_TERMS`, `STOPWORDS`

- LLM filtering (Gemini)
  - `LLM_FILTERING_ENABLED` (bool)
  - `LLM_BATCH_SIZE` (int)
  - `GEMINI_API_KEY`: Read from environment; set via `GEMINI_API_KEY` env var
  - `LLM_RATE_LIMIT_DELAY` (seconds between calls)

- Output
  - `OUTPUT_FILE`: Final results path (default `enhanced_word_list.json`)
  - `INCLUDE_FREQUENCY_COUNTS` (bool): Save counts in the final file
  - `SAVE_RAW_WORDS` (bool): Also save pre-LLM words to `*_raw.json`

- Performance
  - `USE_CACHING` (bool), `CACHE_SIZE_LIMIT` (int)

### Typical adjustments
- Crawl more pages:
  - Increase `Config.MIN_UNIQUE_LINKS` (e.g., 1000+)
  - Ensure `Config.MAX_CRAWL_DEPTH` is sufficient (e.g., 4–6)

- Avoid a single domain dominating:
  - Set `Config.STAY_ON_DOMAIN = False`
  - Lower `Config.MAX_PAGES_PER_DOMAIN` (e.g., 10–50)

- Focus on a specific site:
  - Set `Config.STAY_ON_DOMAIN = True` and use a seed on that domain

- Use your own seed list file:
  - Place a file, e.g., `seeds.json` with `["https://example.com", "https://another.com"]`
  - Set `Config.URL_FILE = 'seeds.json'`

### Outputs
By default, the script writes a structured JSON with metadata:
- `enhanced_word_list.json`: Contains metadata (pages crawled, failures, config snapshot) and the final word list (with counts if enabled).
- If `SAVE_RAW_WORDS = True`, it also writes `enhanced_word_list_raw.json` with pre-LLM words and counts.

### Logging
- All run-time info is written to `word_collector.log` and echoed to the console.
- Look here for crawl progress, page processing stats, and any errors.

### Troubleshooting
- Not crawling cross-domain:
  - Set `Config.STAY_ON_DOMAIN = False`
  - Check `RESPECT_ROBOTS_TXT`; some domains may block aggressive crawling

- Stops around a specific number of pages:
  - Increase `MIN_UNIQUE_LINKS`
  - Check `MAX_PAGES_PER_DOMAIN` and raise it if too restrictive

- Few or no words collected:
  - Verify pages are `text/html` and not blocked by paywalls or scripts
  - Reduce `WORD_FREQUENCY_THRESHOLD`
  - Ensure `MIN_WORD_LENGTH` isn’t too high

- LLM filtering not running:
  - Ensure `LLM_FILTERING_ENABLED = True`
  - Set `GEMINI_API_KEY` in your environment

### Minimal example (edit inside `main()`)
Inside `workflow.py`, you can quickly configure a run:
```python
Config.SEED_URLS = [
    'https://www.bbc.com/news',
    'https://www.reuters.com',
]
Config.MIN_UNIQUE_LINKS = 500
Config.MAX_CRAWL_DEPTH = 5
Config.STAY_ON_DOMAIN = False
Config.MAX_PAGES_PER_DOMAIN = 25
Config.LLM_FILTERING_ENABLED = True
Config.OUTPUT_FILE = 'enhanced_word_list.json'
```
Then run:
```bash
python workflow.py
```

### License
For internal or personal use unless otherwise specified.



