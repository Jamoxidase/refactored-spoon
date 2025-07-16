# Unpaywall PDF Downloader

Downloads academic papers from Unpaywall API (key word search) with Selenium. Claude helped to combine a couple of scripts, seems to work as intended.

## Quick Start

```bash
# Recommended: Interactive mode
python3 unpaywall_system.py --simple
```

This walks you through all options interactively.

## Command Line Usage

```bash
python3 unpaywall_system.py -k "tRNA" "ribosome" --max-pages 2 --download-pdfs
```

## Requirements

- Python 3.7+
- Chrome browser
- `pip install aiohttp selenium requests`

## Key Options

- `--simple`: Interactive mode (recommended for first use)
- `-k`: Search keywords (each keyword searched separately)
- `--max-pages N`: Pages per keyword (1 page ≈ 20 papers)
- `--download-pdfs`: Actually download PDFs (default: metadata only)
- `--headed`: Show browser windows (for Turnstile/Cloudflare)
- `--workers N`: Parallel download workers

## Output

- **CSV**: All papers with metadata and download status
- **PDFs**: Downloaded to specified directory
- **Debug files**: Screenshots/HTML for failed downloads

## Notes

- Paper count is per keyword, not cumulative
- Deduplicates papers found across multiple keywords
- Tracks all papers even if download fails
- Default email: j.martinez.4823@gmail.com
