# Enhancement Module

The `enhancement` module is responsible for enriching bookmarks with additional metadata by scraping their web content. It includes a variety of scrapers with different capabilities to handle various types of websites.

## Modules

### Link Enhancement Engine (`link_enhancement_engine.py`)

The `LinkEnhancementEngine` is the main orchestrator for the bookmark enhancement process. It coordinates the actions of the different scrapers to extract the most relevant information from a given URL.

**Classes:**

- **`EnhancementConfig`**: A data class for configuring the link enhancement process.
- **`LinkEnhancementEngine`**: The main class that orchestrates the enhancement process.
    - **`__init__(config)`**: Initializes the link enhancement engine with the application configuration.
    - **`enhance_bookmark(url, existing_data)`**: Enhances a single bookmark with metadata by selecting the most appropriate scraper and running it.
    - **`enhance_bookmarks_batch(urls)`**: Enhances multiple bookmarks concurrently for better performance.
    - **`enhance_bookmark_collection(bookmarks)`**: Enhances a collection of bookmarks and returns the enhanced data along with a report.
    - **`_select_scraper(url)`**: Selects the most appropriate scraper for a given URL based on its characteristics.
    - **`_apply_enhancement_to_bookmark(original_bookmark, result)`**: Applies the scraping result to the bookmark data.

### Base Scraper (`base_scraper.py`)

The `base_scraper.py` module provides the base framework for all web scrapers. It includes common functionality such as rate limiting, HTTP session management, and utility functions for extracting common metadata.

**Classes:**

- **`ScrapingResult`**: A data class for storing the results of a scraping operation.
- **`RateLimiter`**: A class for rate limiting HTTP requests to avoid overloading websites.
- **`WebScraper`**: An abstract base class that defines the common interface for all scrapers.
    - **`__init__(config)`**: Initializes the scraper with the application configuration.
    - **`can_scrape(url)`**: An abstract method that must be implemented by subclasses to determine if they can handle a given URL.
    - **`scrape(url)`**: An abstract method that must be implemented by subclasses to perform the scraping.
    - **`_make_request(url, **kwargs)`**: A helper method for making HTTP requests with rate limiting and error handling.
    - **`_extract_favicon(url, soup)`**: A helper method for extracting the favicon URL from a page.
    - **`_extract_meta_tags(soup)`**: A helper method for extracting metadata from HTML meta tags.

### BeautifulSoup Scraper (`beautifulsoup_scraper.py`)

The `BeautifulSoupScraper` is a general-purpose scraper that uses the BeautifulSoup library to parse and extract information from HTML content.

**Classes:**

- **`BeautifulSoupScraper`**: The main class that performs scraping using BeautifulSoup.
    - **`__init__(config)`**: Initializes the BeautifulSoup scraper with the application configuration.
    - **`can_scrape(url)`**: Determines if the scraper can handle the given URL.
    - **`scrape(url)`**: Scrapes the given URL and returns the results.
    - **`_extract_*()`**: A series of helper methods for extracting specific pieces of information from the HTML content, such as the title, description, keywords, and more.

### Newspaper Scraper (`newspaper_scraper.py`)

The `NewspaperScraper` is a specialized scraper that uses the newspaper3k library to extract content from news articles and blog posts.

**Classes:**

- **`NewspaperScraper`**: The main class that performs scraping using newspaper3k.
    - **`__init__(config)`**: Initializes the Newspaper scraper with the application configuration.
    - **`can_scrape(url)`**: Determines if the scraper can handle the given URL, which is typically a news article or blog post.
    - **`scrape(url)`**: Scrapes the given URL and returns the results, including the full text of the article.
    - **`_extract_*()`**: A series of helper methods for extracting specific pieces of information from the article, such as the title, author, and publication date.

### Selenium Scraper (`selenium_scraper.py`)

The `SeleniumScraper` is a powerful scraper that uses the Selenium WebDriver to render web pages in a real browser. This allows it to handle JavaScript-heavy sites and dynamic content that other scrapers cannot.

**Classes:**

- **`SeleniumScraper`**: The main class that performs scraping using Selenium.
    - **`__init__(config)`**: Initializes the Selenium scraper with the application configuration.
    - **`can_scrape(url)`**: Determines if the scraper should be used for the given URL, which is typically a JavaScript-heavy site.
    - **`scrape(url)`**: Scrapes the given URL by rendering it in a real browser and returns the results.
    - **`_create_driver()`**: Creates and configures the Selenium WebDriver instance.
    - **`_extract_*()`**: A series of helper methods for extracting specific pieces of information from the rendered page.
    - **`_take_screenshot(url)`**: Takes a screenshot of the page.
    - **`cleanup()`**: Cleans up the WebDriver resources.

### Scraping Cache (`scraping_cache.py`)

The `scraping_cache.py` module provides a persistent caching system for scraping results. This helps to improve performance and reduce the number of requests made to websites.

**Classes:**

- **`CacheEntry`**: A data class for storing a single cache entry.
- **`ScrapingCache`**: The main class that manages the scraping cache.
    - **`__init__(config)`**: Initializes the scraping cache with the application configuration.
    - **`get(url)`**: Gets the cached result for a given URL.
    - **`put(url, result, ttl_seconds)`**: Caches the result for a given URL.
    - **`invalidate(url)`**: Invalidates the cache entry for a given URL.
    - **`clear()`**: Clears all cache entries.
- **`ConcurrentScraper`**: A class for scraping multiple URLs concurrently.
    - **`__init__(config, scrapers, cache)`**: Initializes the concurrent scraper with the application configuration, a list of scrapers, and a cache instance.
    - **`scrape_urls(urls)`**: Scrapes multiple URLs concurrently using a thread pool.
    - **`_scrape_with_retry(url)`**: Scrapes a single URL with retry logic.
