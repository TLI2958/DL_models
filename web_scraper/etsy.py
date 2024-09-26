"""
This is an example web scraper for etsy.com.

To run this scraper set env variable $SCRAPFLY_KEY with your scrapfly API key:
$ export $SCRAPFLY_KEY="your key from https://scrapfly.io/dashboard"
"""
import os
import re
import math
import json
from scrapfly import ScrapeConfig, ScrapflyClient, ScrapeApiResponse
from typing import Dict, List
from loguru import logger as log


SCRAPFLY = ScrapflyClient(key=os.environ["SCRAPFLY_KEY"])

BASE_CONFIG = {
    # bypass Etsy.com web scraping blocking
    "asp": True,
    # set the poxy location to US
    "country": "US",
}

def strip_text(text):
    """remove extra spaces while handling None values"""
    if text != None:
        text = text.strip()
    return text

def parse_search(response: ScrapeApiResponse) -> Dict:
    """parse data from Etsy search pages"""
    selector = response.selector
    data = []
    script = json.loads(selector.xpath("//script[@type='application/ld+json']/text()").get())
    # get the total number of pages
    total_listings = script["numberOfItems"]
    total_pages = math.ceil(total_listings / 48)
    for product in selector.xpath("//div[@data-search-results-lg]/ul/li[div[@data-appears-component-name]]"):
        link = product.xpath(".//a[contains(@class, 'listing-link')]/@href").get()
        rate = product.xpath(".//span[contains(@class, 'review_stars')]/span/text()").get()
        number_of_reviews = strip_text(product.xpath(".//div[contains(@aria-label,'star rating')]/p/text()").get())
        if number_of_reviews:
            number_of_reviews = number_of_reviews.replace("(", "").replace(")", "")
            number_of_reviews = int(number_of_reviews.replace("k", "").replace(".", "")) * 10 if "k" in number_of_reviews else number_of_reviews
        price = product.xpath(".//span[@class='currency-value']/text()").get()
    
        seller = product.xpath(".//span[contains(text(),'From shop')]/text()").get()
        data.append({
            "productLink": '/'.join(link.split('/')[:5]) if link else None,
            "productTitle": strip_text(product.xpath(".//h3[contains(@class, 'v2-listing-card__titl')]/@title").get()),
            # "productImage": product.xpath("//img[@data-listing-card-listing-image]/@src").get(),
            "seller": seller.replace("From shop ", "") if seller else None,
            "listingType": "Paid listing" if product.xpath(".//span[@data-ad-label='Ad by Etsy seller']") else "Free listing",
            "productRate": float(rate.strip()) if rate else None,
            "numberOfReviews": int(number_of_reviews) if number_of_reviews else None,
            "freeShipping": "Yes" if product.xpath(".//span[contains(text(),'Free shipping')]/text()").get() else "No",
            "productPrice": float(price.replace(",", "")) if price else None,
            # "priceCurrency": currency,
            # "originalPrice": float(original_price.split(currency)[-1].strip()) if original_price else "No discount",
            # "discount": discount if discount else "No discount",
        })
    return {
        "search_data": data,
        "total_pages": total_pages
    }

def parse_review(response: ScrapeApiResponse):
    selector = response.selector
    script = selector.xpath("//script[contains(text(),'offers')]/text()").get()
    products_data = json.loads(script)
    return {"product_data":[products_data]}


async def parse_product_page(response, search_data, max_review_pages = 10) -> Dict:
    """Parse hidden product data from product pages, including reviews."""
    selector = response.selector
    # Get total review count
    try:
        review_for_this_item = selector.xpath('//*[@id="same-listing-reviews-tab"]/span').get()
        total_review_for_this_item = strip_text(review_for_this_item.split("\n")[1])
        total_review_pages = min(max_review_pages, math.ceil(int(total_review_for_this_item) / 4))  # adjust the divisor based on actual reviews per page
    except Exception as e:
        log.warning(f"Could not retrieve total review count: {e}")
        total_review_pages = 1
    
    # Scrape the first review page
    # todo: recursively scrape all review pages
    next_page_url = [selector.xpath(f'//*[@id="reviews"]/nav/div/div[{i}]/a/@href').get() for i in range(2,max_review_pages+1)]
    next_page_url = [url for url in next_page_url if url]

    first_page = await SCRAPFLY.async_scrape(
        ScrapeConfig(next_page_url[0], wait_for_selector="//div[@data-reviews-pagination]", render_js=True, **BASE_CONFIG)
    )
    product_data = parse_review(first_page)['product_data']

    # Scrape remaining review pages concurrently, if there are more pages
    log.info(f"Scraping review pagination ({total_review_pages - 1} more pages)")
    if total_review_pages > 1:
        other_pages = [
                ScrapeConfig(next_page_url[i], wait_for_selector="//div[@data-reviews-pagination]", render_js=True, **BASE_CONFIG)
                for i in range(1, len(next_page_url))
            ]

        # Scrape remaining review pages
        async for response in SCRAPFLY.concurrent_scrape(other_pages):
                data = parse_review(response)
                product_data.extend(data["product_data"])

        log.success(f"Scraped {len(product_data)} review pages")
    return product_data

async def scrape_search(url: str, max_pages: int = None) -> List[Dict]:
    """scrape product listing data from Etsy search pages"""
    log.info("scraping the first search page")
    # etsy search pages are dynaminc, requiring render_js enabled
    first_page = await SCRAPFLY.async_scrape(ScrapeConfig(url, wait_for_selector="//div[@data-search-pagination]", render_js=True, **BASE_CONFIG))
    data = parse_search(first_page)
    search_data = data["search_data"]

    # get the number of total pages to scrape
    total_pages = data["total_pages"]
    if max_pages and max_pages < total_pages:
        total_pages = max_pages

    log.info(f"scraping search pagination ({total_pages - 1} more pages)")
        # add the remaining search pages in a scraping list
    other_pages = [
        ScrapeConfig(url + f"&page={page_number}", wait_for_selector="//div[@data-search-pagination]", render_js=True, **BASE_CONFIG)
        for page_number in range(2, total_pages + 1)
    ]
    # scrape the remaining search pages concurrently
    async for response in SCRAPFLY.concurrent_scrape(other_pages):
        data = parse_search(response)
        search_data.extend(data["search_data"])
    log.success(f"scraped {len(search_data)} product listings from search")
    return search_data


async def scrape_product(urls: List[str], search_data, max_review_pages: int) -> List[Dict]:
    """scrape trustpilot company pages"""
    products = []
    # add the product page URLs to a scraping list
    to_scrape = [ScrapeConfig(url, **BASE_CONFIG) for url in urls]
    # scrape all the product pages concurrently
    async for response in SCRAPFLY.concurrent_scrape(to_scrape):
        data = await parse_product_page(response, search_data, max_review_pages)
        products.append(data)
    log.success(f"scraped {len(products)} product listings from product pages")
    return products


async def scrape_search_and_products(search_url: str, max_pages: int = None, max_review_pages: int = None) -> List[Dict]:
    """Scrape product listing data from Etsy search pages and scrape detailed product info."""
    
    # Step 1: Scrape search results to get product listings
    log.info("Starting search scrape")
    search_data = await scrape_search(search_url, max_pages)
    
    # Step 2: Extract product URLs from search data
    product_urls = [product["productLink"] for product in search_data if product.get("productLink")]
    
    log.info(f"Found {len(product_urls)} product links to scrape")
    
    # Step 3: Pass product URLs to scrape_product to scrape product pages
    products_data = await scrape_product(product_urls, search_data, max_review_pages)
    
    log.success(f"Scraped {len(products_data)} product pages successfully")
    
    return search_data, products_data
