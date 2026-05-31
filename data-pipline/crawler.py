# data-pipeline/crawler.py
import asyncio
import re
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

async def get_all_job_ids_from_feed() -> list:
    """Opens the main list page and grabs every job ID currently visible."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        await page.goto("https://camhr.com/a/job", wait_until="networkidle")
        await page.wait_for_selector(".job-list-container", timeout=10000)
        
        html = await page.content()
        await browser.close()
        
        soup = BeautifulSoup(html, "html.parser")
        # Find all anchor tags pointing to a job detail page
        links = soup.find_all("a", href=re.compile(r'/a/job/\d+'))
        
        # Extract the numbers (IDs) from the URLs safely using regex
        job_ids = []
        for link in links:
            match = re.search(r'/a/job/(\d+)', link['href'])
            if match:
                job_ids.append(match.group(1))
                
        return list(set(job_ids)) # Remove duplicates on the page

async def fetch_single_job_payload(page, job_id: str) -> dict:
    """Re-usable function to grab the internal JSON API data for one ID using an open page."""
    captured_data = {}

    async def handle_response(response):
        if job_id in response.url and "json" in response.headers.get("content-type", ""):
            try:
                json_data = await response.json()
                if "data" in json_data:
                    captured_data["raw"] = json_data["data"]
            except Exception:
                pass

    page.on("response", handle_response)
    await page.goto(f"https://camhr.com/a/job/{job_id}", wait_until="networkidle")
    return captured_data.get("raw", None)