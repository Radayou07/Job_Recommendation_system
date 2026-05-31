# data-pipeline/pipeline_runner.py
import asyncio
import time
from crawler import get_all_job_ids_from_feed, fetch_single_job_payload
from cleaner import clean_job_payload
from database import init_db, save_job_to_db
from playwright.async_api import async_playwright

# Change this to whatever speed you want for testing!
INTERVAL_SECONDS = 60  # Wakes up every 1 minute

async def run_pipeline_cycle():
    print("\n=== Pipeline Awakening: Checking for New Jobs ===")
    
    # 1. Get all IDs from the live feed page
    job_ids = await get_all_job_ids_from_feed()
    print(f"Found {len(job_ids)} active job listings on the main feed.")
    
    if not job_ids:
        return

    # 2. Open a single browser instance to loop through them efficiently
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        for idx, job_id in enumerate(job_ids):
            print(f"[{idx+1}/{len(job_ids)}] Processing Job ID: {job_id}...")
            
            # Fetch the raw JSON data
            raw_data = await fetch_single_job_payload(page, job_id)
            
            if raw_data:
                # Clean the fields and extract tags
                clean_job = clean_job_payload(raw_data)
                
                # Save straight to your database!
                save_job_to_db(clean_job)
                print(f"-> Successfully saved to database: {clean_job['title']}")
            
            # Crucial: 1-second pause between jobs so CamHR doesn't ban your IP address
            await asyncio.sleep(1) 
            
        await browser.close()
    print("=== Pipeline Cycle Complete. Database is up to date. ===")

if __name__ == "__main__":
    # Ensure our database table exists before running
    init_db()
    
    while True:
        try:
            asyncio.run(run_pipeline_cycle())
        except Exception as e:
            print(f"Pipeline Error: {e}")
            
        print(f"Sleeping for {INTERVAL_SECONDS} seconds...")
        time.sleep(INTERVAL_SECONDS)