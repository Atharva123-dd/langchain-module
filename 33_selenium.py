from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json
import time
import os
import pandas as pd

topic = "Artificial Intelligence"

# ---------------- Folders ----------------
os.makedirs("data", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

driver = webdriver.Chrome()

try:
    driver.get("https://www.wikipedia.org")

    search_box = driver.find_element(By.NAME, "search")
    search_box.send_keys(topic)
    search_box.submit()

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "firstHeading"))
    )

    # ---------------- SCREENSHOTS + SCROLL ----------------
    for i in range(5):
        driver.save_screenshot(f"screenshots/shot_{i+1}.png")
        driver.execute_script("window.scrollBy(0, 800)")
        time.sleep(1)

    # ---------------- SCRAPING ----------------
    soup = BeautifulSoup(driver.page_source, "html.parser")

    title = soup.find("h1").text.strip()

    # paragraphs
    paragraphs = [
        p.get_text(strip=True)
        for p in soup.find_all("p")
        if p.get_text(strip=True)
    ]

    # links
    links = list(set([
        a.get("href")
        for a in soup.find_all("a", href=True)
        if a.get("href").startswith("/wiki/")
    ]))

    # infobox (VERY important for LinkedIn projects)
    infobox_data = {}

    infobox = soup.find("table", {"class": "infobox"})
    if infobox:
        for row in infobox.find_all("tr"):
            header = row.find("th")
            value = row.find("td")
            if header and value:
                infobox_data[header.text.strip()] = value.text.strip()

    # ---------------- FINAL DATA ----------------
    data = {
        "topic": topic,
        "title": title,
        "summary": paragraphs[:5],
        "paragraphs": paragraphs[:10],
        "links_count": len(links),
        "infobox": infobox_data
    }

    # ---------------- SAVE JSON ----------------
    json_path = "data/article.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    # ---------------- SAVE CSV ----------------
    df = pd.DataFrame(paragraphs[:10], columns=["paragraph"])
    df.to_csv("data/article.csv", index=False)

    print("✅ Data + screenshots saved successfully!")

finally:
    driver.quit()