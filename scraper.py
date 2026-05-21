from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os


# =========================
# CONFIG
# =========================

BASE_URLS = {
    "promotions": "https://www.electroplanet.ma/promotions",
    "vente_flash": "https://www.electroplanet.ma/vente-flash",
    "destockages": "https://www.electroplanet.ma/destockages"
}

MAX_PAGES = 10
OUTPUT_PATH = "data/electroplanet_products_large.csv"


# =========================
# DRIVER
# =========================

def create_driver():
    options = Options()

    # Keep Chrome visible because Cloudflare blocks headless browsers more easily
    options.add_argument("--start-maximized")
    options.add_argument("--lang=fr-FR")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    return driver


# =========================
# CLEANING FUNCTIONS
# =========================

def clean_text(text):
    if text is None:
        return ""

    return re.sub(r"\s+", " ", str(text)).strip()


def extract_price(text):
    if not text:
        return None

    text = text.replace("\u202f", " ").replace("\xa0", " ")

    match = re.search(r"([\d\s]+)\s*DH", text)

    if match:
        price = match.group(1).replace(" ", "")

        try:
            return float(price)
        except Exception:
            return None

    return None


def extract_discount(text):
    if not text:
        return 0

    match = re.search(r"-(\d+)%", text)

    if match:
        return int(match.group(1))

    return 0


# =========================
# CATEGORY DETECTION
# =========================

def detect_category(name):
    name = str(name).lower()

    categories = {
        "Réfrigérateur": [
            "réfrigérateur", "refrigerateur", "frigo", "ref ",
            "comb", "combi", "no frost"
        ],
        "Congélateur": [
            "congélateur", "congelateur", "cong "
        ],
        "Machine à laver": [
            "machine à laver", "lave-linge", "lavelinge",
            "mal ", "washing"
        ],
        "TV": [
            "tv", "smart tv", "led", "oled", "qled",
            "téléviseur", "televiseur"
        ],
        "Téléphone": [
            "smartphone", "iphone", "galaxy", "xiaomi", "oppo",
            "telephone", "téléphone"
        ],
        "Cuisine": [
            "air fry", "airfryer", "four", "plaque", "mixeur",
            "robot", "cuisson", "friteuse", "tefal", "cafetière",
            "cafe", "espresso", "cocotte", "ustensiles"
        ],
        "Climatisation": [
            "climatiseur", "climatisation", "clima"
        ],
        "Accessoires": [
            "cable", "câble", "chargeur", "écouteurs", "ecouteurs",
            "casque", "buds", "type-c", "type c"
        ],
        "Ordinateur": [
            "pc", "ordinateur", "laptop", "imprimante", "écran",
            "ecran", "monitor", "moniteur", "gaming"
        ],
        "Nettoyage": [
            "aspirateur", "nettoyeur", "karcher", "balai", "rowenta"
        ]
    }

    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in name:
                return category

    return "Autre"


# =========================
# IMAGE EXTRACTION
# =========================

def extract_image_url(block):
    image_url = ""

    img_tag = block.find("img")

    if not img_tag:
        return image_url

    # Important:
    # We prefer lazy-loading attributes before src because src often contains
    # a base64 placeholder image.
    possible_attrs = [
        "data-src",
        "data-original",
        "data-lazy",
        "data-image",
        "data-srcset",
        "srcset",
        "src"
    ]

    for attr in possible_attrs:
        value = img_tag.get(attr)

        if not value:
            continue

        value = value.strip()

        # Ignore placeholder images
        if value.startswith("data:image"):
            continue

        # If srcset contains multiple images, take the first one
        if "," in value:
            value = value.split(",")[0].strip()

        # If format is "url 300w", keep only the URL
        value = value.split(" ")[0].strip()

        if value:
            image_url = value
            break

    if image_url.startswith("//"):
        image_url = "https:" + image_url
    elif image_url.startswith("/"):
        image_url = "https://www.electroplanet.ma" + image_url

    return image_url


# =========================
# PRODUCT EXTRACTION
# =========================

def extract_product_from_block(block, source_page, page_number):
    text = clean_text(block.get_text(" ", strip=True))

    if "Prix Spécial" not in text and "Prix normal" not in text:
        return None

    if "Ajouter au panier" not in text:
        return None

    # Product name extraction
    name = text

    if "Promo" in name:
        name = name.split("Promo")[0]
    elif "Prix normal" in name:
        name = name.split("Prix normal")[0]

    name = clean_text(name)

    # Remove ratings like 96% (5)
    name = re.sub(r"\d+%\s*\(\d+\)", "", name).strip()

    # Remove noise words
    name = name.replace("Nouveau", "").replace("Exclus", "").strip()

    if len(name) < 4:
        return None

    old_price = None
    current_price = None

    old_match = re.search(r"Prix normal\s*([\d\s\u202f\xa0]+)\s*DH", text)
    special_match = re.search(r"Prix Spécial\s*([\d\s\u202f\xa0]+)\s*DH", text)

    if old_match:
        old_price = extract_price(old_match.group(0))

    if special_match:
        current_price = extract_price(special_match.group(0))

    discount = extract_discount(text)

    # Calculate discount if not visible
    if discount == 0 and old_price and current_price and old_price > current_price:
        discount = round(((old_price - current_price) / old_price) * 100, 2)

    # Product URL
    product_url = ""
    a_tag = block.find("a", href=True)

    if a_tag:
        href = a_tag["href"]

        if href.startswith("http"):
            product_url = href
        else:
            product_url = "https://www.electroplanet.ma" + href

    # Improved image URL extraction
    image_url = extract_image_url(block)

    category = detect_category(name)

    return {
        "product_name": name,
        "category": category,
        "old_price": old_price,
        "price": current_price,
        "discount_percentage": discount,
        "product_url": product_url,
        "image_url": image_url,
        "source_page": source_page,
        "page_number": page_number,
        "raw_text": text[:500]
    }


# =========================
# PAGE URL
# =========================

def build_page_url(base_url, page_number):
    if page_number == 1:
        return base_url

    return base_url + "?p=" + str(page_number)


# =========================
# SCROLLING
# =========================

def slow_scroll(driver):
    # Scroll slowly to trigger lazy-loaded product images
    scroll_positions = [
        400, 800, 1200, 1800, 2500, 3500, 5000, 7000, 9000
    ]

    for position in scroll_positions:
        driver.execute_script(f"window.scrollTo(0, {position});")
        time.sleep(1)

    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(2)


# =========================
# SCRAPE PAGE
# =========================

def scrape_page(driver, page_name, base_url, page_number):
    url = build_page_url(base_url, page_number)

    print("=" * 100)
    print(f"Scraping: {page_name} | page {page_number}")
    print(url)

    driver.get(url)

    # Wait for Cloudflare / page loading
    time.sleep(5)

    # Important for lazy images
    slow_scroll(driver)

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    page_text = soup.get_text(" ", strip=True)

    if (
        "Cloudflare" in page_text
        or "Attention Required" in page_text
        or "Sorry, you have been blocked" in page_text
    ):
        print("Blocked by Cloudflare.")
        return []

    product_blocks = soup.select(
        ".product-item, .product, li.item, div.product-item-info, div.item"
    )

    print("Candidate blocks:", len(product_blocks))

    products = []

    for block in product_blocks:
        product = extract_product_from_block(block, page_name, page_number)

        if product:
            products.append(product)

    print("Extracted products:", len(products))

    return products


# =========================
# MAIN
# =========================

def main():
    os.makedirs("data", exist_ok=True)

    driver = create_driver()
    all_products = []

    try:
        for page_name, base_url in BASE_URLS.items():
            previous_page_names = set()

            for page_number in range(1, MAX_PAGES + 1):
                products = scrape_page(driver, page_name, base_url, page_number)

                if len(products) == 0:
                    print("No products found. Stopping this section.")
                    break

                current_page_names = set([p["product_name"] for p in products])

                # Stop if pagination repeats the same products
                if current_page_names == previous_page_names:
                    print("This page repeats the previous page. Stopping this section.")
                    break

                all_products.extend(products)
                previous_page_names = current_page_names

                time.sleep(2)

    finally:
        driver.quit()

    df = pd.DataFrame(all_products)

    if df.empty:
        print("No products extracted.")
        return

    print("\nBefore cleaning:", len(df))

    # Remove products without price
    df = df[df["price"].notna()]

    # Remove duplicate products
    df = df.drop_duplicates(subset=["product_name", "price", "old_price"])

    print("After cleaning:", len(df))

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\nSaved:", OUTPUT_PATH)
    print("Total unique products:", len(df))

    print("\nProducts per source:")
    print(df["source_page"].value_counts())

    print("\nProducts per category:")
    print(df["category"].value_counts())

    valid_http_images = df["image_url"].astype(str).str.startswith("http").sum()
    base64_images = df["image_url"].astype(str).str.startswith("data:image").sum()
    empty_images = (df["image_url"].astype(str).str.strip() == "").sum()

    print("\nImage URL stats:")
    print("Valid http image URLs:", valid_http_images)
    print("Base64 placeholder images:", base64_images)
    print("Empty image URLs:", empty_images)

    print("\nPreview:")
    print(
        df[
            [
                "product_name",
                "category",
                "old_price",
                "price",
                "discount_percentage",
                "image_url",
                "source_page",
                "page_number"
            ]
        ].head(30)
    )


if __name__ == "__main__":
    main()