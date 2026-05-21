import pandas as pd

DATA_PATH = "data/electroplanet_products_large.csv"

df = pd.read_csv(DATA_PATH)

print("Total products:", len(df))

if "image_url" not in df.columns:
    print("No image_url column found.")
else:
    empty_images = df["image_url"].isna().sum() + (df["image_url"].astype(str).str.strip() == "").sum()
    http_images = df["image_url"].astype(str).str.startswith("http").sum()

    print("Empty image URLs:", empty_images)
    print("Valid http image URLs:", http_images)

    print("\nSample image URLs:")
    print(df[["product_name", "image_url"]].head(20))