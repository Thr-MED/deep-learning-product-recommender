import os
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px

from recommender import recommend_products


DATA_PATH = "data/electroplanet_products_large.csv"

st.set_page_config(
    page_title="Live Electroplanet Promo Recommender",
    page_icon="🛒",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css');

:root {
    --primary: #e63946;
    --dark: #0f172a;
    --navy: #1e293b;
    --slate: #475569;
    --border: #e2e8f0;
    --bg: #f8fafc;
    --card-bg: #ffffff;
    --accent1: #eff6ff;
    --accent2: #fef2f2;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
}

.main .block-container {
    background: var(--bg);
    padding-top: 2rem !important;
}

h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: var(--dark) !important;
    letter-spacing: -0.5px !important;
    border-bottom: 3px solid var(--primary);
    padding-bottom: 0.6rem;
    margin-bottom: 0.5rem !important;
}

h2 {
    font-size: 1.3rem !important;
    font-weight: 600 !important;
    color: var(--dark) !important;
    padding-left: 0.75rem;
    border-left: 4px solid var(--primary);
    margin: 1.5rem 0 1rem 0 !important;
}

h3 {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: var(--navy) !important;
}

[data-testid="stSidebar"] {
    background: var(--dark) !important;
    border-right: 1px solid var(--navy);
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
    border-left: none !important;
    padding-left: 0 !important;
    font-size: 0.95rem !important;
    border-bottom: 1px solid #1e293b !important;
    padding-bottom: 0.5rem !important;
}

[data-testid="stSidebar"] .stTextInput input {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
    font-size: 0.88rem !important;
}

[data-testid="stSidebar"] .stNumberInput input {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
}

[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    width: 100% !important;
    padding: 0.6rem 1rem !important;
    letter-spacing: 0.3px !important;
    transition: background 0.2s !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: #c1121f !important;
}

[data-testid="metric-container"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-top: 3px solid var(--primary) !important;
    border-radius: 10px !important;
    padding: 1.2rem 1.5rem !important;
    box-shadow: 0 2px 8px rgba(15,23,42,0.06) !important;
}

[data-testid="stMetricLabel"] {
    color: var(--slate) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

[data-testid="stMetricValue"] {
    color: var(--dark) !important;
    font-weight: 700 !important;
    font-size: 1.8rem !important;
}

.product-card {
    background: var(--card-bg);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid var(--border);
    border-left: 4px solid var(--primary);
    box-shadow: 0 2px 8px rgba(15,23,42,0.05);
    display: flex;
    align-items: center;
    gap: 1.5rem;
}

.product-card img {
    width: 100px;
    height: 100px;
    object-fit: contain;
    border-radius: 8px;
    background: var(--bg);
    padding: 0.5rem;
    border: 1px solid var(--border);
}

.product-badge {
    display: inline-block;
    background: var(--accent2);
    color: var(--primary);
    font-weight: 700;
    font-size: 0.75rem;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    margin-bottom: 0.4rem;
    border: 1px solid #fecdd3;
}

.product-category {
    display: inline-block;
    background: var(--accent1);
    color: #1d4ed8;
    font-weight: 600;
    font-size: 0.72rem;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    margin-bottom: 0.4rem;
    margin-left: 0.3rem;
    border: 1px solid #bfdbfe;
}

.score-pill {
    display: inline-block;
    background: var(--bg);
    color: var(--slate);
    font-size: 0.72rem;
    font-weight: 500;
    border-radius: 20px;
    padding: 0.2rem 0.65rem;
    margin-right: 0.25rem;
    margin-top: 0.3rem;
    border: 1px solid var(--border);
    font-family: 'DM Mono', monospace;
}

.score-pill.final {
    background: var(--dark);
    color: white;
    font-weight: 700;
    border-color: var(--dark);
}

.product-link {
    display: inline-block;
    margin-top: 0.5rem;
    font-size: 0.82rem;
    color: var(--primary) !important;
    font-weight: 600;
    text-decoration: none !important;
    border: 1px solid #fecdd3;
    border-radius: 6px;
    padding: 0.25rem 0.7rem;
    background: var(--accent2);
}

[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 2px 8px rgba(15,23,42,0.05) !important;
}

hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.88rem !important;
}

a { color: var(--primary) !important; font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_products():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["product_name", "price"])
    df["discount_percentage"] = df["discount_percentage"].fillna(0)
    df["category"] = df["category"].fillna("Autre")
    return df


st.markdown('<h1><i class="fa-solid fa-tags"></i> Live Electroplanet Promotion Recommender</h1>', unsafe_allow_html=True)

st.markdown("""
This application recommends Electroplanet promotional products using a **Hybrid GRU Deep Learning Model**.
The recommendation considers: user search query, product category, product price, user budget, discount percentage, and GRU model prediction.
""")

df = load_products()

if df.empty:
    st.error("No product data found. Please check the CSV file.")
    st.stop()


st.sidebar.markdown('<h3><i class="fa-solid fa-sliders"></i> User Preferences</h3>', unsafe_allow_html=True)

user_query = st.sidebar.text_input("What are you looking for?", value="réfrigérateur pas cher pour aid al adha")
budget = st.sidebar.number_input("Maximum budget (DH)", min_value=100, max_value=50000, value=9000, step=100)
min_discount = st.sidebar.slider("Minimum discount (%)", min_value=0, max_value=80, value=5, step=1)
top_n = st.sidebar.slider("Number of recommendations", min_value=5, max_value=30, value=10, step=1)
categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
category_filter = st.sidebar.selectbox("Category filter", categories)
show_images = st.sidebar.checkbox("Show product images", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown('<h3><i class="fa-solid fa-rotate"></i> Actualiser les données</h3>', unsafe_allow_html=True)

if st.sidebar.button("Lancer le scraping"):
    import subprocess
    with st.sidebar:
        with st.spinner("Scraping en cours... (2-5 minutes)"):
            try:
                result = subprocess.run(
                    [r"C:\Users\admin\AppData\Local\Programs\Python\Python311\python.exe", "scraper.py"],
                    capture_output=True, text=True, timeout=600
                )
                if result.returncode == 0:
                    st.success("Données mises à jour avec succès !")
                    st.cache_data.clear()
                else:
                    st.error("Erreur pendant le scraping")
                    st.code(result.stderr)
            except subprocess.TimeoutExpired:
                st.error("Timeout - le scraping a pris trop longtemps")
            except Exception as e:
                st.error(f"Erreur: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.info("Live data was collected from Electroplanet using Selenium. The app uses the latest cached scraped data.")
st.sidebar.caption("Images are loaded from the scraped Electroplanet product URLs.")


st.markdown(f"""
<div style="background:white;border:1px solid #e2e8f0;border-radius:14px;padding:1.5rem 2rem;box-shadow:0 2px 10px rgba(15,23,42,0.06);margin-bottom:1rem;">
    <div style="display:flex;gap:0;justify-content:space-between;align-items:center;flex-wrap:wrap;">
        <div style="text-align:center;padding:0.5rem 2rem;border-right:1px solid #e2e8f0;">
            <div style="font-size:0.72rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.3rem;"><i class="fa-solid fa-box-open"></i> Products Collected</div>
            <div style="font-size:2rem;font-weight:700;color:#0f172a;">{len(df)}</div>
        </div>
        <div style="text-align:center;padding:0.5rem 2rem;border-right:1px solid #e2e8f0;">
            <div style="font-size:0.72rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.3rem;"><i class="fa-solid fa-layer-group"></i> Categories</div>
            <div style="font-size:2rem;font-weight:700;color:#0f172a;">{df["category"].nunique()}</div>
        </div>
        <div style="text-align:center;padding:0.5rem 2rem;border-right:1px solid #e2e8f0;">
            <div style="font-size:0.72rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.3rem;"><i class="fa-solid fa-percent"></i> Average Discount</div>
            <div style="font-size:2rem;font-weight:700;color:#e63946;">{df["discount_percentage"].mean():.1f}%</div>
        </div>
        <div style="text-align:center;padding:0.5rem 2rem;">
            <div style="font-size:0.72rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.3rem;"><i class="fa-solid fa-fire"></i> Max Discount</div>
            <div style="font-size:2rem;font-weight:700;color:#e63946;">{df["discount_percentage"].max():.0f}%</div>
        </div>
    </div>
    <div style="margin-top:1rem;padding-top:0.8rem;border-top:1px solid #e2e8f0;font-size:0.75rem;color:#94a3b8;">
        <i class="fa-solid fa-clock"></i> Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown("---")
st.markdown('<h2><i class="fa-solid fa-bullseye"></i> Recommended Promotions</h2>', unsafe_allow_html=True)

recommendations = recommend_products(
    user_query=user_query,
    budget=budget,
    min_discount=min_discount,
    category_filter=category_filter,
    top_n=top_n,
    data_path=DATA_PATH
)

if recommendations.empty:
    st.warning("No recommendations found. Try increasing your budget or lowering the minimum discount.")
else:
    display_cols = ["product_name", "category", "price", "old_price", "discount_percentage",
                    "model_score", "query_relevance_score", "discount_score", "final_score", "source_page"]
    st.dataframe(recommendations[display_cols], use_container_width=True)

    st.markdown('<h2><i class="fa-solid fa-trophy"></i> Top Recommended Products</h2>', unsafe_allow_html=True)

    for i, (_, row) in enumerate(recommendations.iterrows(), start=1):
        image_url = str(row.get("image_url", ""))
        product_url = str(row.get("product_url", ""))

        img_tag = (
            f'<img src="{image_url}" onerror="this.style.display=\'none\'">'
            if show_images and image_url.startswith("http")
            else '<div style="width:100px;height:100px;background:#f8fafc;border-radius:8px;display:flex;align-items:center;justify-content:center;border:1px solid #e2e8f0;"><i class="fa-solid fa-cart-shopping" style="font-size:2rem;color:#94a3b8;"></i></div>'
        )

        link_tag = (
            f'<a class="product-link" href="{product_url}" target="_blank"><i class="fa-solid fa-arrow-up-right-from-square"></i> Voir le produit</a>'
            if product_url.startswith("http") else ""
        )

        st.markdown(f"""
        <div class="product-card">
            {img_tag}
            <div style="flex:1">
                <div>
                    <span class="product-badge">#{i}</span>
                    <span class="product-category"><i class="fa-solid fa-layer-group"></i> {row['category']}</span>
                </div>
                <div style="font-size:1rem;font-weight:700;color:#0f172a;margin:0.3rem 0;">{row['product_name']}</div>
                <div style="margin-bottom:0.5rem;">
                    <span style="font-size:1.15rem;font-weight:700;color:#0f172a;">{row['price']} DH</span>
                    <span style="font-size:0.88rem;color:#94a3b8;text-decoration:line-through;margin-left:0.4rem;">{row['old_price']} DH</span>
                    <span class="product-badge" style="margin-left:0.5rem;">-{row['discount_percentage']}%</span>
                </div>
                <div style="margin-bottom:0.5rem;">
                    <span class="score-pill"><i class="fa-solid fa-microchip"></i> GRU {row['model_score']}</span>
                    <span class="score-pill"><i class="fa-solid fa-bullseye"></i> Relevance {row['query_relevance_score']}</span>
                    <span class="score-pill"><i class="fa-solid fa-tag"></i> Discount {row['discount_score']}</span>
                    <span class="score-pill final"><i class="fa-solid fa-star"></i> Score {row['final_score']}</span>
                </div>
                {link_tag}
            </div>
        </div>
        """, unsafe_allow_html=True)


st.markdown('<h2><i class="fa-solid fa-chart-bar"></i> Data Analysis</h2>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<h3><i class="fa-solid fa-layer-group"></i> Products by Category</h3>', unsafe_allow_html=True)
    category_counts = df["category"].value_counts().reset_index()
    category_counts.columns = ["category", "count"]
    fig_cat = px.bar(category_counts, x="category", y="count", title="Number of Products per Category")
    st.plotly_chart(fig_cat, use_container_width=True)

with col_b:
    st.markdown('<h3><i class="fa-solid fa-percent"></i> Discount Distribution</h3>', unsafe_allow_html=True)
    fig_discount = px.histogram(df, x="discount_percentage", nbins=20, title="Distribution of Discount Percentages")
    st.plotly_chart(fig_discount, use_container_width=True)

st.markdown('<h3><i class="fa-solid fa-chart-scatter"></i> Price vs Discount</h3>', unsafe_allow_html=True)
fig_scatter = px.scatter(df, x="price", y="discount_percentage", color="category",
                         hover_data=["product_name", "source_page"], title="Price vs Discount by Category")
st.plotly_chart(fig_scatter, use_container_width=True)


st.markdown("---")
st.markdown('<h2><i class="fa-solid fa-brain"></i> Deep Learning Model Explanation</h2>', unsafe_allow_html=True)

st.markdown("""<div style="background:white;border:1px solid #e2e8f0;border-radius:14px;padding:2rem;box-shadow:0 2px 10px rgba(15,23,42,0.06);">
  <div style="display:flex;gap:1rem;margin-bottom:1.5rem;">
    <div style="flex:1;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:1.2rem;">
      <div style="font-size:0.7rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.8rem;"><i class="fa-solid fa-font"></i> Text Input</div>
      <div style="display:flex;flex-direction:column;gap:0.4rem;">
        <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;padding:0.4rem 0.7rem;font-size:0.8rem;color:#1d4ed8;font-weight:500;"><i class="fa-solid fa-magnifying-glass"></i> User Query</div>
        <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;padding:0.4rem 0.7rem;font-size:0.8rem;color:#1d4ed8;font-weight:500;"><i class="fa-solid fa-tag"></i> Product Name</div>
        <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;padding:0.4rem 0.7rem;font-size:0.8rem;color:#1d4ed8;font-weight:500;"><i class="fa-solid fa-layer-group"></i> Product Category</div>
      </div>
    </div>
    <div style="display:flex;align-items:center;font-size:1.3rem;color:#94a3b8;"><i class="fa-solid fa-arrow-right"></i></div>
    <div style="flex:1;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:1.2rem;">
      <div style="font-size:0.7rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.8rem;"><i class="fa-solid fa-microchip"></i> Text Processing</div>
      <div style="display:flex;flex-direction:column;gap:0.4rem;">
        <div style="background:#fef9c3;border:1px solid #fde68a;border-radius:6px;padding:0.4rem 0.7rem;font-size:0.8rem;color:#92400e;font-weight:500;"><i class="fa-solid fa-table-cells"></i> Embedding Layer</div>
        <div style="text-align:center;color:#94a3b8;font-size:0.75rem;">↓</div>
        <div style="background:#fef9c3;border:1px solid #fde68a;border-radius:6px;padding:0.4rem 0.7rem;font-size:0.8rem;color:#92400e;font-weight:500;"><i class="fa-solid fa-wave-square"></i> GRU Layer</div>
      </div>
    </div>
    <div style="display:flex;align-items:center;font-size:1.3rem;color:#94a3b8;"><i class="fa-solid fa-arrow-right"></i></div>
    <div style="flex:1;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:1.2rem;">
      <div style="font-size:0.7rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.8rem;"><i class="fa-solid fa-hashtag"></i> Numerical Input</div>
      <div style="display:flex;flex-direction:column;gap:0.4rem;">
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;padding:0.4rem 0.7rem;font-size:0.8rem;color:#166534;font-weight:500;"><i class="fa-solid fa-money-bill"></i> Price</div>
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;padding:0.4rem 0.7rem;font-size:0.8rem;color:#166534;font-weight:500;"><i class="fa-solid fa-percent"></i> Discount</div>
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;padding:0.4rem 0.7rem;font-size:0.8rem;color:#166534;font-weight:500;"><i class="fa-solid fa-wallet"></i> Budget</div>
      </div>
    </div>
  </div>
  <div style="display:flex;justify-content:center;margin-bottom:1.5rem;">
    <div style="background:#0f172a;border-radius:10px;padding:1rem 2.5rem;text-align:center;color:white;min-width:280px;">
      <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:#94a3b8;margin-bottom:0.5rem;"><i class="fa-solid fa-code-merge"></i> Hybrid Fusion Layer</div>
      <div style="font-size:0.85rem;font-weight:600;color:#f1f5f9;">GRU Features + Numerical Features</div>
      <div style="font-size:0.72rem;color:#64748b;margin-top:0.3rem;">Linear - ReLU - Dropout - Linear - Sigmoid</div>
    </div>
  </div>
  <div style="text-align:center;font-size:1.3rem;color:#94a3b8;margin-bottom:1.5rem;"><i class="fa-solid fa-arrow-down"></i></div>
  <div style="background:#fef2f2;border:1px solid #fecdd3;border-radius:10px;padding:1.2rem 1.5rem;">
    <div style="font-size:0.7rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.8rem;"><i class="fa-solid fa-star"></i> Final Recommendation Score</div>
    <div style="display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;">
      <div style="background:white;border:1px solid #fecdd3;border-radius:8px;padding:0.5rem 1rem;text-align:center;">
        <div style="font-size:1.1rem;font-weight:700;color:#e63946;">55%</div>
        <div style="font-size:0.72rem;color:#64748b;font-weight:500;">GRU Model</div>
      </div>
      <div style="color:#94a3b8;font-size:1.2rem;">+</div>
      <div style="background:white;border:1px solid #fecdd3;border-radius:8px;padding:0.5rem 1rem;text-align:center;">
        <div style="font-size:1.1rem;font-weight:700;color:#e63946;">25%</div>
        <div style="font-size:0.72rem;color:#64748b;font-weight:500;">Query Relevance</div>
      </div>
      <div style="color:#94a3b8;font-size:1.2rem;">+</div>
      <div style="background:white;border:1px solid #fecdd3;border-radius:8px;padding:0.5rem 1rem;text-align:center;">
        <div style="font-size:1.1rem;font-weight:700;color:#e63946;">20%</div>
        <div style="font-size:0.72rem;color:#64748b;font-weight:500;">Discount Score</div>
      </div>
      <div style="color:#94a3b8;font-size:1.2rem;">=</div>
      <div style="background:#0f172a;border-radius:8px;padding:0.5rem 1.2rem;text-align:center;">
        <div style="font-size:1.1rem;font-weight:700;color:white;">Score Final</div>
        <div style="font-size:0.72rem;color:#94a3b8;font-weight:500;">Ranked Result</div>
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)