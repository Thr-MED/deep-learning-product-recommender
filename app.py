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


@st.cache_data
def load_products():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["product_name", "price"])
    df["discount_percentage"] = df["discount_percentage"].fillna(0)
    df["category"] = df["category"].fillna("Autre")
    return df


# =========================
# HEADER
# =========================

st.title("🛒 Live Electroplanet Promotion Recommender")

st.markdown(
    """
    This application recommends Electroplanet promotional products using a **Hybrid GRU Deep Learning Model**.

    The recommendation considers:
    - user search query,
    - product category,
    - product price,
    - user budget,
    - discount percentage,
    - GRU model prediction.
    """
)


# =========================
# LOAD DATA
# =========================

df = load_products()

if df.empty:
    st.error("No product data found. Please check the CSV file.")
    st.stop()


# =========================
# SIDEBAR
# =========================

st.sidebar.header("🔎 User Preferences")

user_query = st.sidebar.text_input(
    "What are you looking for?",
    value="réfrigérateur pas cher pour aid al adha"
)

budget = st.sidebar.number_input(
    "Maximum budget (DH)",
    min_value=100,
    max_value=50000,
    value=9000,
    step=100
)

min_discount = st.sidebar.slider(
    "Minimum discount (%)",
    min_value=0,
    max_value=80,
    value=5,
    step=1
)

top_n = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=30,
    value=10,
    step=1
)

categories = ["All"] + sorted(df["category"].dropna().unique().tolist())

category_filter = st.sidebar.selectbox(
    "Category filter",
    categories
)

show_images = st.sidebar.checkbox(
    "Show product images",
    value=True
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Live data was collected from Electroplanet using Selenium. "
    "The app uses the latest cached scraped data."
)

st.sidebar.caption(
    "Images are loaded from the scraped Electroplanet product URLs."
)


# =========================
# DATA STATUS
# =========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Products collected", len(df))

with col2:
    st.metric("Categories", df["category"].nunique())

with col3:
    st.metric("Average discount", f"{df['discount_percentage'].mean():.1f}%")

with col4:
    st.metric("Max discount", f"{df['discount_percentage'].max():.0f}%")

st.caption(f"Last app refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# =========================
# RECOMMENDATIONS
# =========================

st.markdown("---")
st.header("🎯 Recommended Promotions")

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
    display_cols = [
        "product_name",
        "category",
        "price",
        "old_price",
        "discount_percentage",
        "model_score",
        "query_relevance_score",
        "discount_score",
        "final_score",
        "source_page"
    ]

    st.dataframe(
        recommendations[display_cols],
        use_container_width=True
    )

    st.subheader("🏆 Top Recommended Products")

    for i, (_, row) in enumerate(recommendations.iterrows(), start=1):
        with st.container():
            col_img, col_info = st.columns([1, 4])

            with col_img:
                image_url = str(row.get("image_url", ""))

                # Performance improvement:
                # Load images only for top 3 products.
                if show_images and image_url.startswith("http"):
                    try:
                        st.image(image_url, width=100)
                    except Exception:
                        st.write("🛒")
                else:
                    st.write("🛒")

            with col_info:
                st.markdown(f"### {i}. {row['product_name']}")
                st.write(f"**Category:** {row['category']}")
                st.write(
                    f"**Price:** {row['price']} DH | "
                    f"**Old price:** {row['old_price']} DH | "
                    f"**Discount:** {row['discount_percentage']}%"
                )
                st.write(
                    f"**GRU model score:** {row['model_score']} | "
                    f"**Query relevance:** {row['query_relevance_score']} | "
                    f"**Discount score:** {row['discount_score']} | "
                    f"**Final score:** {row['final_score']}"
                )

                product_url = str(row.get("product_url", ""))
                if product_url.startswith("http"):
                    st.markdown(f"[Open product page]({product_url})")

            st.markdown("---")


# =========================
# ANALYTICS
# =========================

st.header("📊 Data Analysis")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Products by Category")
    category_counts = df["category"].value_counts().reset_index()
    category_counts.columns = ["category", "count"]

    fig_cat = px.bar(
        category_counts,
        x="category",
        y="count",
        title="Number of Products per Category"
    )

    st.plotly_chart(fig_cat, use_container_width=True)

with col_b:
    st.subheader("Discount Distribution")

    fig_discount = px.histogram(
        df,
        x="discount_percentage",
        nbins=20,
        title="Distribution of Discount Percentages"
    )

    st.plotly_chart(fig_discount, use_container_width=True)


st.subheader("Price vs Discount")

fig_scatter = px.scatter(
    df,
    x="price",
    y="discount_percentage",
    color="category",
    hover_data=["product_name", "source_page"],
    title="Price vs Discount by Category"
)

st.plotly_chart(fig_scatter, use_container_width=True)


# =========================
# MODEL EXPLANATION
# =========================

st.markdown("---")
st.header("🧠 Deep Learning Model Explanation")

st.markdown(
    """
    The project uses a **Hybrid GRU Recommendation Model**.

    ### Text input

    The model receives:

    `user query + product name + product category`

    This text passes through:

    `Embedding Layer → GRU Layer`

    The GRU learns patterns from product names and user needs.

    ### Numerical input

    The model also receives:

    - product price,
    - discount percentage,
    - user budget,
    - minimum discount,
    - budget match,
    - discount match.

    ### Final recommendation score

    The final score is calculated as:

    `Final Score = 0.55 × GRU Model Score + 0.25 × Query Relevance + 0.20 × Discount Score`

    So the system uses Deep Learning while also giving importance to the promotion discount.
    """
)