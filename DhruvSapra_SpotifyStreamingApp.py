# =============================================================================
# DhruvSapra_SpotifyStreamingApp.py
# Author  : Dhruv Sapra
# Dataset : Spotify_Streaming_Performance_Dataset.csv
# Run     : streamlit run DhruvSapra_SpotifyStreamingApp.py
# =============================================================================

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spotify Streaming Analysis — Dhruv Sapra",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

# ---------------------------------------------------------------------------
# Helper: draw a matplotlib figure into a Streamlit column
# ---------------------------------------------------------------------------
def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ===========================================================================
# DATA LOADING & CLEANING  (cached so it only runs once per session)
# ===========================================================================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Strip whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # Flexible column rename via substring matching
    rename_map = {}
    for col in df.columns:
        cl = col.lower()
        if "artist name"            in cl: rename_map[col] = "Artist Name"
        elif cl == "sex":                  rename_map[col] = "Sex"
        elif "country"              in cl: rename_map[col] = "Country of Origin"
        elif "language"             in cl: rename_map[col] = "Primary Language"
        elif "genre"                in cl: rename_map[col] = "Primary Genre"
        elif "artist type"          in cl: rename_map[col] = "Artist Type"
        elif "debut year"           in cl: rename_map[col] = "Debut Year"
        elif "total streams"        in cl: rename_map[col] = "Total Streams (M)"
        elif "lead streams"         in cl: rename_map[col] = "Lead Streams (M)"
        elif "feature streams"      in cl: rename_map[col] = "Feature Streams (M)"
        elif "solo streams ("       in cl: rename_map[col] = "Solo Streams (M)"
        elif "collaborative streams (" in cl: rename_map[col] = "Collab Streams (M)"
        elif "solo stream %"        in cl: rename_map[col] = "Solo Stream %"
        elif "collaborative stream %" in cl: rename_map[col] = "Collab Stream %"
    df.rename(columns=rename_map, inplace=True)

    # Drop full-NaN numeric rows, remove duplicates
    num_cols = ["Total Streams (M)", "Lead Streams (M)", "Feature Streams (M)",
                "Solo Streams (M)", "Collab Streams (M)", "Solo Stream %", "Collab Stream %"]
    existing = [c for c in num_cols if c in df.columns]
    df.dropna(subset=existing, how="all", inplace=True)
    df.drop_duplicates(inplace=True)

    # Feature engineering
    if "Debut Year" in df.columns:
        df["Debut Year"] = pd.to_numeric(df["Debut Year"], errors="coerce")
        df["Years Active"] = 2025 - df["Debut Year"]
        df["Debut Decade"] = (df["Debut Year"] // 10 * 10).astype("Int64").astype(str) + "s"

    return df, existing


# ===========================================================================
# SIDEBAR — file uploader + filters
# ===========================================================================
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/168px-Spotify_logo_without_text.svg.png",
    width=48,
)
st.sidebar.title("🎵 Spotify Analysis")
st.sidebar.markdown("**Author:** Dhruv Sapra")
st.sidebar.markdown("---")

uploaded = st.sidebar.file_uploader(
    "Upload CSV dataset", type=["csv"],
    help="Upload Spotify_Streaming_Performance_Dataset.csv"
)

DATA_PATH = "Spotify_Streaming_Performance_Dataset.csv"
if uploaded is not None:
    DATA_PATH = uploaded

try:
    df_full, existing_numeric = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        "⚠️ Dataset not found.\n\n"
        "Place **Spotify_Streaming_Performance_Dataset.csv** in the same folder as this script, "
        "or upload it using the sidebar uploader."
    )
    st.stop()

# ---- Sidebar filters -------------------------------------------------------
st.sidebar.markdown("### Filters")

all_genres   = sorted(df_full["Primary Genre"].dropna().unique()) if "Primary Genre" in df_full.columns else []
all_countries= sorted(df_full["Country of Origin"].dropna().unique()) if "Country of Origin" in df_full.columns else []
all_types    = sorted(df_full["Artist Type"].dropna().unique()) if "Artist Type" in df_full.columns else []
all_genders  = sorted(df_full["Sex"].dropna().unique()) if "Sex" in df_full.columns else []

sel_genres   = st.sidebar.multiselect("Genre",       all_genres,    default=all_genres)
sel_countries= st.sidebar.multiselect("Country",     all_countries, default=all_countries)
sel_types    = st.sidebar.multiselect("Artist Type", all_types,     default=all_types)
sel_genders  = st.sidebar.multiselect("Gender",      all_genders,   default=all_genders)

# Year range slider
if "Debut Year" in df_full.columns:
    yr_min = int(df_full["Debut Year"].min())
    yr_max = int(df_full["Debut Year"].max())
    yr_range = st.sidebar.slider("Debut Year Range", yr_min, yr_max, (yr_min, yr_max))
else:
    yr_range = (1900, 2025)

# Apply filters
df = df_full.copy()
if sel_genres    and "Primary Genre"      in df.columns: df = df[df["Primary Genre"].isin(sel_genres)]
if sel_countries and "Country of Origin"  in df.columns: df = df[df["Country of Origin"].isin(sel_countries)]
if sel_types     and "Artist Type"        in df.columns: df = df[df["Artist Type"].isin(sel_types)]
if sel_genders   and "Sex"                in df.columns: df = df[df["Sex"].isin(sel_genders)]
if "Debut Year"  in df.columns:
    df = df[(df["Debut Year"] >= yr_range[0]) & (df["Debut Year"] <= yr_range[1])]

st.sidebar.markdown(f"**{len(df):,} artists** after filters")

# ===========================================================================
# MAIN HEADER
# ===========================================================================
st.title("🎵 Spotify Streaming Performance Dashboard")
st.markdown(
    "Interactive EDA of **500 global Spotify artists** · "
    "Dataset: [Kaggle — srisyra02](https://www.kaggle.com/datasets/srisyra02/spotify-music-artist-streaming-analytics)"
)
st.markdown("---")

# ===========================================================================
# TAB LAYOUT
# ===========================================================================
tabs = st.tabs([
    "📊 Overview",
    "🎸 Genre",
    "🚻 Gender",
    "🌍 Country",
    "🎤 Artist Type",
    "📅 Debut Decade",
    "🤝 Solo vs Collab",
    "🔥 Correlations",
    "💡 Key Insights",
])

# ---------------------------------------------------------------------------
# TAB 0 — OVERVIEW (KPI cards + dataset preview)
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Dataset Overview")

    total_artists  = len(df)
    total_streams  = df["Total Streams (M)"].sum() if "Total Streams (M)" in df.columns else 0
    avg_streams    = df["Total Streams (M)"].mean() if "Total Streams (M)" in df.columns else 0
    num_genres     = df["Primary Genre"].nunique() if "Primary Genre" in df.columns else 0
    num_countries  = df["Country of Origin"].nunique() if "Country of Origin" in df.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🎤 Artists",         f"{total_artists:,}")
    c2.metric("🎵 Total Streams",   f"{total_streams:,.0f}M")
    c3.metric("📈 Avg Streams",     f"{avg_streams:,.0f}M")
    c4.metric("🎸 Genres",          num_genres)
    c5.metric("🌍 Countries",       num_countries)

    st.markdown("---")

    # Summary statistics table
    st.subheader("Summary Statistics")
    if existing_numeric:
        cols_present = [c for c in existing_numeric if c in df.columns]
        summary = df[cols_present].describe().T.round(2)
        summary["median"] = df[cols_present].median().round(2)
        st.dataframe(summary, use_container_width=True)

    st.markdown("---")
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 1 — GENRE
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Streams by Primary Genre")

    if "Primary Genre" in df.columns and "Total Streams (M)" in df.columns:
        n_genres = st.slider("Show top N genres", 5, min(20, df["Primary Genre"].nunique()), 10, key="n_genres")
        genre_streams = (
            df.groupby("Primary Genre")["Total Streams (M)"]
            .sum().sort_values(ascending=False).head(n_genres)
        )

        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.barplot(x=genre_streams.values, y=genre_streams.index, palette="Blues_d", ax=ax)
            ax.set_title(f"Top {n_genres} Genres by Total Streams", fontweight="bold")
            ax.set_xlabel("Total Streams (Millions)")
            ax.set_ylabel("")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            show_fig(fig)

        with col_b:
            st.markdown("**Genre Totals (M)**")
            st.dataframe(
                genre_streams.reset_index().rename(columns={"Total Streams (M)": "Total (M)"}),
                use_container_width=True, hide_index=True
            )

        # Box plot distribution
        st.markdown("#### Stream Distribution per Genre (Box Plot)")
        top_g = genre_streams.index.tolist()
        df_g  = df[df["Primary Genre"].isin(top_g)]
        fig2, ax2 = plt.subplots(figsize=(11, 5))
        sns.boxplot(data=df_g, x="Primary Genre", y="Total Streams (M)",
                    palette="Set3", order=top_g, ax=ax2)
        ax2.set_title(f"Total Streams Distribution — Top {n_genres} Genres", fontweight="bold")
        ax2.tick_params(axis="x", rotation=35)
        ax2.set_xlabel("")
        show_fig(fig2)

# ---------------------------------------------------------------------------
# TAB 2 — GENDER
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Streams by Gender")

    if "Sex" in df.columns and "Total Streams (M)" in df.columns:
        gender_agg = (
            df.groupby("Sex")["Total Streams (M)"]
            .agg(Total="sum", Average="mean", Artists="count")
            .round(2).sort_values("Total", ascending=False)
        )

        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=gender_agg.index, y=gender_agg["Total"],
                        palette="Set2", ax=ax)
            ax.set_title("Total Streams by Gender", fontweight="bold")
            ax.set_ylabel("Total Streams (Millions)")
            show_fig(fig)

        with col_b:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=gender_agg.index, y=gender_agg["Average"],
                        palette="Set2", ax=ax)
            ax.set_title("Avg Streams per Artist by Gender", fontweight="bold")
            ax.set_ylabel("Avg Streams (Millions)")
            show_fig(fig)

        st.dataframe(gender_agg.reset_index(), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 3 — COUNTRY
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Streams by Country of Origin")

    if "Country of Origin" in df.columns and "Total Streams (M)" in df.columns:
        n_ctry = st.slider("Show top N countries", 5, min(30, df["Country of Origin"].nunique()), 15, key="n_ctry")
        country_streams = (
            df.groupby("Country of Origin")["Total Streams (M)"]
            .sum().sort_values(ascending=False).head(n_ctry)
        )

        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.barplot(x=country_streams.values, y=country_streams.index,
                        palette="Greens_d", ax=ax)
            ax.set_title(f"Top {n_ctry} Countries by Total Streams", fontweight="bold")
            ax.set_xlabel("Total Streams (Millions)")
            ax.set_ylabel("")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            show_fig(fig)

        with col_b:
            st.markdown("**Country Totals (M)**")
            st.dataframe(
                country_streams.reset_index().rename(columns={"Total Streams (M)": "Total (M)"}),
                use_container_width=True, hide_index=True
            )

# ---------------------------------------------------------------------------
# TAB 4 — ARTIST TYPE
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Streams by Artist Type")

    if "Artist Type" in df.columns and "Total Streams (M)" in df.columns:
        type_agg = (
            df.groupby("Artist Type")["Total Streams (M)"]
            .agg(Total="sum", Average="mean", Artists="count")
            .round(2).sort_values("Total", ascending=False)
        )

        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(
                type_agg["Total"],
                labels=type_agg.index,
                autopct="%1.1f%%",
                startangle=140,
                colors=sns.color_palette("pastel", len(type_agg))
            )
            ax.set_title("Share of Total Streams\nby Artist Type", fontweight="bold")
            show_fig(fig)

        with col_b:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=type_agg.index, y=type_agg["Average"],
                        palette="Purples_d", ax=ax)
            ax.set_title("Avg Streams per Artist\nby Artist Type", fontweight="bold")
            ax.set_ylabel("Avg Streams (Millions)")
            show_fig(fig)

        st.dataframe(type_agg.reset_index(), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 5 — DEBUT DECADE
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Streams by Debut Decade")

    if "Debut Decade" in df.columns and "Total Streams (M)" in df.columns:
        decade_agg = (
            df.groupby("Debut Decade")["Total Streams (M)"]
            .agg(Total="sum", Average="mean", Artists="count")
            .round(2).sort_index()
        )

        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig, ax = plt.subplots(figsize=(9, 5))
            bars = ax.bar(decade_agg.index, decade_agg["Total"],
                          color=sns.color_palette("rocket", len(decade_agg)))
            ax.set_title("Total Streams by Debut Decade", fontweight="bold")
            ax.set_xlabel("Debut Decade")
            ax.set_ylabel("Total Streams (Millions)")
            ax.tick_params(axis="x", rotation=30)
            # Annotate artist counts
            for bar, cnt in zip(bars, decade_agg["Artists"]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + decade_agg["Total"].max() * 0.01,
                        f"n={int(cnt)}", ha="center", va="bottom", fontsize=9)
            show_fig(fig)

        with col_b:
            st.markdown("**Decade Summary**")
            st.dataframe(decade_agg.reset_index(), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 6 — SOLO vs COLLABORATIVE
# ---------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Solo vs. Collaborative Streaming Split")

    solo_col   = "Solo Streams (M)"
    collab_col = "Collab Streams (M)"
    pct_solo   = "Solo Stream %"

    if solo_col in df.columns and collab_col in df.columns:
        total_solo   = df[solo_col].sum()
        total_collab = df[collab_col].sum()
        pct          = total_solo / (total_solo + total_collab) * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("Solo Streams",  f"{total_solo:,.0f}M")
        m2.metric("Collab Streams",f"{total_collab:,.0f}M")
        m3.metric("Solo Share",    f"{pct:.1f}%")

        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie([total_solo, total_collab],
                   labels=["Solo", "Collaborative"],
                   autopct="%1.1f%%", startangle=90,
                   colors=["#4C72B0", "#DD8452"])
            ax.set_title("Overall Solo vs Collaborative\nStream Share", fontweight="bold")
            show_fig(fig)

        with col_b:
            if pct_solo in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(df[pct_solo].dropna(), bins=20, kde=True,
                             color="#4C72B0", ax=ax)
                ax.set_title("Distribution of Solo Stream %", fontweight="bold")
                ax.set_xlabel("Solo Stream %")
                ax.set_ylabel("Number of Artists")
                show_fig(fig)

        # By Artist Type breakdown
        if "Artist Type" in df.columns:
            st.markdown("#### Solo vs Collab by Artist Type")
            split_type = (
                df.groupby("Artist Type")[[solo_col, collab_col]]
                .sum().rename(columns={solo_col: "Solo (M)", collab_col: "Collab (M)"})
            )
            fig, ax = plt.subplots(figsize=(7, 4))
            split_type.plot(kind="bar", color=["#4C72B0", "#DD8452"],
                            edgecolor="white", ax=ax)
            ax.set_title("Solo vs Collaborative Streams by Artist Type", fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Streams (Millions)")
            ax.tick_params(axis="x", rotation=25)
            ax.legend(title="Stream Type")
            show_fig(fig)
            st.dataframe(split_type.reset_index(), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 7 — CORRELATIONS
# ---------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Correlation Analysis")

    corr_cols = [c for c in existing_numeric if c in df.columns]
    if len(corr_cols) >= 2:
        corr_matrix = df[corr_cols].corr()

        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.markdown("#### Pearson Correlation Heatmap")
            mask_arr = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, mask=mask_arr, annot=True, fmt=".2f",
                        cmap="coolwarm", center=0, linewidths=0.5,
                        ax=ax, square=True, cbar_kws={"shrink": 0.8})
            ax.set_title("Correlation Heatmap", fontweight="bold")
            show_fig(fig)

        with col_b:
            st.markdown("#### Correlation Matrix (raw)")
            st.dataframe(corr_matrix.round(2), use_container_width=True)

        # Scatter: total streams vs years active
        if "Years Active" in df.columns and "Total Streams (M)" in df.columns:
            st.markdown("#### Total Streams vs Years Active")
            hue = "Artist Type" if "Artist Type" in df.columns else None
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.scatterplot(data=df, x="Years Active", y="Total Streams (M)",
                            hue=hue, alpha=0.7, s=55, ax=ax)
            ax.set_title("Total Streams vs Years Active", fontweight="bold")
            if hue:
                ax.legend(title=hue)
            show_fig(fig)

# ---------------------------------------------------------------------------
# TAB 8 — KEY INSIGHTS
# ---------------------------------------------------------------------------
with tabs[8]:
    st.subheader("💡 Key Insights")

    insights = []

    if "Primary Genre" in df.columns and "Total Streams (M)" in df.columns:
        top_genre     = df.groupby("Primary Genre")["Total Streams (M)"].sum().idxmax()
        top_genre_val = df.groupby("Primary Genre")["Total Streams (M)"].sum().max()
        insights.append(("🎸 Top Genre",
                          f"**{top_genre}** leads with **{top_genre_val:,.0f}M** total streams."))

    if "Sex" in df.columns and "Total Streams (M)" in df.columns:
        top_gender     = df.groupby("Sex")["Total Streams (M)"].sum().idxmax()
        top_gender_val = df.groupby("Sex")["Total Streams (M)"].sum().max()
        insights.append(("🚻 Top Gender",
                          f"**{top_gender}** artists lead with **{top_gender_val:,.0f}M** combined streams."))

    if "Country of Origin" in df.columns and "Total Streams (M)" in df.columns:
        top_country     = df.groupby("Country of Origin")["Total Streams (M)"].sum().idxmax()
        top_country_val = df.groupby("Country of Origin")["Total Streams (M)"].sum().max()
        insights.append(("🌍 Top Country",
                          f"**{top_country}** dominates with **{top_country_val:,.0f}M** streams."))

    if "Artist Type" in df.columns and "Total Streams (M)" in df.columns:
        type_avg = df.groupby("Artist Type")["Total Streams (M)"].mean()
        top_type = type_avg.idxmax()
        insights.append(("🎤 Highest Avg",
                          f"**{top_type}** artists average the most streams: **{type_avg[top_type]:,.0f}M** per artist."))

    if "Debut Decade" in df.columns and "Total Streams (M)" in df.columns:
        top_decade = df.groupby("Debut Decade")["Total Streams (M)"].sum().idxmax()
        insights.append(("📅 Peak Decade",
                          f"Artists who debuted in the **{top_decade}** generate the most total streams."))

    if "Solo Streams (M)" in df.columns and "Collab Streams (M)" in df.columns:
        ts = df["Solo Streams (M)"].sum()
        tc = df["Collab Streams (M)"].sum()
        pp = ts / (ts + tc) * 100
        insights.append(("🤝 Solo/Collab Split",
                          f"Solo streams make up **{pp:.1f}%** of total streams "
                          f"(**{ts:,.0f}M** solo vs **{tc:,.0f}M** collaborative)."))

    insights.append(("📋 Dataset Size",
                      f"**{len(df):,}** artists across "
                      f"**{df['Country of Origin'].nunique() if 'Country of Origin' in df.columns else 'N/A'}** countries "
                      f"and **{df['Primary Genre'].nunique() if 'Primary Genre' in df.columns else 'N/A'}** genres analysed."))

    for icon_label, body in insights:
        with st.container(border=True):
            st.markdown(f"**{icon_label}**")
            st.markdown(body)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:13px;'>"
    "Spotify Streaming Performance Analysis · Dhruv Sapra · 2025 · "
    "<a href='https://www.kaggle.com/datasets/srisyra02/spotify-music-artist-streaming-analytics' target='_blank'>Dataset</a>"
    "</div>",
    unsafe_allow_html=True,
)
