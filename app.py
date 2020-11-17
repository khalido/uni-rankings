# user interface
import streamlit as st

# data wrangling
import numpy as np
import pandas as pd

# utils
import os
import random
from typing import Dict, List

# viz
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# ml stuff
from sklearn.cluster import Birch
from sklearn.decomposition import PCA

st.set_page_config(
    page_title=f"University Rankings Exploration",
)

DEBUG = True


@st.cache
def get_data(verbose=False) -> Dict[str, pd.DataFrame]:
    base_url = "https://github.com/khalido/uni-rankings/raw/main/data/"
    data_urls = [
        ("shanghai", "shanghaiData.csv"),
        ("cwur", "cwurData.csv"),
        ("times", "timesData.csv"),
        ("cwur_all", "cwurAll.csv"),
    ]

    country_lookup = pd.read_csv(base_url + "school_and_country_table.csv").rename(
        columns={"school_name": "university"}
    )

    data = {}
    for name, url in data_urls:
        url = f"data/{url}"

        df = pd.read_csv(url)

        # rename to make col names consistent
        df = df.rename(
            columns={"institution": "university", "university_name": "university"}
        )

        if name == "shanghai":
            # add country col to shanghai data
            df = df.merge(country_lookup, on="university", how="left")

        print(f"**{name}** data has {df.shape} values.")
        data[name] = df

    return data


def some_viz(df):
    """takes in a df and returns a fig"""
    fig = px.scatter(
        df, x="quality_of_education", y="score", color="country", size="citations"
    )
    fig.update_layout(
        title="Quality of Education Vs Score with Country & #Students",
        xaxis_title="Quality of Education",
        yaxis_title="Score",
    )
    return fig


def par_plot(df):
    dimensions = [
        col
        for col in df.columns
        if col
        not in [
            "year",
            "broad_impact",
        ]
    ]
    fig = px.parallel_coordinates(
        df,
        color="world_rank",
        dimensions=dimensions,
        color_continuous_scale=px.colors.diverging.Tealrose,
        color_continuous_midpoint=50,
    )
    return fig


def radar_plot(df):
    all_unis = list(df.institution.unique())
    unis = st.selectbox("Select a few unis", all_unis)


def line_plot(df, N=200, countries=False):
    """simple line plot of ranking per year"""
    d = df.query("world_rank <= @N")

    if countries:
        d = d.query("country == @countries")
        if d.shape[0] < 1:
            return "error, selected zero countries"

    fig = px.line(
        d, x="year", y="world_rank", title="Rank per year", color="university"
    )
    fig.update_yaxes(range=[d.world_rank.max() + 1, 0])
    return fig


def make_corr(df, max_rank=100, nrows=3, ncols=3):
    """takes in df, returns corr plot"""
    st.write("Shows a correlation heatmap for each year.")

    years = sorted(df.year.unique())

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12), sharey=True, sharex=True)
    fig.suptitle(f"Correlation plot for top {max_rank} unis")

    for year, ax in zip(years, axes.flat):
        d = df.query("year == @year and world_rank <=@max_rank")
        corr = d.corr()

        ax.set_title(f"{year}")
        sns.heatmap(corr, ax=ax)

    return fig


def df_filter_country_and_year(df):
    with st.sidebar:
        st.subheader("Data")
        df_org = df.copy()

        all_countries = list(df.country.unique())

        st.text("Select 1 or more countries")
        select_all = st.checkbox("tick to select all countries")
        if select_all:
            initial_countries = all_countries
        else:
            initial_countries = ["Australia"]

        selected_countries = st.multiselect(
            "Countries", all_countries, initial_countries
        )
        df = df.query("country == @selected_countries")

        st.text("Select range of years to look at")
        yr_min = int(df.year.min())
        yr_max = int(df.year.max())
        yr_low, yr_high = st.slider("Years", yr_min, yr_max, (yr_min, yr_max), step=1)

        st.write(yr_low, yr_high)
        df = df.query("@yr_low <= year <= @yr_high")

        topN = st.slider(
            "Select top N universities to display",
            10,
            int(df.world_rank.max()),
            100,
            10,
        )

        df = df.query("world_rank <= @topN")

        st.markdown(
            f"Selected {df.shape[0]} universities from {len(selected_countries)} countries."
        )
        if df.shape[0] < 1:
            st.error(
                "You have selected no universities!\nPlease select a better vars above. Using the original unfiltered data for now."
            )
            df = df_org

    if DEBUG:
        st.write(df)

    return df


def bar_chart(df, max_rank=1000, year=2014):
    """shows top countries"""
    df = df.query("world_rank <= @max_rank and year==@year")
    order = df["country"].value_counts(ascending=False).index[:20]
    st.write(df)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Yr {year}: Top {max_rank:,} universities (top 20 countries shown)")
    sns.countplot(y="country", data=df, order=order, ax=ax)
    return fig


def pca_cluster(df, n_components=4, n_clusters=5):

    d = df.copy()  # copy to add temp cols to df

    # filter for year
    col1, col2 = st.beta_columns(2)
    with col1:
        yr = st.selectbox("Select Year", np.arange(df.year.min(), df.year.max() + 1))
    with col2:
        n_clusters = st.slider("Select number of clusters", 2,15,5)
    d = d.query("year == @yr")

    # select only numeric cols, drop broad_impact as its full of nulls
    X = d.drop(columns="broad_impact").select_dtypes(np.number)

    with st.spinner("Computing pca and clustering..."):
        pca = PCA(n_components=n_components)
        pca.fit(X)

        x = pca.transform(X)

        # clustering
        y_pred = Birch(n_clusters=n_clusters).fit_predict(X)

        # viz part
        d["marker_size"] = (d.world_rank - d.world_rank.max() - 1) * -1
        d["x"] = x[:, 0]
        d["y"] = x[:, 1]
        d["cluster"] = y_pred

    fig = px.scatter(
        d,
        x="x",
        y="y",
        size="marker_size",
        color="cluster",
        hover_name="university",
        title=f"Clustered PCA projection of top {d.world_rank.max()} unis",
        hover_data={
            "world_rank": True,
            "national_rank": True,
            "university": False,
            "country": True,
            "x": False,
            "y": False,
            "marker_size": False,
        },
    )

    fig.update_layout(coloraxis_showscale=False)
    return fig


##################
# streamlit pagees
##################


def cwur_kaggle(df):
    df = df_filter_country_and_year(df)

    st.subheader("PCA & Clustering")
    
    fig = pca_cluster(df)
    st.plotly_chart(fig)

    st.subheader("Line plot")

    fig = line_plot(df)
    st.plotly_chart(fig)

    st.subheader("Some Viz")
    fig = some_viz(df)
    st.plotly_chart(fig)

    st.subheader("Correlation heatmap")
    fig = make_corr(df, max_rank=2000, nrows=2, ncols=2)
    st.pyplot(fig)

    fig = par_plot(df)
    st.plotly_chart(fig)


def cwur_all(df):
    df = df_filter_country_and_year(df)

    st.subheader("Country distribution")

    fig = bar_chart(df)
    st.pyplot(fig)

    st.subheader("Line plot")

    fig = line_plot(df)
    st.plotly_chart(fig)

    fig = par_plot(df)
    st.plotly_chart(fig)


def compare_rankings(data):
    st.header("Ranking vs Rankings")
    pass


def main():
    """the starting point of the app"""
    st.title("University Rankings")

    st.sidebar.title("Choose Dataset")

    app_modes = [
        "CWUR Kaggle",
        "CWUR All years",
        "Shanghai",
        "Times",
        "Compare Rankings",
    ]
    app_mode = st.sidebar.radio("", app_modes)

    data = get_data()

    st.header(app_mode)

    if app_mode == app_modes[0]:
        cwur_kaggle(data["cwur"])
    elif app_mode == app_modes[1]:
        cwur_all(data["cwur_all"])
    elif app_mode == app_modes[2]:
        pass
    elif app_mode == app_modes[3]:
        pass
    elif app_mode == app_modes[4]:
        compare_rankings(data)


if __name__ == "__main__":
    main()
