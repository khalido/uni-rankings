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
    dimensions = [col for col in df.columns if col not in ["year", "broad_impact",]]
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


def df_filter_country_and_year(df):
    st.subheader("Data")

    all_countries = list(df.country.unique())

    st.text("Select 1 or more countries")
    selected_countries = st.multiselect("Countries", all_countries, ["Australia"])
    df = df.query("country == @selected_countries")

    st.text("Select range of years to look at")
    yr_min = int(df.year.min())
    yr_max = int(df.year.max())
    yr_low, yr_high = st.slider("Years", yr_min, yr_max, (yr_min, yr_max), step=1)

    st.write(yr_low, yr_high)
    df = df.query("@yr_low <= year <= @yr_high")

    st.markdown(
        f"Showing {df.shape[0]} universities from {len(selected_countries)} countries."
    )
    st.dataframe(df)

    return df


def cwur_kaggle(df):
    df = df_filter_country_and_year(df)

    st.subheader("Line plot")
    topN = st.slider(
        "Select top N universities to display", 10, int(df.world_rank.max()), 250
    )
    fig = line_plot(df, N=topN)
    st.plotly_chart(fig)

    st.subheader("Some Viz")
    fig = some_viz(df)
    st.plotly_chart(fig)

    fig = par_plot(df)
    st.plotly_chart(fig)


def cwur_all(df):
    df = df_filter_country_and_year(df)

    st.subheader("Line plot")
    topN = st.slider(
        "Select top N universities to display", 10, int(df.world_rank.max()), 250
    )
    fig = line_plot(df, N=topN)
    st.plotly_chart(fig)

    st.subheader("Some Viz")
    fig = some_viz(df)
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
