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


def main():
    """the starting point of the app"""
    st.title("University Rankings")

    st.subheader("Data")
    data = get_data()
    df = data["cwur"]

    all_countries = list(df.country.unique())
    with st.sidebar:
        st.header("Options")
        selected_countries = st.multiselect("Countries", all_countries, ["Australia"])
        df = df.query("country == @selected_countries")

        yr_low, yr_high = st.slider(
            "Years", int(df.year.min()), int(df.year.max()), (2013, 2015), step=1
        )

        st.write(yr_low, yr_high)
        df = df.query("@yr_low <= year <= @yr_high")

    # some selectors
    left_col, right_col = st.beta_columns(2)
    with left_col:
        st.write("left!")
    with right_col:
        st.write("right")

    st.markdown(
        f"Showing {df.shape[0]} universities from {len(selected_countries)} countries."
    )
    st.dataframe(df)

    st.subheader("Line plot")
    topN = st.slider("Select top N universities to display", 10, int(df.world_rank.max()), 250)
    fig = line_plot(df, N=topN)
    st.plotly_chart(fig)

    st.subheader("Some Viz")
    fig = some_viz(df)
    st.plotly_chart(fig)

    fig = par_plot(df)
    st.plotly_chart(fig)


if __name__ == "__main__":
    main()
    st.balloons()
