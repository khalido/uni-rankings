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


@st.cache
def get_data() -> Dict[str, pd.DataFrame]:
    """returns all the datafiles"""
    df = pd.read_csv("data/cwurData.csv")
    return df


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


def main():
    """the starting point of the app"""
    st.title("University Rankings")

    st.subheader("Data")
    df = get_data()

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

    st.subheader("Some Viz")
    fig = some_viz(df)
    st.plotly_chart(fig)

    fig = par_plot(df)
    st.plotly_chart(fig)


if __name__ == "__main__":
    main()
    st.balloons()
