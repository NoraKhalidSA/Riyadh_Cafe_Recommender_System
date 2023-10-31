import streamlit as st
import plotly
import plotly.express as px
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"
import seaborn as sns
import os
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Read the dataset
df = pd.read_csv('Riyadh_Caffee.csv', index_col=None)

# Create the cafe_pivot DataFrame for the recommender system
cafe_pivot = df.pivot_table(columns='userId', index='coffeeName', values="rating")
cafe_pivot.fillna(0, inplace=True)

# Convert cafe_pivot to a sparse matrix
cafe_sparse = csr_matrix(cafe_pivot.values)

# Create and fit the NearestNeighbors model
model = NearestNeighbors(algorithm='brute')
model.fit(cafe_sparse)

# Set page title and favicon
st.set_page_config(page_title='Coffee Recommender', page_icon='☕️')

# Set page background color
st.markdown(
    """
    <style>
    body {
        background-color: #FFECD7;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set header and subheader with coffee-themed font and color
st.markdown(
    """
    <style>
    .coffee-header {
        font-family: 'Pacifico', cursive;
        color: #4E3629;
        font-size: 36px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .coffee-subheader {
        font-family: 'Lato', sans-serif;
        color: #4E3629;
        font-size: 18px;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.image('coffee_image.png', caption='Image source: Unsplash', use_column_width=True)

st.markdown('<p class="coffee-header">Welcome</p>', unsafe_allow_html=True)
st.markdown('<p class="coffee-subheader">Find your perfect cup of coffee!</p>', unsafe_allow_html=True)
st.markdown('<p class="coffee-subheader">The recommender system developed for this project is based on Collaborative Filtering</p>', unsafe_allow_html=True)
st.markdown('<p class="coffee-subheader">The goal of the system is to provide personalized recommendations for users based on their preferences and the preferences of similar users.</p>', unsafe_allow_html=True)


# Set button style with coffee-themed color
st.markdown(
    """
    <style>
    .coffee-button {
        background-color: #4E3629;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
    }
    .coffee-button:hover {
        background-color: #63483C;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set the plot parameters
p1 = {'lat': 24.719462, 'lon': 46.719108}  # all boroughs
fig = px.scatter_mapbox(df,
                        lat=df['lan'],
                        lon=df['lon'],
                        center=p1,
                        color=df['rating'],
                        color_continuous_scale='YlOrBr',  # Set the color scale to brown shades
                        zoom=9,
                        mapbox_style="open-street-map",
                        title='Riyadh Cafes'
                       )


cafe_vs_rating = df.groupby(['coffeeName', 'rating']).size().unstack().fillna(0)
# Let's calculate the Weighted Average for dataframe rows
def Weighted_Average(df):
    x = []
    for i in range(0, df.shape[0]):
        x.append((np.average(df.iloc[i].index, weights=df.iloc[i].values, axis=0)).round(2))
    return x

# Weighted Average calculation for each cafe_vs_rating row
cafe_vs_rating['weightedAverage'] = Weighted_Average(cafe_vs_rating)
cafe_vs_rating.sort_values('weightedAverage', ascending=False).head()

# Set the color scale to shades of brown
color_scale = px.colors.sequential.YlOrBr

# Create the figure for the top 15 cafes with the highest weighted averages
fig1 = px.bar(cafe_vs_rating, x=cafe_vs_rating['weightedAverage'].nlargest(15).index,
              y=cafe_vs_rating['weightedAverage'].nlargest(15),
              text=cafe_vs_rating['weightedAverage'].nlargest(15),
              labels={"x": "Cafe", 'y': 'Weighted Rating Averages'},
              color=cafe_vs_rating['weightedAverage'].nlargest(15),
              color_continuous_scale=color_scale
              )
fig1.update_traces(textposition='outside')
fig1.update_layout(title_text='Top 15 Cafes with the Highest Weighted Averages',
                   title_x=0.5, title_font=dict(size=24))
fig1.update_traces(marker=dict(line=dict(color='#000000', width=2)))
fig1.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

# Create the figure for the top 15 cafes with the smallest weighted averages
fig2 = px.bar(cafe_vs_rating, x=cafe_vs_rating['weightedAverage'].nsmallest(15).index,
              y=cafe_vs_rating['weightedAverage'].nsmallest(15),
              text=cafe_vs_rating['weightedAverage'].nsmallest(15),
              labels={"x": "Cafe", 'y': 'Weighted Rating Averages'},
              color=cafe_vs_rating['weightedAverage'].nsmallest(15),
              color_continuous_scale=color_scale
              )
fig2.update_traces(textposition='outside')
fig2.update_layout(title_text='Top 15 Cafes with the Smallest Weighted Averages',
                   title_x=0.5, title_font=dict(size=24))
fig2.update_traces(marker=dict(line=dict(color='#000000', width=2)))
fig2.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

# Calculate the frequency of ratings
rating_val_count = df.rating.value_counts()

# Set the color scale to shades of brown
color_scale = px.colors.sequential.YlOrBr

# Create the figure for the frequency of ratings
fig3 = px.bar(rating_val_count, x=rating_val_count.index, y=rating_val_count, text=rating_val_count,
             labels={"index": "Ratings", 'y': 'Number of Ratings'},
             color=rating_val_count,
             color_continuous_scale=color_scale
             )
fig3.update_traces(textposition='outside')
fig3.update_layout(title_text='Frequency of the Ratings',
                  title_x=0.5, title_font=dict(size=24))
fig3.update_traces(marker=dict(line=dict(color='#000000', width=2)))
fig3.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

# Display the plot in Streamlit
st.plotly_chart(fig)
st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)

# Add a sidebar for user input
with st.sidebar:
    st.header('User Input')
    cafe_index = st.number_input('Enter the index of the cafe:', min_value=0, max_value=len(cafe_pivot)-1, step=1)

# Calculating distances and suggestions
distances, suggestions = model.kneighbors(cafe_pivot.iloc[cafe_index, :].values.reshape(1, -1))

st.header('Recommended Cafes')

# Display the recommendations in an organized manner
for i in range(len(suggestions)):
    st.write(cafe_pivot.index[suggestions[i]])
