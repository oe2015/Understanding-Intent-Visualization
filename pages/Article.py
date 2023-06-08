import streamlit as st
import seaborn as sns

import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import MultiLabelBinarizer
import json
import torch
from torch.utils.data import  DataLoader

# from download_button import download_button

from newspaper import Article

flag = 0
st.set_page_config(
    page_title="News Framing Demo",
    page_icon="📰",
)
@st.cache_resource
def fetch_load():

    # Define the path to your JSONL file
    file_path = "newfile.jsonl"

    # Define the desired batch size
    batch_size = 10000000

    # Create an empty list to store the DataFrame chunks
    df_chunks = []

    # Read the JSONL file in batches
    with open(file_path, "r") as f:
        # Read a batch of lines from the file
        lines = f.readlines(batch_size)

        # Convert the lines to JSON objects
        sample_data = [json.loads(line) for line in lines]

        # Convert problematic column to string type
        for data in sample_data:
            if "labels" in data:
                data["labels"] = str(data["labels"])

        # Create a DataFrame from the batch of JSON objects
        df = pd.DataFrame(sample_data)

        # Append the DataFrame to the list of chunks
        df_chunks.append(df)

    # Concatenate the DataFrame chunks into a single DataFrame
    sample_data_df = pd.concat(df_chunks)

    # Display the DataFrame
    return sample_data_df


sample_data_df = fetch_load()
sample_data_df = sample_data_df.drop_duplicates()
# st.write(sample_data_df)

def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

_max_width_()

c30, c31, c32 = st.columns([20, 1, 3])

with c30:
    st.title("News Framing")
    st.header("")

with st.expander("About this app", expanded=False):

    st.write(
        """     
-   The goal of this app is to detect the framing of a text. The model was trained on English news articles.
-   The model is able to detect the existence of a frame or multiple frames out of 14 possible frames:
    - Economic
    - Capacity and Resources
    - Morality
    - Fairness and Equality
    - Legality Constitutionality and Jurisprudence
    - Policy Prescription and Evaluation
    - Crime and Punishment
    - Security and Defense
    - Health and Safety
    - Quality of Life
    - Cultural Identity
    - Public Opinion
    - Political
    - External Regulation and Reputation

-   App created by Tarek Mahmoud
	    """
    )

    st.markdown("")

st.markdown("")
st.markdown("## Demo! ")
url = ""

cx, cy, cz = st.columns([5, 2, 5])

with st.form(key="my_form"):
    url = st.text_input(
        "Enter the article url to get framings",
    )

    submit_button = st.form_submit_button(label="Get news framing!")

if not submit_button:
    st.stop()

if len(url) > 0:
    article = Article(url)
    try:
        article.download()
        article.parse()
        doc = article.text
        st.warning("Article successfully processed.")
        flag = 1 
    except:
        st.warning("Access denied.")
    if flag:
        article.parse()
        doc = article.text
        with st.expander("Read the article", expanded=False):
            st.write(doc)
        mask = sample_data_df["link"] == url

        # Apply the mask to retrieve the filtered rows
        labels = sample_data_df[mask]
        
        labels = labels["labels"].to_numpy()
        labels = eval(labels[0])
        df = pd.DataFrame(labels, columns =["labels","probability"])
        
        x = df["probability"]
        y = df["labels"]
        chart = alt.Chart(df).mark_bar().encode(
            x = 'probability',
            y = 'labels',
            color=alt.Color('labels', scale=alt.Scale(scheme='category10')),
            tooltip=['probability', 'labels']
        ).properties(
            width=600,
            height=400
        )

        # Display the chart using Streamlit
        st.altair_chart(chart, use_container_width=True)

        

    