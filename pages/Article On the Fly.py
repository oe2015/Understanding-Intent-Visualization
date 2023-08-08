import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import json
import torch
from architecture import FramingModel, CustomModel, update_and_load_model, PersuasionModel
from config import CFG
from torch.utils.data import  DataLoader
# from newspaper import Article
import matplotlib.pyplot as plt
from transformers import XLMRobertaTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModel
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from annotated_text import annotated_text
import time
import requests
import random
from bs4 import BeautifulSoup 


################################################################################
# import os
# from transformers import AutoConfig

# def upload_model(model_path, repo_name):
#     xlmrobertaconfig = AutoConfig.from_pretrained("xlm-roberta-base")
#     framing_model = FramingModel(xlmrobertaconfig, 14)
#     ckpt = update_and_load_model(model_path)
#     framing_model.load_state_dict(ckpt, strict=False)
#     torch.save(framing_model.state_dict(), os.path.join(repo_name, 'pytorch_model.bin'))
#     # Create the correct configuration for your model
#     cfg = AutoConfig.from_pretrained("xlm-roberta-base", num_labels = 14)
#     cfg.save_pretrained(repo_name)

# # Call the function with model_path, the desired model_name, and the repo name
# upload_model("subtask2.pt", "XLMRobertaFraming")

################################################################################

def get_random_color():
    letters = "ABCDEF"
    color = "#"

    for _ in range(6):
        component = random.randint(0, len(letters) - 1)
        color += letters[component]

    return color



def extract_title_and_sentences(text):
    # Split the text into sentences
    sentences = sent_tokenize(text)
    
    # Extract the title from the first sentence
    title = sentences[0]
    
    # Remove the title from the list of sentences
    sentences = sentences[1:]
    
    return sentences


flag = 0
st.set_page_config(
    page_title="FRAPPE",
    page_icon=":газета:",
    layout="wide"
)

labels_list = [
    "Economic",
    "Capacity_and_resources",
    "Morality",
    "Fairness_and_equality",
    "Legality_Constitutionality_and_jurisprudence",
    "Policy_prescription_and_evaluation",
    "Crime_and_punishment",
    "Security_and_defense",
    "Health_and_safety",
    "Quality_of_life",
    "Cultural_identity",
    "Public_opinion",
    "Political",
    "External_regulation_and_reputation",
]

labels_list3 = [
    "Appeal_to_Authority",
    "Appeal_to_Popularity",
    "Appeal_to_Values",
    "Appeal_to_Fear-Prejudice",
    "Flag_Waving",
    "Causal_Oversimplification",
    "False_Dilemma-No_Choice",
    "Consequential_Oversimplification",
    "Straw_Man",
    "Red_Herring",
    "Whataboutism",
    "Slogans",
    "Appeal_to_Time",
    "Conversation_Killer",
    "Loaded_Language",
    "Repetition",
    "Exaggeration-Minimisation",
    "Obfuscation-Vagueness-Confusion",
    "Name_Calling-Labeling",
    "Doubt",
    "Guilt_by_Association",
    "Appeal_to_Hypocrisy",
    "Questioning_the_Reputation",
    "None",
]

def _max_width_():
    max_width_str = f"max-width: 5000px;"
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
    st.title("FRAPPE")
    st.header("")
with st.expander("About this app", expanded=False):
    st.write(
        """
        This app performs three major tasks on English news articles:
        
        **1. Article Type Classification:** 
        The model classifies an article into one of the three types:
        - Opinion
        - Reporting
        - Satire
        
        **2. Frame Detection:**
        The model is capable of detecting the existence of one or multiple frames out of 14 possible frames:
        - Economic
        - Capacity and Resources
        - Morality
        - Fairness and Equality
        - Legality, Constitutionality, and Jurisprudence
        - Policy Prescription and Evaluation
        - Crime and Punishment
        - Security and Defense
        - Health and Safety
        - Quality of Life
        - Cultural Identity
        - Public Opinion
        - Political
        - External Regulation and Reputation
        
        **3. Persuasion Technique Classification:**
        The model identifies the usage of various persuasion techniques at the sentence level, including but not limited to:
        - Loaded Language
        - Obfuscation, Vagueness, Confusion
        - Conversation Killer
        - Appeal to Time
        - Whataboutism
        - Red Herring
        - Straw Man
        - Causal Oversimplification
        - Appeal to Values
        - Appeal to Popularity
        - Appeal to Hypocrisy
        - Appeal to Authority
        - Consequential Oversimplification
        - False Dilemma/No Choice
        - Repetition
        - Slogans
        - Doubt
        - Exaggeration/Minimisation
        - Name Calling/Labeling
        - Flag Waving
        - Appeal to Fear/Prejudice
        - Guilt by Association
        - Questioning the Reputation
        """
    )
    st.markdown("")
st.markdown("")
st.markdown("## Demo! ")
url = ""
cx, cy, cz = st.columns([5, 2, 5])
doc = ""
option = st.radio("Choose an option to analyze an article", ("Pass URL", "Enter Text"))
if option == "Pass URL":
        with st.form(key="url_form"):
            url = st.text_input("Enter the article URL to get analysis")
            print(url)
            submit_button = st.form_submit_button(label="Get article analysis!")
            if submit_button:
                if len(url) > 0:
                    try:
                        response = requests.get(url)
                        response.raise_for_status()  # Raise an exception for any HTTP error status
                        soup = BeautifulSoup(response.text, 'html.parser')
                        title = soup.title.text
                        paragraphs = soup.find_all('p')
                        par = ''
                        for p in paragraphs:
                            par += "\n" + p.text
                        doc = title + "\n" + par
                        st.warning("Article successfully processed.")
                        flag = 1
                        with st.expander("Read the article", expanded=False):
                            st.write(doc)
                    except requests.exceptions.RequestException as e:
                        st.warning("Access denied.")
                        exit()
                    except Exception as e:
                        print("An error occurred during text extraction:", e)
                else:
                    print("URL is empty.")
elif option == "Enter Text":
    with st.form(key="text_form"):
        text = st.text_area("Enter the text to get article analysis")
        submit_button = st.form_submit_button(label="Get article analysis!")
        if submit_button:
            doc = text
            with st.expander("Read the article", expanded=False):
                st.write(doc)

text = " ".join(doc.split())
if text:
    # predicted_probabilities = torch.softmax(outputs, dim=1).squeeze().tolist()
    response = requests.post("https://62ef-196-155-78-194.ngrok-free.app/task1", json={"text": text})
    # Get the prediction from the response
    predicted_probabilities = response.json()
    print(predicted_probabilities)
    class_names = CFG.CLASSES
    max_probability_index = np.argmax(predicted_probabilities)
    max_probability_class = class_names[max_probability_index]
    max_probability = predicted_probabilities[max_probability_index]
    for class_name, probability in zip(class_names, predicted_probabilities):
        print(f"{class_name}: {probability:.4f}")
    # SUBTASK 1 VISUALIZATION
    with st.expander(f"### Get Genre for this Article", expanded=False):
        # st.markdown(f"### This article is classified as: {max_probability_class}")
        st.markdown("#### Output Probabilities Pie Chart")
        fig, ax = plt.subplots()
        ax.pie(predicted_probabilities, labels=class_names, autopct="%1.1f%%")
        ax.axis("equal")
        st.pyplot(fig)

    # SUBTASK 2 VISUALIZATION
    response = requests.post("https://62ef-196-155-78-194.ngrok-free.app/task2", json={"text": text})
    # Get the prediction from the response
    output = response.json()
    data = output
    df = pd.DataFrame(data)
    sorted_df = df.sort_values(by='Probability', ascending=False)
    sorted_df = sorted_df.reset_index(drop=True)

    with st.expander(f"### Get analysis for this Article", expanded=False):
        x = sorted_df["Probability"]
        y = sorted_df["Label"]
        chart = alt.Chart(sorted_df).mark_bar().encode(
            x = 'Probability',
            y = 'Label',
            color=alt.Color('Label', scale=alt.Scale(scheme='category10')),
            tooltip=['Probability', 'Label']
        ).properties(
            width=1000,
            height=600
        )
        # Display the chart using Streamlit
        st.altair_chart(chart, use_container_width=True)

    
    import streamlit as st
    import requests
    import json
    
    # Send the text to the API
    response = requests.post("https://62ef-196-155-78-194.ngrok-free.app/task3", json={"text": text})
    # Get the prediction from the response
    prediction = response.json()
    # Display the prediction
    # st.write(prediction)
    
    # annotated_text(
    #     ("Team Name:", "", "#000", "#fff"),
    #     (
    #         "NLP2",
    #         "The Best Team",
    #         "#8ef",
    #     ),
    # )
    
    #SUBTASK 3 Visualization
    label_colors = {}  # Dictionary to store assigned colors for each label
    
    with st.expander(f"### Get Persuasion Techniques for this Article", expanded=False):
        for sentence, entry in prediction.items():
                labels = entry['labels']
                outputs = entry['outputs']
    
                # Create the annotated text
                annotations = [(sentence, "", "#fff", "#000")]
                sorted_outputs = sorted(outputs, reverse=True)
    
                for label, output in zip(labels, sorted_outputs):
                    # output =  output.item()
                    if label == "None":
                        continue
    
                    if label not in label_colors:
                # Generate a random color for a new label
                        label_colors[label] = get_random_color()
    
                    color1 = label_colors[label]
                    annotation_text = f"{label.replace('_', ' ')}  {round(output * 100, 2)}%"
                    annotations.append((annotation_text, "", color1))
    
                # Display the annotated text using annotated_text
                annotated_text(*annotations)
                st.write("\n\n")
