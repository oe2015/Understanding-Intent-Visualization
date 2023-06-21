import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import json
import torch
from architecture import FramingModel, CustomModel, update_and_load_model, PersuasionModel
from config import CFG
from torch.utils.data import  DataLoader
from newspaper import Article
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
    page_title="News Framing Demo",
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
        """
    )
    st.markdown("")
st.markdown("")
st.markdown("## Demo! ")
url = ""
cx, cy, cz = st.columns([5, 2, 5])
doc = ""
option = st.radio("Choose an option to get framings", ("Pass URL", "Enter Text"))
if option == "Pass URL":
        with st.form(key="url_form"):
            url = st.text_input("Enter the article URL to get framings")
            print(url)
            submit_button = st.form_submit_button(label="Get news framing!")
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
        text = st.text_area("Enter the text to get framings")
        submit_button = st.form_submit_button(label="Get news framing!")
        if submit_button:
            doc = text
            with st.expander("Read the article", expanded=False):
                st.write(doc)

text = " ".join(doc.split())


import numpy as np
import re
import math
max_tokens = 512
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
tokens = tokenizer.encode(text, return_tensors='pt')
truncated_tensor = tokens[:,:510]
decoded_string = tokenizer.decode(truncated_tensor[0], skip_special_tokens=True)
# print(len(decoded_string))

# text = text[:2200]
# print(text)

if doc:
    ############# for subtask1 #####################
    API_TOKEN = "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"

    def make_request(input_text):
        API_URL = "https://api-inference.huggingface.co/models/oe2015/XLMRobertaNews"
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
  
        data = {
        "inputs": input_text
    }
        response = requests.post(API_URL, headers=headers, json=data)
        if 'error' in response.json() and 'estimated_time' in response.json():
            wait_time = response.json()['estimated_time']
            print(f"Model loading, waiting for {wait_time} seconds.")
            time.sleep(wait_time)
            response = requests.post(API_URL, headers=headers, json=data)

        return response.json()
    
    response = make_request(decoded_string)
    print(response)
###########################################################


############### for subtask 2 #############################
    API_TOKEN = "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"

    def make_request2(input_text):
        API_URL = "https://api-inference.huggingface.co/models/oe2015/XLMsubtask2"
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
  
        data = {
        "inputs": input_text
    }
        response = requests.post(API_URL, headers=headers, json=data)
        if 'error' in response.json() and 'estimated_time' in response.json():
            wait_time = response.json()['estimated_time']
            print(f"Model loading, waiting for {wait_time} seconds.")
            time.sleep(wait_time)
            response = requests.post(API_URL, headers=headers, json=data)

        return response.json()
    
    response2 = make_request2(decoded_string)
    print(response2)
    
##########################################################################

############### subtask 3 ##########################################
    # API_TOKEN = "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"

    # def make_request3(input_text):
    #     API_URL = "https://api-inference.huggingface.co/models/DianaTurmakhan/XLMRobertaPersuasion"
    #     headers = {"Authorization": f"Bearer {API_TOKEN}"}
  
    #     data = {
    #     "inputs": input_text
    # }
    #     response = requests.post(API_URL, headers=headers, json=data)
    #     if 'error' in response.json() and 'estimated_time' in response.json():
    #         wait_time = response.json()['estimated_time']
    #         print(f"Model loading, waiting for {wait_time} seconds.")
    #         time.sleep(wait_time)
    #         response = requests.post(API_URL, headers=headers, json=data)

    #     return response.json()

    import pandas as pd
    import matplotlib.pyplot as plt
    import altair as alt
    import streamlit as st

    # Convert responses to pandas dataframes
    df1 = pd.DataFrame(response[0])
    df2 = pd.DataFrame(response2[0])

    # Define the actual label names
    label_names1 = {
        'LABEL_0': 'reporting',
        'LABEL_1': 'opinion',
        'LABEL_2': 'satire'
    }

    label_names2 = {
        'LABEL_0': 'Economic',
        'LABEL_1': 'Capacity and resources',
        'LABEL_2': 'Morality',
        'LABEL_3': 'Fairness and equality',
        'LABEL_4': 'Legality Constitutionality and jurisprudence',
        'LABEL_5': 'Policy prescription and evaluation',
        'LABEL_6': 'Crime and punishment',
        'LABEL_7': 'Security and defense',
        'LABEL_8': 'Health and safety',
        'LABEL_9': 'Quality of life',
        'LABEL_10': 'Cultural identity',
        'LABEL_11': 'Public opinion',
        'LABEL_12': 'Political',
        'LABEL_13': 'External regulation and reputation',
    }

    # Map labels to actual names
    df1['label'] = df1['label'].map(label_names1)
    df2['label'] = df2['label'].map(label_names2)

    # Sort dataframes by 'score'
    df1.sort_values(by='score', ascending=False, inplace=True)
    df2.sort_values(by='score', ascending=False, inplace=True)

    # SUBTASK 1 VISUALIZATION
    with st.expander(f"### Get Genre for This Article", expanded=False):
        st.markdown("#### Output Probabilities Pie Chart")
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(df1['score'], labels=df1['label'], autopct="%1.1f%%", pctdistance=0.85)
        ax.axis("equal")
        ax.legend(wedges, df1['label'],
            title="Labels",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1))
        st.pyplot(fig)

    frames_colors = {
        'Economic': '#6941BF',
        'External regulation and reputation': '#0468BF',
        'Capacity and resources': '#8FCEFF',
        'Political': '#FFD16A',
        'Security and defense': '#F22E2E',
        'Quality of life': '#80F29D',
        'Policy prescription and evaluation': '#FFABAB',
        'Legality Constitutionality and jurisprudence': '#9f7fe3',
        'Cultural identity': '#b1cce3',
        'Fairness and equality': '#b39696',
        'Crime and punishment': '#2f8dd6',
        'Health and safety': '#bd7373',
        'Public opinion': '#d4cdcd',
        'Morality': '#5c9c6c'
    }

    with st.expander(f"### Get Framings for this Article", expanded=False):
        chart = alt.Chart(df2).mark_bar().encode(
            x = 'score',
            y = 'label',
            color=alt.Color('label', scale=alt.Scale(domain=list(frames_colors.keys()), range=list(frames_colors.values()))),
            tooltip=['score', 'label']
        ).properties(
            width=1000,
            height=600
        )
        # Display the chart using Streamlit
        st.altair_chart(chart, use_container_width=True)  


    # output_map = {}
    # sentences = extract_title_and_sentences(decoded_string)
    # for sentence in sentences:
    #     response = make_request3(sentence)
    #     print(response)
    #     filtered_labels = []
    #     filtered_outputs = []
    #     for item in response[0]:
    #         lab = item['label']
    #         index = int(lab.split('_')[1])
    #         score = item['score']
    #         if labels_list3[index] == "None":
    #             continue
    #         if score >= 0.05:
    #             filtered_labels.append(labels_list3[index])
    #             filtered_outputs.append(score)
    #     output_map[sentence] = {'labels': filtered_labels, 'outputs': filtered_outputs}
    # print(output_map)


##############################################################################################################
    API_TOKEN = "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"

    def make_request3(sentences):  # Modify the function to take a list of sentences
        API_URL = "https://api-inference.huggingface.co/models/oe2015/XLMPersuasion"
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        payload = {"inputs": sentences}  # Inputs is now a list of sentences
        response = requests.request("POST", API_URL, headers=headers, data=json.dumps(payload))
        return json.loads(response.content.decode("utf-8"))

    output_map = {}
    sentences = extract_title_and_sentences(text)
    print(sentences)
    response = make_request3(sentences)  # Call the API once with all the sentences
    print(response)
    for i in range(len(sentences)):  # Loop over each sentence
        sentence = sentences[i]
        response_for_sentence = response[i]  # Get the corresponding response for the sentence
        filtered_labels = []
        filtered_outputs = []
        for item in response_for_sentence:
            lab = item['label']
            index = int(lab.split('_')[1])
            score = item['score']
            if labels_list3[index] == "None":
                continue
            if score >= 0.05:
                filtered_labels.append(labels_list3[index])
                filtered_outputs.append(score)
        output_map[sentence] = {'labels': filtered_labels, 'outputs': filtered_outputs}

            ###################################################################################

    # SUBTASK 3 Visualization
    label_colors = {}  # Dictionary to store assigned colors for each label
    with st.expander(f"### Get persuasion techniques for this Article", expanded=False):
        for sentence, entry in output_map.items():
            labels = entry['labels']
            outputs = entry['outputs']
            # Create the annotated text
            annotations = [(sentence, "", "#fff", "#000")]
            sorted_outputs = sorted(outputs, reverse=True)
            for label, output in zip(labels, sorted_outputs):
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
