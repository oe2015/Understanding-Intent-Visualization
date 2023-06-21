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
import os
from transformers import AutoConfig

# def upload_model(model_path, repo_name, base=True, subtask=3):
#     config = "xlm-roberta-base" if base else "xlm-roberta-large"
#     xlmrobertaconfig = AutoConfig.from_pretrained(config)

#     if subtask == 1:
#         model_cls = CustomModel
#     elif subtask == 2:
#         model_cls = FramingModel
#     else:
#         model_cls = PersuasionModel

#     model = model_cls(config)
    
#     ckpt = update_and_load_model(model_path)
#     model_cls.load_state_dict(ckpt, strict=False)
#     torch.save(model_cls.state_dict(), os.path.join(repo_name, 'pytorch_model.bin'))
#     # Create the correct configuration for your model

# # Call the function with model_path, the desired model_name, and the repo name
# upload_model("subtask2.pt", "XLMRobertaFraming")

################################################################################
# def update_and_load_model(ckpt_path, device="cpu"):
#     ckpt = torch.load(ckpt_path, map_location=device)
#     if "state_dict" in ckpt:
#         ckpt = ckpt["state_dict"]

#     for k in list(ckpt.keys()):
#         if k.startswith("head2."):
#             # rename the key to classifier
#             ckpt[k.replace("head2.", "classifier.")] = ckpt.pop(k)
#         if k.startswith("model."):
#             # rename the key to classifier
#             ckpt[k.replace("model.", "")] = ckpt.pop(k)

#     return ckpt

# @st.cache_resource
# def load_model():
#     tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
#     # Load fine-tuned model checkpoint
#     model_path = "best_model-v1.ckpt"
#     # model_state_dict = torch.load(model_path)["state_dict"]
#     ckpt = update_and_load_model(model_path, device="cpu")
#     xlmrobertaconfig = AutoConfig.from_pretrained("xlm-roberta-base")
#     custom_xlm_roberta = CustomModel(xlmrobertaconfig)
#     custom_xlm_roberta.load_state_dict(ckpt, strict=False)
#     custom_xlm_roberta.eval()
#     return tokenizer, custom_xlm_roberta

# def upload_model(repo_name):
#     model = load_model()[1]
#     model.save_pretrained(repo_name)

# upload_model("XLMRobert_task1")
# print("Done")
# exit(1)


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

@st.cache_resource()
def load_model1():
    model = CustomModel.from_pretrained("oe2015/XLMRobertaNews")
    return model

@st.cache_resource()
def load_model2():
    model = FramingModel.from_pretrained("oe2015/XLMsubtask2")
    return model

@st.cache_resource()
def load_model3():
    model = PersuasionModel.from_pretrained("oe2015/XLMPersuasion")
    return model

import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import streamlit as st

import numpy as np
import re
import math
max_tokens = 512
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
tokens = tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding="longest",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
)
####################################################
model1 = load_model1()
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']
predicted_probabilities = torch.softmax(model1(input_ids, attention_mask), dim=1).squeeze().tolist()
CLASSES = ["reporting", "opinion", "satire"]
class_names = CLASSES 
max_probability_index = np.argmax(predicted_probabilities)
max_probability_class = class_names[max_probability_index]
max_probability = predicted_probabilities[max_probability_index]
# st.write(f"{max_probability_class}: {max_probability:.4f} (highest probability)")
for class_name, probability in zip(class_names, predicted_probabilities):
    print(f"{class_name}: {probability:.4f}")
with st.expander(f"### This article is classified as: {max_probability_class}", expanded=False):
    # st.markdown(f"### This article is classified as: {max_probability_class}")
    st.markdown("#### Output Probabilities Pie Chart")
    fig, ax = plt.subplots()
    ax.pie(predicted_probabilities, labels=class_names, autopct="%1.1f%%")
    ax.axis("equal")
    st.pyplot(fig)

###########################################


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

labels_list = [
    "Economic",
    "Capacity and resources",
    "Morality",
    "Fairness and equality",
    "Legality Constitutionality and jurisprudence",
    "Policy prescription and evaluation",
    "Crime and punishment",
    "Security and defense",
    "Health and safety",
    "Quality of life",
    "Cultural identity",
    "Public opinion",
    "Political",
    "External regulation and reputation",
]

model2 = load_model2()
probabilities = torch.sigmoid(model2(text))
print(model2(text))
print(probabilities)
data = {
    'Label': labels_list,
    'Probability': probabilities.flatten().tolist()
}
df = pd.DataFrame(data)
sorted_df = df.sort_values(by='Probability', ascending=False)
sorted_df = sorted_df.reset_index(drop=True)
print(sorted_df)

with st.expander(f"### Get Framings for this Article", expanded=False):
    x = sorted_df["Probability"]
    y = sorted_df["Label"]
    
    # Define the custom color scale
    color_scale = alt.Scale(domain=list(frames_colors.keys()), range=list(frames_colors.values()))
    
    chart = alt.Chart(sorted_df).mark_bar().encode(
        x='Probability',
        y='Label',
        # Use the custom color scale
        color=alt.Color('Label', scale=color_scale),
        tooltip=['Probability', 'Label']
    ).properties(
        width=1000,
        height=600
    )
    # Display the chart using Streamlit
    st.altair_chart(chart, use_container_width=True)

# ##############################################################################################################

# def make_request3(sentences):  # Modify the function to take a list of sentences
#     API_URL = "https://api-inference.huggingface.co/models/DianaTurmakhan/XLMRobertaPersuasion"
#     headers = {"Authorization": f"Bearer {API_TOKEN}"}
#     payload = {"inputs": sentences}  # Inputs is now a list of sentences
#     response = requests.request("POST", API_URL, headers=headers, data=json.dumps(payload))
#     return json.loads(response.content.decode("utf-8"))

from transformers import XLMRobertaTokenizer

def process_batch(batch_tokens, model, threshold=0.05):
    # Pad the sequences to make them of equal length
    max_length = max(len(tokens) for tokens in batch_tokens)
    batch_tokens = [tokens + [tokenizer.pad_token_id]*(max_length - len(tokens)) for tokens in batch_tokens]

    # Convert list of lists into tensors
    batch_tokens_tensor = torch.tensor(batch_tokens)
    batch_attention_mask = torch.tensor([[1 if token != tokenizer.pad_token_id else 0 for token in tokens] for tokens in batch_tokens])

    # Pass the tokens to the model

    logits = model(content_input_ids=batch_tokens_tensor, content_attention_mask=batch_attention_mask)

    # Apply the sigmoid function to convert logits to probabilities
    probabilities = torch.sigmoid(logits).tolist()

    output_map = {}
    for i, sentence in enumerate(batch_sentences):  # Loop over each sentence
        response_for_sentence = probabilities[i]  # Get the corresponding response for the sentence
        filtered_labels = []
        filtered_outputs = []
        for j, score in enumerate(response_for_sentence):
            if labels_list3[j] == "None":
                continue
            if score >= threshold:
                filtered_labels.append(labels_list3[j])
                filtered_outputs.append(score)
        output_map[sentence] = {'labels': filtered_labels, 'outputs': filtered_outputs}
    return output_map



output_map = {}
sentences = extract_title_and_sentences(text)

# Load the model and tokenizer
model3 = load_model3()
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")

# Tokenize the sentences and keep track of total tokens
tokens = []
total_tokens = 0
batch_sentences = []

for sentence in sentences:
    sentence_tokens = tokenizer.encode(sentence, add_special_tokens=True)
    total_tokens += len(sentence_tokens)

    if total_tokens > 512:  # If total tokens exceeds 512, process the tokens collected so far and reset
        output_map.update(process_batch(tokens, model3))  # Your function to process a batch
        tokens = [sentence_tokens]
        total_tokens = len(sentence_tokens)
        batch_sentences = [sentence]
    else:
        tokens.append(sentence_tokens)
        batch_sentences.append(sentence)

# Process remaining tokens if any
if tokens:
    output_map.update(process_batch(tokens, model3))

print(output_map)
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

                

