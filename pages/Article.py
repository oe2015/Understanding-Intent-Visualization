import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import json
import torch
from architecture import FramingModel, CustomModel, update_and_load_model,PersuasionModel
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

import random

# def get_random_color():
#     letters = "CDEF"
#     color = "#"
#     for i in range(6):
#         color += letters[random.randint(0, len(letters) - 1)]
#     return color



def get_random_color():
    letters = "ABCDEF"
    color = "#"

    for _ in range(6):
        component = random.randint(0, len(letters) - 1)
        color += letters[component]

    return color

# from download_button import download_button

#################################################################################

# import os
# from transformers import AutoConfig
# def upload_model(model_path, repo_name):
    
#     xlmrobertaconfig = AutoConfig.from_pretrained("xlm-roberta-large")
#     persuasion_model = PersuasionModel(xlmrobertaconfig, 24)
#     ckpt = update_and_load_model("./subtask3.pth")
#     persuasion_model.load_state_dict(ckpt, strict=False)
#     torch.save(persuasion_model.state_dict(), os.path.join(repo_name, 'pytorch_model.bin'))

    
#     # Create the correct configuration for your model
#     cfg = AutoConfig.from_pretrained("xlm-roberta-large", num_labels = 24)
#     cfg.save_pretrained(repo_name)


# # Call the function with model_path, the desired model_name, and the repo name
# upload_model("subtask3.pth", "XLMRobertaPersuasion")
# ################################################################################

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
)
@st.cache_resource
def load_model():
    # Specify your Hugging Face model hub repository
    model_name = "oe2015/XLMRobertaNews"
    # Load the tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    # Download the model weights
    model_file = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
    model = CustomModel(CFG) # Initialize your model with its configuration
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    # Set the model in evaluation mode
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_model_task2():
    model_name = "LaraHassan/XLMRobertaFrames"
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    model_file = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
    xlmrobertaconfig = AutoConfig.from_pretrained("xlm-roberta-base")
    model = FramingModel(xlmrobertaconfig, 14)
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    # Set the model in evaluation mode
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_model_task3():
    model_name = "DianaTurmakhan/XLMRobertaPersuasion"
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
    model_file = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
    xlmrobertaconfig = AutoConfig.from_pretrained("xlm-roberta-large")
    model = PersuasionModel(xlmrobertaconfig, 24)
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    # Set the model in evaluation mode
    model.eval()
    return tokenizer, model

tokenizer2, model2 = load_model_task2()
# tokenizer3, model3 = load_model_task3()
tokenizer, custom_xlm_roberta = load_model()
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
        submit_button = st.form_submit_button(label="Get news framing!")
        if submit_button:
            if len(url) > 0:
                article = Article(url)
                try:
                    article.download()
                    article.parse()
                    # print(article.text)
                    doc = article.text
                    st.warning("Article successfully processed.")
                    flag = 1
                    with st.expander("Read the article", expanded=False):
                        st.write(doc)
                except:
                    st.warning("Access denied.")
                    exit()
elif option == "Enter Text":
    with st.form(key="text_form"):
        text = st.text_area("Enter the text to get framings")
        submit_button = st.form_submit_button(label="Get news framing!")
        if submit_button:
            doc = text
            with st.expander("Read the article", expanded=False):
                st.write(doc)
def split_title_content(text):
    # Split the text by the first new line character
    parts = text.split('\n', 1)
    # Check if there is a new line character in the text
    if len(parts) > 1:
        title = parts[0]  # First part is the title
        content = parts[1]  # Second part is the content
    else:
        title = parts[0]  # Entire text is considered as the title
        content = ''
    # Split the title by the first period (.) to get the first sentence
    title_parts = title.split('.', 1)
    # Check if there is a period in the title
    if len(title_parts) > 1:
        first_sentence = title_parts[0] + '.'  # Add the period back to the first sentence
        title = title_parts[1]  # Remaining part becomes the new title
    return title.strip(), content.strip()
# Tokenize input text
text = " ".join(doc.split())
inputs = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    max_length=512,
    padding="longest",
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt",
)
title, content = split_title_content(text)
title_tok = tokenizer2.encode_plus(
    title,
    add_special_tokens=True,
    max_length=512,
    padding="longest",
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt",
)
content_tok = tokenizer2.encode_plus(
    content,
    add_special_tokens=True,
    max_length=512,
    padding="longest",
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt",
)

if doc:
    outputs = custom_xlm_roberta(inputs)
    inputs = {
        'title_input_ids' : title_tok['input_ids'],
        'title_attention_mask' :  title_tok['attention_mask'],
        'content_input_ids' : content_tok['input_ids'],
        'content_attention_mask' :  content_tok['attention_mask']
    }
    outputs_2 = model2(**inputs)
    probabilities = torch.sigmoid(outputs_2)
    data = {
        'Label': labels_list,
        'Probability': probabilities.flatten().tolist()
    }
    df = pd.DataFrame(data)
    sorted_df = df.sort_values(by='Probability', ascending=False)
    sorted_df = sorted_df.reset_index(drop=True)

    predicted_probabilities = torch.softmax(outputs, dim=1).squeeze().tolist()
    class_names = CFG.CLASSES
    max_probability_index = np.argmax(predicted_probabilities)
    max_probability_class = class_names[max_probability_index]
    max_probability = predicted_probabilities[max_probability_index]
    for class_name, probability in zip(class_names, predicted_probabilities):
        print(f"{class_name}: {probability:.4f}")
    # SUBTASK 1 VISUALIZATION
    with st.expander(f"### Get Genre for This Article", expanded=False):
        # st.markdown(f"### This article is classified as: {max_probability_class}")
        st.markdown("#### Output Probabilities Pie Chart")
        fig, ax = plt.subplots()
        ax.pie(predicted_probabilities, labels=class_names, autopct="%1.1f%%")
        ax.axis("equal")
        st.pyplot(fig)
    # SUBTASK 2 VISUALIZATION
    with st.expander(f"### Get Framings for this Article", expanded=False):
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


    # # SUBTASK 3 Model Loading
    # output_map = {}
    # sentences = extract_title_and_sentences(text)
   
    # for sentence in sentences:
    #     # Tokenize the sentence
    #     inputs = tokenizer3.encode_plus(
    #         sentence,
    #         add_special_tokens=True,
    #         max_length=512,
    #         padding="longest",
    #         truncation=True,
    #         return_attention_mask=True,
    #         return_tensors="pt",
    #     )

    #     inputs_3 = {
    #     'content_input_ids' : inputs['input_ids'],
    #     'content_attention_mask' :  inputs['attention_mask']
    #     }
        
    #     # Pass the tokenized sentence to the model
    #     outputs_3 = model3(**inputs_3)
    #     probabilities_3 = torch.sigmoid(outputs_3)
        
    #     filtered_labels = []
    #     filtered_outputs = []
    #     for index, output in enumerate(probabilities_3[0]):
    #         if labels_list3[index] == "None":
    #             continue
    #         if output >= 0.09:
    #             filtered_labels.append(labels_list3[index])
    #             filtered_outputs.append(output)
    #     # Store the filtered labels and outputs in the map with the corresponding sentence
    #     output_map[sentence] = {'labels': filtered_labels, 'outputs': filtered_outputs}
    
    # print(output_map)

    
    # annotated_text(
    #     ("Team Name:", "", "#000", "#fff"),
    #     (
    #         "The Syllogist",
    #         "The Best One",
    #         "#8ef",
    #     ),
    # )
    
    # #SUBTASK 3 Visualization
    # label_colors = {}  # Dictionary to store assigned colors for each label

    # with st.expander(f"### Get persuasion techniques for this Article", expanded=False):
    #     for sentence, entry in output_map.items():
    #             labels = entry['labels']
    #             outputs = entry['outputs']

    #             # Create the annotated text
    #             annotations = [(sentence, "", "#fff", "#000")]
    #             sorted_outputs = sorted(outputs, reverse=True)

    #             for label, output in zip(labels, sorted_outputs):
    #                 output =  output.item()

    #                 if label == "None":
    #                     continue

    #                 if label not in label_colors:
    #             # Generate a random color for a new label
    #                     label_colors[label] = get_random_color()

    #                 color1 = label_colors[label]
    #                 annotation_text = f"{label.replace('_', ' ')}  {round(output * 100, 2)}%"
    #                 annotations.append((annotation_text, "", color1))

    #             # Display the annotated text using annotated_text
    #             annotated_text(*annotations)
    #             st.write("\n\n")

                

