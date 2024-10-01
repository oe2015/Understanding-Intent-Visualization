import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import json
import torch
# from architecture import FramingModel, CustomModel, update_and_load_model, PersuasionModel
# from config import CFG
from torch.utils.data import  DataLoader
# from newspaper import Article
import matplotlib.pyplot as plt
# from transformers import XLMRobertaTokenizer, AutoModel
from huggingface_hub import hf_hub_download
# from transformers import AutoConfig, AutoModel
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from annotated_text import annotated_text
import time
import requests
import random
from bs4 import BeautifulSoup 
from time import sleep

print("hello")


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
# with st.expander("About this platform!", expanded=False):
#     st.write(
#         """
#         The abundance of news sources and the urgent demand for reliable information have led to serious concerns about the threat of misleading information. We present FRAPPE, a FRAming, Persuasion, and Propaganda Explorer system. FRAPPE goes beyond conventional news analysis of articles and unveils the intricate linguistic techniques used to shape readers’ opinions and emotions. Our system allows users not only to analyze individual articles for their genre, framings, and use of persuasion techniques, but also to draw comparisons between the strategies of persuasion and framing adopted by a diverse pool of news outlets and countries across multiple languages for different topics, thus providing a comprehensive understanding of how information is presented and manipulated.
#         You can find the paper here: https://aclanthology.org/2024.eacl-demo.22/ and a video of our demo here: https://aclanthology.org/2024.eacl-demo.22.mp4
#         """
#     )
#     st.markdown("")
framing_explanations = {
"Economic": "Costs, benefits, or other financial implications",
"Capacity and resources": "Availability of physical, human or financial resources, and capacity of current systems",
"Morality": "Religious or ethical implications",
"Fairness and equality": "Balance or distribution of rights, responsibilities, and resources",
"Legality Constitutionality and jurisprudence": "Rights, freedoms, and authority of individuals, corporations, and government",
"Policy prescription and evaluation": "Discussion of specific policies aimed at addressing problems",
"Crime and punishment": "Effectiveness and implications of laws and their enforcement",
"Security and defense": "Threats to welfare of the individual, community, or nation",
"Health and safety": "Health care, sanitation, public safety",
"Quality of life": "Threats and opportunities for the individual's wealth, happiness, and well-being",
"Cultural identity": "Traditions, customs, or values of a social group in relation to a policy issue",
"Public opinion": "Attitudes and opinions of the general public, including polling and demographics",
"Political": "Considerations related to politics and politicians, including lobbying, elections, and attempts to sway voters",
"External regulation and reputation": "International reputation or foreign policy"
}

techniques = {
    "Name Calling or Labelling": "Using insulting or desirable labels for individuals or groups.",
    "Guilt by Association": "Attacking by linking to negatively viewed groups or concepts.",
    "Casting Doubt": "Undermining credibility by questioning character.",
    "Appeal to Hypocrisy": "Accusing of hypocrisy to attack reputation.",
    "Questioning the Reputation": "Undermining character with negative claims.",
    "Flag Waiving": "Justifying ideas by appealing to group pride or benefits.",
    "Appeal to Authority": "Citing authority to support an argument.",
    "Appeal to Popularity": "Claiming widespread agreement to justify a stance.",
    "Appeal to Values": "Linking ideas to positive values.",
    "Appeal to Fear, Prejudice": "Using fear or prejudice to promote or reject ideas.",
    "Strawman": "Misrepresenting an argument to refute it easily.",
    "Red Herring": "Distracting from the main issue with irrelevant topics.",
    "Whataboutism": "Accusing of hypocrisy without disproving the argument.",
    "Causal Oversimplification": "Oversimplifying causes of an issue.",
    "False Dilemma or No Choice": "Presenting only two options when more exist.",
    "Consequential Oversimplification": "Claiming improbable chain reactions.",
    "Slogans": "Using catchy phrases with emotional appeal.",
    "Conversation Killer": "Discouraging discussion with dismissive phrases.",
    "Appeal to Time": "Arguing that it's the right time for action.",
    "Loaded Language": "Using emotionally charged words to influence.",
    "Obfuscation, Intentional Vagueness, Confusion": "Being unclear to allow varied interpretations.",
    "Exaggeration or Minimisation": "Overstating or downplaying significance.",
    "Repetition": "Repeating phrases to persuade."
}

# (Keep your framing_explanations and techniques dictionaries as they are)

with st.expander("About FRAPPE", expanded=False):
    st.markdown("""
    ## Welcome to FRAPPE: Your FRAming, Persuasion, and Propaganda Explorer

    In today's information-rich world, distinguishing reliable news from misleading content is more crucial than ever. FRAPPE is here to help you navigate this complex landscape.

    ### What FRAPPE Does:
    - Analyzes news articles beyond the surface level
    - Uncovers linguistic techniques that shape opinions and emotions
    - Analyzes articles for specific frames and persuasion techniques, including:
    """)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Frames:")
        for frame in framing_explanations.keys():
            st.markdown(f"- {frame}", help=framing_explanations[frame])

    with col2:
        st.markdown("#### Persuasion Techniques:")
        for technique in techniques.keys():
            st.markdown(f"- {technique}", help=techniques[technique])

    st.markdown("""
    - Compares persuasion and framing strategies across:
      - Various news outlets
      - Multiple countries
      - Different languages and topics

    ### Why Use FRAPPE:
    - Gain deeper insights into news content
    - Understand how information is presented and potentially manipulated
    - Develop a more critical and informed approach to media consumption

    ### Learn More:
    - Read our research paper: [FRAPPE: FRAming, Persuasion, and Propaganda Explorer](https://aclanthology.org/2024.eacl-demo.22/)
    - Watch our demo video: [FRAPPE in Action](https://aclanthology.org/2024.eacl-demo.22.mp4)

    Empower yourself with FRAPPE and become a more discerning news reader!
    """)

st.markdown("")
st.markdown("## 🚀 On-The-Fly Analysis!")

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
# if text:
    # # predicted_probabilities = torch.softmax(outputs, dim=1).squeeze().tolist()
    # response = requests.post("https://rpmsgs3cj0.execute-api.us-east-1.amazonaws.com/run/task1", json={"text": text})
    # # response = requests.get("https://rpmsgs3cj0.execute-api.us-east-1.amazonaws.com/run/" + response["task_id"]
    # status = "PENDING"
    # while status != "COMPLETED":
    #     response1 = requests.get("https://rpmsgs3cj0.execute-api.us-east-1.amazonaws.com/run/" + response.json()["task_id"])
    #     status = response1.json()["status"]
    #     sleep(3)
    
    # # Get the prediction from the response
    # predicted_probabilities = json.loads(response1.json()["response"])["Probabilities"]
    # print(predicted_probabilities)
    # class_names = CFG.CLASSES
    # max_probability_index = np.argmax(predicted_probabilities)
    # max_probability_class = class_names[max_probability_index]
    # max_probability = predicted_probabilities[max_probability_index]
    # for class_name, probability in zip(class_names, predicted_probabilities):
    #     print(f"{class_name}: {probability:.4f}")
    # # SUBTASK 1 VISUALIZATION
    # with st.expander(f"### Get Genre for this Article", expanded=False):
    #     # st.markdown(f"### This article is classified as: {max_probability_class}")
    #     st.markdown("#### Output Probabilities Pie Chart")
    #     fig, ax = plt.subplots()
    #     ax.pie(predicted_probabilities, labels=class_names, autopct="%1.1f%%")
    #     ax.axis("equal")
    #     st.pyplot(fig)
if text:
    # SUBTASK 2 VISUALIZATION
    print("hello")
    response = requests.post("https://rpmsgs3cj0.execute-api.us-east-1.amazonaws.com/run/task2", json={"title": title, "content": text})
    print(response)
    # response = requests.get("https://rpmsgs3cj0.execute-api.us-east-1.amazonaws.com/run/" + response["task_id"]
    status = "PENDING"
    while status != "COMPLETED":
        response1 = requests.get("https://rpmsgs3cj0.execute-api.us-east-1.amazonaws.com/run/" + response.json()["task_id"])
        print(response1)
        status = response1.json()["status"]
        print(status)
        sleep(2)
        
    print("done")
    
    # Get the prediction from the response
    predicted_probabilities = json.loads(response1.json()["response"])
    # Get the prediction from the response
    # output = response.json()
    # data = output
    print(predicted_probabilities)
    df = pd.DataFrame(predicted_probabilities)
    sorted_df = df.sort_values(by='Probabilities', ascending=False)
    sorted_df = sorted_df.reset_index(drop=True)

    with st.expander(f"### Get analysis for this Article", expanded=False):
        x = sorted_df["Probabilities"]
        y = sorted_df["Labels"]
        chart = alt.Chart(sorted_df).mark_bar().encode(
            x = 'Probabilities',
            y = 'Labels',
            color=alt.Color('Labels', scale=alt.Scale(scheme='category10')),
            tooltip=['Probabilities', 'Labels']
        ).properties(
            width=1000,
            height=600
        )
        # Display the chart using Streamlit
        st.altair_chart(chart, use_container_width=True)

    
    response = requests.post("https://rpmsgs3cj0.execute-api.us-east-1.amazonaws.com/run/task3", json={"text": text})
    # response = requests.get("https://rpmsgs3cj0.execute-api.us-east-1.amazonaws.com/run/" + response["task_id"]
    status = "PENDING"
    while status != "COMPLETED":
        response1 = requests.get("https://rpmsgs3cj0.execute-api.us-east-1.amazonaws.com/run/" + response.json()["task_id"])
        status = response1.json()["status"]
        sleep(3)
    
    # Get the prediction from the response
    prediction = json.loads(response1.json()["response"])

    print(prediction)
    
    #SUBTASK 3 Visualization
    label_colors = {}  # Dictionary to store assigned colors for each label
    
    with st.expander(f"### Get Persuasion Techniques for this Article", expanded=False):
        for sentence, entry in prediction.items():
                labels = entry['Labels']
                outputs = entry['Probabilities']
    
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
