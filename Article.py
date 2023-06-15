import streamlit as st
import seaborn as sns

import pandas as pd
import numpy as np
import altair as alt
# from sklearn.preprocessing import MultiLabelBinarizer
import json
import torch
# from transformers import XLMRobertaTokenizer
from architecture import CustomModel
from config import CFG
import numpy as np

# import torch
# from torch.utils.data import  DataLoader

# from download_button import download_button
##################################################################################
# import os
# from transformers import AutoConfig

# def upload_model(model_path, model_name, repo_name):
#     # Create the correct configuration for your model
#     cfg = AutoConfig.from_pretrained(CFG.MODEL_NAME, num_labels=CFG.NUM_CLASSES)
#     cfg.save_pretrained(repo_name)

#     # Then, load your actual model weights, and save them with the config file
#     model = CustomModel(CFG)

#     model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))["state_dict"]
#     model_state_dict = {
#         k.replace("model.model.", "model."): v for k, v in model_state_dict.items()
#     }

#     model_state_dict = {
#         k.replace("model.fc.", "fc."): v for k, v in model_state_dict.items()
#     }

#     model.load_state_dict(model_state_dict)
#     torch.save(model.state_dict(), os.path.join(repo_name, 'pytorch_model.bin'))

# # Call the function with model_path, the desired model_name, and the repo name
# upload_model("best_model_exp_9.ckpt", "my_custom_model", "XLMRobertaNews")



#################################################################################

from newspaper import Article
import matplotlib.pyplot as plt
flag = 0
st.set_page_config(
    page_title="News Framing Demo",
    page_icon="📰",
)
# @st.cache_resource
# def fetch_load():

#     # Define the path to your JSONL file
#     file_path = "newfile.jsonl"

#     # Define the desired batch size
#     batch_size = 10000000

#     # Create an empty list to store the DataFrame chunks
#     df_chunks = []

#     # Read the JSONL file in batches
#     with open(file_path, "r") as f:
#         # Read a batch of lines from the file
#         lines = f.readlines(batch_size)

#         # Convert the lines to JSON objects
#         sample_data = [json.loads(line) for line in lines]

#         # Convert problematic column to string type
#         for data in sample_data:
#             if "labels" in data:
#                 data["labels"] = str(data["labels"])

#         # Create a DataFrame from the batch of JSON objects
#         df = pd.DataFrame(sample_data)

#         # Append the DataFrame to the list of chunks
#         df_chunks.append(df)

#     # Concatenate the DataFrame chunks into a single DataFrame
#     sample_data_df = pd.concat(df_chunks)

#     # Display the DataFrame
#     return sample_data_df


# sample_data_df = fetch_load()
# sample_data_df = sample_data_df.drop_duplicates()
# st.write(sample_data_df)


# @st.cache_resource
# def load_model():
    
#     tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

#     # Load fine-tuned model checkpoint
#     # model_path = "runs/exp_3/checkpoints/best_model_xlmroberta.ckpt"
#     model_path = "https://mbzuaiac-my.sharepoint.com/:u:/g/personal/osama_afzal_mbzuai_ac_ae/EbC5ht0iUuRPsbhwad4j0JIBlVox9_0Zud5xLOjZBMtxWQ?e=kZRoY4"
#     model_state_dict = torch.load(model_path)["state_dict"]
#     model_state_dict = {
#         k.replace("model.model.", "model."): v for k, v in model_state_dict.items()
#     }

#     model_state_dict = {
#         k.replace("model.fc.", "fc."): v for k, v in model_state_dict.items()
#     }

#     # Instantiate your customized XLM-RoBERTa model
#     custom_xlm_roberta = CustomModel(CFG)
# from transformers import XLMRobertaTokenizer

#     # Load the model state dictionary
#     custom_xlm_roberta.load_state_dict(model_state_dict)

#     # Set the model in evaluation mode
#     custom_xlm_roberta.eval()
#     return tokenizer, custom_xlm_roberta


from transformers import XLMRobertaTokenizer, AutoModel
from huggingface_hub import hf_hub_download


@st.cache_resource
def load_model():
    # Specify your Hugging Face model hub repository
    model_name = "oe2015/XLMRobertaNews"

    # Load the tokenizer
    # tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

    # tokenizer = AutoTokenizer.from_pretrained("oe2015/XLMRobertaNews")

    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")


    # Load the model
    # model = AutoModel.from_pretrained("oe2015/XLMRobertaNews")

    # Download the model weights
    model_file = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")

    model = CustomModel(CFG) # Initialize your model with its configuration
    model.load_state_dict(torch.load(model_file, map_location="cpu"))

    # Set the model in evaluation mode
    model.eval()

    return tokenizer, model


tokenizer, custom_xlm_roberta = load_model()


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

                # if flag:
                #     article.parse()
                #     doc = article.text
        

elif option == "Enter Text":
    with st.form(key="text_form"):
        text = st.text_area("Enter the text to get framings")
        submit_button = st.form_submit_button(label="Get news framing!")

        if submit_button:
            doc = text


            with st.expander("Read the article", expanded=False):
                st.write(doc)






# with st.form(key="my_form"):
#     url = st.text_input(
#         "Enter the article url to get framings",
#     )

#     submit_button = st.form_submit_button(label="Get news framing!")

# if not submit_button:
#     st.stop()

# if len(url) > 0:
#     article = Article(url)
#     try:
#         article.download()
#         article.parse()
#         doc = article.text
#         st.warning("Article successfully processed.")
#         flag = 1 
#     except:
#         st.warning("Access denied.")
#         exit()

#     if flag:
#         article.parse()
#         doc = article.text
#         with st.expander("Read the article", expanded=False):
#             st.write(doc)

        # mask = sample_data_df["link"] == url

        # # Apply the mask to retrieve the filtered rows
        # labels = sample_data_df[mask]
        
        # labels = labels["labels"].to_numpy()
        # labels = eval(labels[0])
        # df = pd.DataFrame(labels, columns =["labels","probability"])
        
        # x = df["probability"]
        # y = df["labels"]
        # chart = alt.Chart(df).mark_bar().encode(
        #     x = 'probability',
        #     y = 'labels',
        #     color=alt.Color('labels', scale=alt.Scale(scheme='category10')),
        #     tooltip=['probability', 'labels']
        # ).properties(
        #     width=600,
        #     height=400
        # )

        # # Display the chart using Streamlit
        # st.altair_chart(chart, use_container_width=True)



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
if doc:
    outputs = custom_xlm_roberta(inputs)
    print(outputs)

    predicted_probabilities = torch.softmax(outputs, dim=1).squeeze().tolist()

    # Print predicted probabilities for each class
    class_names = CFG.CLASSES
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

