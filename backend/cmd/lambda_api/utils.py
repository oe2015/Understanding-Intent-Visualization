import torch
from nltk.tokenize import sent_tokenize


def split_title_content(text):
    # Split the text by the first new line character
    parts = text.split("\n", 1)
    # Check if there is a new line character in the text
    if len(parts) > 1:
        title = parts[0]  # First part is the title
        content = parts[1]  # Second part is the content
    else:
        title = parts[0]  # Entire text is considered as the title
        content = ""
    # Split the title by the first period (.) to get the first sentence
    title_parts = title.split(".", 1)
    # Check if there is a period in the title
    if len(title_parts) > 1:
        first_sentence = title_parts[0] + "."  # Add the period back to the first sentence
        title = title_parts[1]  # Remaining part becomes the new title
    return title.strip(), content.strip()


def update_and_load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    for k in list(ckpt.keys()):
        if k.startswith("head2."):
            # rename the key to classifier
            ckpt[k.replace("head2.", "classifier.")] = ckpt.pop(k)
        if k.startswith("model."):
            # rename the key to classifier
            ckpt[k.replace("model.", "")] = ckpt.pop(k)
    return ckpt


def extract_title_and_sentences(text):
    # Split the text into sentences
    sentences = sent_tokenize(text)
    # Extract the title from the first sentence
    title = sentences[0]
    sentences = sentences[1:]
    # Remove the title from the list of sentences
    if len(sentences) > 10:
        sentences = sentences[1:10]
    return sentences


def split_title_content(text):
    # Split the text by the first new line character
    parts = text.split("\n", 1)
    # Check if there is a new line character in the text
    if len(parts) > 1:
        title = parts[0]  # First part is the title
        content = parts[1]  # Second part is the content
    else:
        title = parts[0]  # Entire text is considered as the title
        content = ""
    # Split the title by the first period (.) to get the first sentence
    title_parts = title.split(".", 1)
    # Check if there is a period in the title
    if len(title_parts) > 1:
        first_sentence = title_parts[0] + "."  # Add the period back to the first sentence
        title = title_parts[1]  # Remaining part becomes the new title
    return title.strip(), content.strip()
