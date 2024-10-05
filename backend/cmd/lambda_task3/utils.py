from nltk.tokenize import sent_tokenize


def extract_title_and_sentences(text):
    # Split the text into sentences
    sentences = sent_tokenize(text)
    # Extract the title from the first sentence
    title = sentences[0]
    sentences = sentences[1:]
    # Remove the title from the list of sentences
    # if len(sentences) > 10:
    #     sentences = sentences[1:10]
    return sentences
