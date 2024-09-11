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
