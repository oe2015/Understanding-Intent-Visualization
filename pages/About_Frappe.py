
# import streamlit as st

# framing_explanations = {
#     "Economic": "Costs, benefits, or other financial implications",
#     "Capacity and resources": "Availability of physical, human or financial resources, and capacity of current systems",
#     "Morality": "Religious or ethical implications",
#     "Fairness and equality": "Balance or distribution of rights, responsibilities, and resources",
#     "Legality Constitutionality and jurisprudence": "Rights, freedoms, and authority of individuals, corporations, and government",
#     "Policy prescription and evaluation": "Discussion of specific policies aimed at addressing problems",
#     "Crime and punishment": "Effectiveness and implications of laws and their enforcement",
#     "Security and defense": "Threats to welfare of the individual, community, or nation",
#     "Health and safety": "Health care, sanitation, public safety",
#     "Quality of life": "Threats and opportunities for the individual's wealth, happiness, and well-being",
#     "Cultural identity": "Traditions, customs, or values of a social group in relation to a policy issue",
#     "Public opinion": "Attitudes and opinions of the general public, including polling and demographics",
#     "Political": "Considerations related to politics and politicians, including lobbying, elections, and attempts to sway voters",
#     "External regulation and reputation": "International reputation or foreign policy"
# }

# techniques = {
#     "Name Calling or Labelling": "Using insulting or desirable labels for individuals or groups.",
#     "Guilt by Association": "Attacking by linking to negatively viewed groups or concepts.",
#     "Casting Doubt": "Undermining credibility by questioning character.",
#     "Appeal to Hypocrisy": "Accusing of hypocrisy to attack reputation.",
#     "Questioning the Reputation": "Undermining character with negative claims.",
#     "Flag Waiving": "Justifying ideas by appealing to group pride or benefits.",
#     "Appeal to Authority": "Citing authority to support an argument.",
#     "Appeal to Popularity": "Claiming widespread agreement to justify a stance.",
#     "Appeal to Values": "Linking ideas to positive values.",
#     "Appeal to Fear, Prejudice": "Using fear or prejudice to promote or reject ideas.",
#     "Strawman": "Misrepresenting an argument to refute it easily.",
#     "Red Herring": "Distracting from the main issue with irrelevant topics.",
#     "Whataboutism": "Accusing of hypocrisy without disproving the argument.",
#     "Causal Oversimplification": "Oversimplifying causes of an issue.",
#     "False Dilemma or No Choice": "Presenting only two options when more exist.",
#     "Consequential Oversimplification": "Claiming improbable chain reactions.",
#     "Slogans": "Using catchy phrases with emotional appeal.",
#     "Conversation Killer": "Discouraging discussion with dismissive phrases.",
#     "Appeal to Time": "Arguing that it's the right time for action.",
#     "Loaded Language": "Using emotionally charged words to influence.",
#     "Obfuscation, Intentional Vagueness, Confusion": "Being unclear to allow varied interpretations.",
#     "Exaggeration or Minimisation": "Overstating or downplaying significance.",
#     "Repetition": "Repeating phrases to persuade."
# }

# st.title("Welcome to FRAPPE: Your FRAming, Persuasion, and Propaganda Explorer")

# st.write("""
# In today's information-rich world, distinguishing reliable news from misleading content is more crucial than ever. FRAPPE is here to help you navigate this complex landscape.
# """)

# st.header("What FRAPPE Does:")
# st.markdown("""
# - Analyzes news articles beyond the surface level
# - Uncovers linguistic techniques that shape opinions and emotions
# - Analyzes articles for specific frames and persuasion techniques, including:
# """)

# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Frames:")
#     for frame, explanation in framing_explanations.items():
#         st.markdown(f"- {frame}", help=explanation)

# with col2:
#     st.subheader("Persuasion Techniques:")
#     for technique, explanation in techniques.items():
#         st.markdown(f"- {technique}", help=explanation)

# st.markdown("""
# - Compares persuasion and framing strategies across:
#   - Various news outlets
#   - Multiple countries
#   - Different languages and topics
# """)

# st.header("Why Use FRAPPE:")
# st.markdown("""
# - Gain deeper insights into news content
# - Understand how information is presented and potentially manipulated
# - Develop a more critical and informed approach to media consumption
# """)

# st.header("Learn More:")
# st.markdown("""
# - Read our research paper: [FRAPPE: FRAming, Persuasion, and Propaganda Explorer](https://aclanthology.org/2024.eacl-demo.22/)
# - Watch our demo video: [FRAPPE in Action](https://aclanthology.org/2024.eacl-demo.22.mp4)
# """)

# st.markdown("**Empower yourself with FRAPPE and become a more discerning news reader!**")

import streamlit as st

# Dictionaries for frames and techniques (as provided earlier)
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

st.set_page_config(layout="wide", page_title="About FRAPPE")

# Custom CSS
st.markdown("""
<style>
    body {
        color: #E0E0E0;
        background-color: #1E1E1E;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #4CAF50;
    }
    .section {
        padding: 1.5rem;
        background-color: #2A2A2A;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        background-color: #2A2A2A;
        border-radius: 10px;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Welcome to FRAPPE: Your FRAming, Persuasion, and Propaganda Explorer")

st.write("In today's information-rich world, distinguishing reliable news from misleading content is more crucial than ever. FRAPPE is here to help you navigate this complex landscape.")

tab1, tab2, tab3 = st.tabs(["What FRAPPE Does", "Why Use FRAPPE", "Learn More"])

with tab1:
    st.header("What FRAPPE Does")
    st.markdown("""
    <div class="section">
    <ul>
        <li>Analyzes news articles beyond the surface level</li>
        <li>Uncovers linguistic techniques that shape opinions and emotions</li>
        <li>Analyzes articles for specific frames and persuasion techniques</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Frames")
        for frame, explanation in framing_explanations.items():
            st.markdown(f"- {frame}", help=explanation)
    
    with col2:
        st.subheader("Persuasion Techniques")
        for technique, explanation in techniques.items():
            st.markdown(f"- {technique}", help=explanation)
    
    st.markdown("""
    <div class="section">
    <h3>FRAPPE compares persuasion and framing strategies across:</h3>
    <ul>
        <li>Various news outlets</li>
        <li>Multiple countries</li>
        <li>Different languages and topics</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("Why Use FRAPPE")
    st.markdown("""
    <div class="section">
    <ul>
        <li>Gain deeper insights into news content</li>
        <li>Understand how information is presented and potentially manipulated</li>
        <li>Develop a more critical and informed approach to media consumption</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.header("Learn More")
    st.markdown("""
    <div class="section">
    <ul>
        <li>Read our research paper: <a href="https://aclanthology.org/2024.eacl-demo.22/">FRAPPE: FRAming, Persuasion, and Propaganda Explorer</a></li>
        <li>Watch our demo video: <a href="https://aclanthology.org/2024.eacl-demo.22.mp4">FRAPPE in Action</a></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
<strong>Empower yourself with FRAPPE and become a more discerning news reader!</strong>
<br><br>
Contact us: frappe@example.com | Follow us: @FRAPPEtool
</div>
""", unsafe_allow_html=True)