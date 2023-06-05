import streamlit as st
import pandas as pd
import numpy as np

countries = ['Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia',
             'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary',
             'Iceland', 'Ireland', 'Italy', 'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta',
             'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Poland', 'Portugal',
             'Romania', 'Russia', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland',
             'Ukraine', 'United Kingdom', 'Vatican City']

propaganda_techniques = ['Loaded_Language', 'Appeal_to_fear_prejudice', 'Name_Calling', 'Flag_waiving', 'doubt',
                         'exaggeration/minimization', 'Slogans', 'Casual_oversimplification', 'repetition',
                         'thought_terminating_cliches', 'appeal_to_authority', 'black_and_white_fallacy',
                         'reductio_ad_hitlerum', 'whataboutism', 'straw_men', 'red_herring',
                         'obfuscation/intentional_vagueness/confusion']

articles_per_country = 10

data = []
for country in countries:
    for _ in range(articles_per_country):
        article = f"Article from {country}"
        row = [article, country]
        row.extend(np.random.choice(6, size=len(propaganda_techniques), p=[0.45, 0.1, 0.1, 0.1, 0.1, 0.15]))
        data.append(row)

df = pd.DataFrame(data, columns=['Article', 'Country'] + propaganda_techniques)


# Dataframes
st.header('Dataframes')
st.write(df)