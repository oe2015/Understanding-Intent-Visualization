
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import json
# import folium
# import pygal
import plotly.graph_objects as go
import plotly.offline as py
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide")

# Read the data from the JSON file
with open("newfile.jsonl", "r") as f:
    sample_data = [json.loads(line) for line in f.readlines()]

# Create a DataFrame from the sample data
sample_data_df = pd.DataFrame(sample_data)

# Group the data by country and calculate the number of persuasion techniques
country_counts = sample_data_df.groupby('country')['persuasion_techniques'].count().reset_index()
print(country_counts)

# Select countries (allowing multiple selections)
selected_countries = st.multiselect('Select countries', country_counts['country'].unique())

# If no countries are selected, default to all countries
if not selected_countries:
    selected_countries = country_counts['country'].unique()

###########################################################################
# Create a Choropleth plot
fig = go.Figure(go.Choropleth())
# Update layout settings
fig.update_layout(height=300, margin={"r": 0, "t": 0, "l": 0, "b": 0})
# Define the countries to highlight
highlighted_countries = selected_countries
# Set the country names and marker color for highlighting
fig.add_trace(go.Choropleth(
    locationmode='country names',
    locations=highlighted_countries,
    z=[1] * len(highlighted_countries),  # Set a uniform value to show the entire country
    colorscale=[[0, 'red']],
    showscale=False
))
# Render the plot using Streamlit
st.plotly_chart(fig)
###########################################################################

# Explode the persuasion techniques so each one gets its own row
sample_data_df = sample_data_df.explode('persuasion_techniques')
# Group the data by country and persuasion techniques
country_technique_counts = sample_data_df[sample_data_df['country'].isin(selected_countries)].groupby(['country', 'persuasion_techniques']).size().unstack()

# Transpose the DataFrame
country_technique_counts = country_technique_counts.transpose()

# Check if country_technique_counts is empty
if not country_technique_counts.empty:

    # Reset index
    country_technique_counts.reset_index(inplace=True)

    # Convert DataFrame from wide to long format
    country_technique_counts_long = country_technique_counts.melt(id_vars='persuasion_techniques', var_name='country', value_name='counts')

    fig = px.bar(country_technique_counts_long, 
                 y="persuasion_techniques", 
                 x="counts", 
                 color="country", 
                 title="Number of Persuasion Techniques by Country", 
                 labels={'counts':'Count', 'persuasion_techniques':'Persuasion Technique'}, 
                 height=600, 
                 orientation='h')
    
    # Add solid lines for axes
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_layout(autosize=False, width=800, height=600, margin=dict(l=50,r=50,b=100,t=100,pad=4))

    st.plotly_chart(fig)

else:
    st.write("Please select one or more countries.")

# Remove the time zone offset from the pubDate column
sample_data_df['pubDate'] = sample_data_df['pubDate'].str[:-6]

# Convert pubDate column to datetime
sample_data_df['pubDate'] = pd.to_datetime(sample_data_df['pubDate'], format="%Y-%m-%dT%H:%M:%S")

# Group the data by country and date
grouped_data = sample_data_df.groupby(['country', sample_data_df['pubDate'].dt.date]).agg(
    num_articles=('id', 'count'),
    num_techniques=('persuasion_techniques', lambda x: sum(len(techs) for techs in x) / len(x))
).reset_index()

# Filter data based on selected countries
filtered_data = grouped_data[grouped_data['country'].isin(selected_countries)]

# Reshape the data from wide format to long format
long_data_articles = filtered_data.melt(id_vars=['country', 'pubDate'], value_vars='num_articles', var_name='measure', value_name='value')
long_data_techniques = filtered_data.melt(id_vars=['country', 'pubDate'], value_vars='num_techniques', var_name='measure', value_name='value')
# Default to "Aggregate" if no option is chosen
aggregate_compare_option = st.radio("Select view", options=["Aggregate", "Compare"], index=0)

if aggregate_compare_option == "Aggregate":
    # ... Code for "Aggregate" view ...
    aggregated_data_articles = long_data_articles.groupby(['measure', 'pubDate'])['value'].sum().reset_index()
    aggregated_data_techniques = long_data_techniques.groupby(['measure', 'pubDate'])['value'].mean().reset_index()

    fig_articles = go.Figure()
    fig_articles.add_trace(go.Scatter(x=aggregated_data_articles['pubDate'], y=aggregated_data_articles['value'], mode='lines', name='Number of Articles'))
    fig_articles.update_layout(title='Number of Articles Over Time', xaxis_title='Date', yaxis_title='Number of Articles')

    fig_techniques = go.Figure()
    fig_techniques.add_trace(go.Scatter(x=aggregated_data_techniques['pubDate'], y=aggregated_data_techniques['value'], mode='lines', name='Average Techniques per Article'))
    fig_techniques.update_layout(title='Average Techniques per Article Over Time', xaxis_title='Date', yaxis_title='Average Techniques per Article')

    st.plotly_chart(fig_articles)
    st.plotly_chart(fig_techniques)

elif aggregate_compare_option == "Compare":
    # ... Code for "Compare" view ...
    fig_articles_compare = go.Figure()
    fig_techniques_compare = go.Figure()

    for country in long_data_articles['country'].unique():
        country_data_articles = long_data_articles[long_data_articles['country'] == country]
        fig_articles_compare.add_trace(go.Scatter(x=country_data_articles['pubDate'], y=country_data_articles['value'], mode='lines', name=f'{country}'))

    for country in long_data_techniques['country'].unique():
        country_data_techniques = long_data_techniques[long_data_techniques['country'] == country]
        fig_techniques_compare.add_trace(go.Scatter(x=country_data_techniques['pubDate'], y=country_data_techniques['value'], mode='lines', name=f'{country}'))

    fig_articles_compare.update_layout(title='Number of Articles Over Time (Comparison)', xaxis_title='Date', yaxis_title='Number of Articles')
    fig_techniques_compare.update_layout(title='Average Techniques per Article Over Time (Comparison)', xaxis_title='Date', yaxis_title='Average Techniques per Article')

    st.plotly_chart(fig_articles_compare)
    st.plotly_chart(fig_techniques_compare)



import streamlit as st
import pandas as pd
import json
import plotly.express as px
# Read the JSON data
with open("sample_data_2.jsonl", "r") as f:
    sample_data = [json.loads(line) for line in f.readlines()]
# Create a dictionary to store the aggregated framings
framings_data = {}
# Process the JSON data
for data in sample_data:
    country = data['country']
    framings = data['framings']
    # Calculate the total percentage for each framing
    for framing in framings:
        framing_name = framing[0]
        framing_percentage = framing[1]
        if framing_name in framings_data:
            if country in framings_data[framing_name]:
                framings_data[framing_name][country] += framing_percentage
            else:
                framings_data[framing_name][country] = framing_percentage
        else:
            framings_data[framing_name] = {country: framing_percentage}
# Create a DataFrame from the aggregated framings data
df = pd.DataFrame(framings_data)
# Reset the index to make 'Country' a column
df = df.reset_index()
# Rename the 'index' column to 'Country'
df = df.rename(columns={'index': 'Country'})
# Melt the DataFrame to have a separate row for each framing
df = df.melt(id_vars=['Country'], var_name='Framing', value_name='Percentage')
# Filter by country
selected_countries = st.multiselect('Select Countries', df['Country'].unique())
df_filtered = df[df['Country'].isin(selected_countries)]  if selected_countries else df
# Plotting the graph using Plotly Express
fig = px.bar(df_filtered, x='Percentage', y='Country', color='Framing', orientation='h')
st.plotly_chart(fig, use_container_width=True)

import pandas as pd
import json
import numpy as np
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
import streamlit as st
import plotly.express as px
# Function to get region based on country
def get_region(country):
    try:
        # Convert country name to ISO alpha-2 code
        country_code = country_name_to_country_alpha2(country)
        # Convert ISO alpha-2 code to continent code
        continent_code = country_alpha2_to_continent_code(country_code)
        # Return the continent code as region
        return continent_code
    except KeyError:
        # If the country is not found, return "Unknown"
        return "Unknown"
st.title('News Analysis Dashboard')
st.markdown("""
Distribution of Propaganda Techniques in the Articles across Countries
""")
with open("sample_data_2.jsonl", "r") as f:
    sample_data = [json.loads(line) for line in f.readlines()]
# Convert problematic column to string type
for data in sample_data:
    if "framings" in data:
        data["framings"] = str(data["framings"])
sample_data_csv = pd.DataFrame(sample_data)
random_numbers = np.random.randint(8, size=len(sample_data_csv))
# Add random numbers as a new column in the DataFrame
sample_data_csv['propaganda_percentage_1'] = random_numbers
sample_data_csv['persuasion_technique1'] = 'Red_Herring'
sample_data_csv['persuasion_technique2'] = 'Name_Calling'
random_numbers = np.random.randint(8, size=len(sample_data_csv))
sample_data_csv['propaganda_percentage_2'] = random_numbers
sample_data_csv['region'] = sample_data_csv['country'].apply(get_region)
sample_data_csv = sample_data_csv.dropna(subset=['region', 'country', 'source'])
selected_countries = st.sidebar.multiselect('Choose one or more countries', sample_data_csv.country.unique())
# selected_topics = st.sidebar.selectbox('Topics', sample_data_csv.source.unique())
selected_countries = list(selected_countries)
# Filter the dataframe based on selected countries
filtered_df = sample_data_csv[sample_data_csv['country'].isin(selected_countries)]
# Display the filtered
# st.dataframe(filtered_df)
filtered_df1 = filtered_df.drop(['persuasion_technique1', 'propaganda_percentage_1'], axis=1)
filtered_df1.rename(columns={'persuasion_technique2': 'persuasion_technique1', 'propaganda_percentage_2' : 'propaganda_percentage_1'}, inplace=True)
filtered_df2 = filtered_df.drop(['persuasion_technique2', 'propaganda_percentage_2'], axis=1)
merged_df = pd.concat([filtered_df1, filtered_df2])
# st.dataframe(merged_df)
fig = px.sunburst(merged_df, path=['region', 'country',  'persuasion_technique1'], values='propaganda_percentage_1',  color='propaganda_percentage_1',
                 )
fig.update_layout(width=1200, height=1000)
st.plotly_chart(fig)



