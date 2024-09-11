import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import numpy as np
import ast
from ast import literal_eval
from streamlit_server_state import server_state, server_state_lock
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2


st.set_page_config(layout="wide")

# df = pd.read_parquet('newfile.parquet')
# print(df['source'])
# print(df['labels'])



# Load the DataFrame
# df = pd.read_parquet('subtask3.parquet')
# print(df)

# # Drop the 'labels.None' column
# df = df.drop('labels.None', axis=1)

# # Group by the 'source' column and compute the sum for each label
# media_agg_subtask3 = df.groupby('source').sum()

# # # Calculate the total number of articles for each source
# # total_articles_subtask3 = df.groupby('source').size()

# # # Normalize the sums by the total number of articles
# # media_agg_subtask3 = media_agg_subtask3.div(total_articles_subtask3, axis=0)

# # Save DataFrame
# media_agg_subtask3.to_parquet('media_agg_subtask3.parquet')
# print(media_agg_subtask3)

###############################################################################################

# import pandas as pd
# import json
# import pyarrow as pa
# import pyarrow.parquet as pq
# # Read the Parquet file
# df = pd.read_parquet('subtask3.parquet')
# # Drop the 'labels.None' column
# df = df.drop('labels.None', axis=1)
# # Group the data and count occurrences
# aggregated_df = df.groupby(['country', 'source']).size().reset_index(name='frequency')
# # Convert the DataFrame into the desired format
# result = aggregated_df.groupby('country').apply(lambda x: x[['source', 'frequency']].values.tolist()).reset_index(name='source_frequencies')
# # Convert the result DataFrame to a dictionary
# result_dict = result.to_dict(orient='records')
# # Save the result as a JSON file
# with open('country2.json', 'w') as f:
#     json.dump(result_dict, f)
# # Read the JSON file
# with open('country2.json', 'r') as f:
#     data = json.load(f)
# # Convert the data back to a DataFrame
# df = pd.DataFrame(data)clea
# # Convert the nested lists to JSON strings
# df['source_frequencies'] = df['source_frequencies'].apply(json.dumps)
# # Convert DataFrame to PyArrow Table
# table = pa.Table.from_pandas(df)
# # Save the Table as a Parquet file
# pq.write_table(table, 'country_media_subtask3.parquet', compression='snappy')
# df = pd.read_parquet('country_media_subtask3.parquet')
# print(pd.DataFrame(df))

###################################################################################################

# Group by the 'country' column and compute the sum for each label
# aggregated_df_subtask3 = df.groupby('country').sum()
# # print(aggregated_df_subtask3)

# # Calculate the total number of articles for each country
# # total_articles_subtask3 = df.groupby('country').size()
# # print(total_articles_subtask3)
# # Normalize the sums by the total number of articles
# # aggregated_df_subtask3 = aggregated_df_subtask3.div(total_articles_subtask3, axis=0)

# # Save DataFrame
# aggregated_df_subtask3.to_parquet('aggregated_df_subtask3.parquet')

# print(aggregated_df_subtask3)

# ####################################################################################

# # Remove timezone part and convert dates to datetime format
# df['pubDate'] = pd.to_datetime(df['pubDate'].str[:-6])

# df.set_index('pubDate', inplace=True)

# # Resample the data by month and count the number of articles in each month
# article_counts_subtask3 = df.resample('M').size()

# # Convert series to DataFrame
# article_counts_df_subtask3 = article_counts_subtask3.to_frame('number_of_articles')

# article_counts_df_subtask3.to_parquet('article_counts_df_subtask3.parquet')

# print(article_counts_df_subtask3)

# ###############################################################################

# # Group the data by 'country' and 'pubDate'
# # Then resample by month and count the number of articles in each month
# country_article_counts_subtask3 = df.groupby('country').resample('M').size()

# # Convert series to DataFrame
# country_article_counts_df_subtask3 = country_article_counts_subtask3.to_frame('number_of_articles')

# # Save the DataFrame to a parquet file
# country_article_counts_df_subtask3.to_parquet('country_article_counts_df_subtask3.parquet')

# print(country_article_counts_df_subtask3)

###################################################################################################
# Group the data by 'country' and 'pubDate'
# Then resample by month and count the number of articles in each month
# source_article_counts_subtask3 = df.groupby('source').resample('M').size()

# # Convert series to DataFrame
# source_article_counts_subtask3 = source_article_counts_subtask3.to_frame('number_of_articles')

# # Save the DataFrame to a parquet file
# source_article_counts_subtask3.to_parquet('source_article_counts_df_subtask3.parquet')

# print(source_article_counts_subtask3)

##########################################################################################################
##########################################################################################################

# # load the dataframe
# df = pd.read_parquet('newfile.parquet')

# # Convert the labels from string to list of lists
# df['labels'] = df['labels'].apply(ast.literal_eval)

# # Transform the list of lists into a dictionary
# df['labels'] = df['labels'].apply(lambda x: {k: v for k, v in x})

# # Create a DataFrame from the labels column
# labels_df = pd.DataFrame(list(df['labels']))

# # Concatenate this DataFrame with the original DataFrame
# final_df = pd.concat([df.drop('labels', axis=1), labels_df], axis=1)

# # Group by the 'source' column and compute the sum for each label
# media_agg = final_df.groupby('source').sum()

# # Calculate the total number of articles for each source
# total_articles = final_df.groupby('source').size()

# # Normalize the sums by the total number of articles
# media_agg = media_agg.div(total_articles, axis=0)

# # Save DataFrame
# media_agg.to_parquet('media_agg.parquet')

# Load DataFrame
# media_agg = pd.read_parquet('media_agg.parquet')

###############################################################################################

# Assuming your dataframe df has a 'country' column
# Count the number of articles per source and country
# count_df = df.groupby(['country', 'source']).size().reset_index(name='counts')

# # Convert the counts to a string so they can be combined with the source
# count_df['counts'] = count_df['counts'].astype(str)

# # Combine the source and counts columns
# count_df['source_counts'] = count_df[['source', 'counts']].apply(lambda x: {x[0]: int(x[1])}, axis=1)

# # Group by country and combine the source_counts dictionaries into a list
# country_to_media_df = count_df.groupby('country')['source_counts'].apply(list).reset_index()

# # Save DataFrame
# country_to_media_df.to_parquet('country_to_media.parquet')

# Load DataFrame
# country_to_media = pd.read_parquet('country_to_media.parquet')
# print(country_to_media_df)

###################################################################################################
# print("hello")
# sample_data_df = pd.DataFrame(df)

# import json

# # Convert the labels from string to list of lists
# sample_data_df['labels'] = sample_data_df['labels'].apply(ast.literal_eval)

# # Transform the list of lists into a dictionary
# sample_data_df['labels'] = sample_data_df['labels'].apply(lambda x: {k: v for k, v in x})

# # Create a DataFrame from the labels column
# labels_df = pd.DataFrame(list(sample_data_df['labels']))
# print("hello")

# # Concatenate this DataFrame with the original DataFrame
# final_df = pd.concat([sample_data_df.drop('labels', axis=1), labels_df], axis=1)
# print("hello")
# print(final_df)

# # Group by the 'country' column and compute the sum for each label
# aggregated_df = final_df.groupby('country').sum()
# print("hello")

# # Calculate the total number of articles for each country
# total_articles = final_df.groupby('country').size()
# print("hello")

# # Normalize the sums by the total number of articles
# aggregated_df = aggregated_df.div(total_articles, axis=0)
# print("hello")

# # Save DataFrame
# aggregated_df.to_parquet('aggregated_df.parquet')

# Load DataFrame
# aggregated_df = pd.read_parquet('aggregated_df.parquet')
# print(aggregated_df)


####################################################################################

# Remove timezone part and convert dates to datetime format
# df['pubDate'] = pd.to_datetime(df['pubDate'].str[:-6])

# df.set_index('pubDate', inplace=True)

# print(type(df.index))  # print type of the index

# # Resample the data by month and count the number of articles in each month
# article_counts = df.resample('M').size()

# # Convert series to DataFrame
# article_counts_df = article_counts.to_frame('number_of_articles')

# article_counts_df.to_parquet('article_counts_df.parquet')

# article_counts_df = pd.read_parquet('article_counts_df.parquet')
# print(article_counts_df)

###############################################################################

# Group the data by 'country' and 'pubDate'
# Then resample by month and count the number of articles in each month
# country_article_counts = df.groupby('country').resample('M').size()

# # Convert series to DataFrame
# country_article_counts_df = country_article_counts.to_frame('number_of_articles')

# # Save the DataFrame to a parquet file
# country_article_counts_df.to_parquet('country_article_counts_df.parquet')

# Load the saved DataFrame
# country_article_counts_df = pd.read_parquet('country_article_counts_df.parquet')
# print(country_article_counts_df)

# print(country_article_counts_df)

# ###########################################################################

# Group the data by 'country' and 'pubDate'
# Then resample by month and count the number of articles in each month
# df['pubDate'] = pd.to_datetime(df['pubDate'].str[:-6])
# df.set_index('pubDate', inplace=True)

# source_article_counts = df.groupby('source').resample('M').size()

# # Convert series to DataFrame
# source_article_counts = source_article_counts.to_frame('number_of_articles')

# # Save the DataFrame to a parquet file
# source_article_counts.to_parquet('source_article_counts_df.parquet')

# Load the saved DataFrame
# source_article_counts = pd.read_parquet('source_article_counts_df.parquet')
# print(source_article_counts)


###############################################################################

#####loading data ######
@st.cache_data
def load_data(path):
    return pd.read_parquet(path)

# RU_media_agg = load_data('media_agg.parquet')
# # print(media_agg)
# RU_country_to_media = load_data('country_media.parquet')
# # print(country_to_media)
# RU_aggregated_df = load_data('aggregated_df.parquet')
# # print(aggregated_df)
# RU_article_counts_df = load_data('article_counts_df.parquet')
# # print(article_counts_df)
# RU_country_article_counts_df = load_data('country_article_counts_df.parquet')
# # print(country_article_counts_df)

# RU_media_agg_subtask3 = load_data('media_agg_subtask3.parquet')
# RU_country_to_media_subtask3 = load_data('country_media_subtask3.parquet')
# RU_aggregated_df_subtask3 = load_data('aggregated_df_subtask3.parquet')
# RU_article_counts_df_subtask3 = load_data('article_counts_df_subtask3.parquet')
# RU_country_article_counts_df_subtask3 = load_data('country_article_counts_df_subtask3.parquet')
# RU_source_article_counts = load_data('source_article_counts_df.parquet')
# # print(source_article_counts)
# RU_source_article_counts_subtask3 = load_data('source_article_counts_df_subtask3.parquet')
        
#     # elif option == 'Climate Change':
# CH_media_agg = load_data('media_agg_df2_new2.parquet')
# # print(media_agg)
# CH_country_to_media = load_data('country_media_df2_new.parquet')
# # print(country_to_media)
# CH_aggregated_df = load_data('aggregated_df2_new2.parquet')
# # print(aggregated_df)
# CH_article_counts_df = load_data('article_counts_df2_new.parquet')
# # print(article_counts_df)
# CH_country_article_counts_df = load_data('country_article_counts_df2_new.parquet')
# # print(country_article_counts_df)

# CH_media_agg_subtask3 = load_data('media_agg_df3_new2.parquet')
# CH_country_to_media_subtask3 = load_data('country_media_df3_new.parquet')
# CH_aggregated_df_subtask3 = load_data('aggregated_df3_new2.parquet')
# CH_article_counts_df_subtask3 = load_data('article_counts_df3_new.parquet')
# CH_country_article_counts_df_subtask3 = load_data('country_article_counts_df3_new.parquet')

# CH_source_article_counts = load_data('source_article_counts_df2_new.parquet')
# # print(source_article_counts)
# CH_source_article_counts_subtask3 = load_data('source_article_counts_df3_new.parquet') 



both_media_agg = load_data('media_agg_df2_combined.parquet')
# print(media_agg)
both_country_to_media = load_data('country_media_df2_combined.parquet')
# print(country_to_media)
both_aggregated_df = load_data('aggregated_df2_combined.parquet')
# print(aggregated_df)
both_article_counts_df = load_data('article_counts_df2_combined.parquet')
# print(article_counts_df)
both_country_article_counts_df = load_data('country_article_counts_df2_combined.parquet')
# print(country_article_counts_df)

both_media_agg_subtask3 = load_data('media_agg_df3_combined.parquet')
both_country_to_media_subtask3 = load_data('country_media_df3_combined.parquet')
both_aggregated_df_subtask3 = load_data('aggregated_df3_combined.parquet')
both_article_counts_df_subtask3 = load_data('article_counts_df3_combined.parquet')
both_country_article_counts_df_subtask3 = load_data('country_article_counts_df3_combined.parquet')
both_source_article_counts = load_data('source_article_counts_df2_combined.parquet')
# print(source_article_counts)
both_source_article_counts_subtask3 = load_data('source_article_counts_df3_combined.parquet')
       
    ###########################


pages = ["Framings and Persuasion Techniques: Countries", "Framings: Countries and Sources", "Persuasion Techniques: Fine-Grained Propaganda", "Persuasion Techniques: Coarse-Grained Propaganda", "Persuasion Techniques: Ethos, Logos, Pathos"]
# if 'page' not in st.session_state:
st.session_state['page'] = pages[0]  # Set default page to Home
st.session_state.page = st.sidebar.radio("Navigation", pages, index=pages.index(st.session_state.page))


if st.session_state.page == "Framings and Persuasion Techniques: Countries":
    # option = 'Russo-Ukrainian War'
    # option = st.selectbox('Choose a topic', ('Climate Change', 'Russo-Ukrainian War', 'Combined'), index=1)

    #####loading data ######
    # if option == 'Russo-Ukrainian War':
    #     media_agg = RU_media_agg
    #     # print(media_agg)
    #     country_to_media = RU_country_to_media
    #     # print(country_to_media)
    #     aggregated_df = RU_aggregated_df
    #     # print(aggregated_df)
    #     article_counts_df = RU_article_counts_df
    #     # print(article_counts_df)
    #     country_article_counts_df = RU_country_article_counts_df
    #     # print(country_article_counts_df)

    #     media_agg_subtask3 = RU_media_agg_subtask3
    #     country_to_media_subtask3 = RU_country_to_media_subtask3
    #     aggregated_df_subtask3 = RU_aggregated_df_subtask3
    #     article_counts_df_subtask3 = RU_article_counts_df_subtask3
    #     country_article_counts_df_subtask3 = RU_country_article_counts_df_subtask3
    #     source_article_counts = RU_source_article_counts
    #     # print(source_article_counts)
    #     source_article_counts_subtask3 = RU_source_article_counts_subtask3
    # elif option == 'Climate Change':
    #     media_agg = CH_media_agg
    #     # print(media_agg)
    #     country_to_media = CH_country_to_media
    #     # print(country_to_media)
    #     aggregated_df = CH_aggregated_df
    #     # print(aggregated_df)
    #     article_counts_df = CH_article_counts_df
    #     # print(article_counts_df)
    #     country_article_counts_df = CH_country_article_counts_df
    #     # print(country_article_counts_df)

    #     media_agg_subtask3 = CH_media_agg_subtask3
    #     country_to_media_subtask3 = CH_country_to_media_subtask3
    #     aggregated_df_subtask3 = CH_aggregated_df_subtask3
    #     article_counts_df_subtask3 = CH_article_counts_df_subtask3
    #     country_article_counts_df_subtask3 = CH_country_article_counts_df_subtask3

    #     source_article_counts = CH_source_article_counts
    #     # print(source_article_counts)
    #     source_article_counts_subtask3 = CH_source_article_counts_subtask3
        
    # elif option == 'Combined':
    media_agg = both_media_agg
    # print(media_agg)
    country_to_media = both_country_to_media
    # print(country_to_media)
    aggregated_df = both_aggregated_df
    # print(aggregated_df)
    article_counts_df = both_article_counts_df
    # print(article_counts_df)
    country_article_counts_df = both_country_article_counts_df
    # print(country_article_counts_df)

    media_agg_subtask3 = both_media_agg_subtask3
    country_to_media_subtask3 = both_country_to_media_subtask3
    aggregated_df_subtask3 = both_aggregated_df_subtask3
    article_counts_df_subtask3 = both_article_counts_df_subtask3
    country_article_counts_df_subtask3 = both_country_article_counts_df_subtask3

    source_article_counts = both_source_article_counts
    # print(source_article_counts)
    source_article_counts_subtask3 = both_source_article_counts_subtask3        
    
    ###########################

    # Number of countries
    num_countries = len(country_to_media)

    # Number of articles
    num_articles = article_counts_df['number_of_articles'].sum()

    # Number of media sources
    num_media_sources = len(media_agg)

    # Create columns
    col1, col2, col3 = st.columns(3)

    # Display statistics in each column within a box
    with col1:
        st.markdown(f"""
            <div style="background-color: #f5f3f6; border: 1px solid #6c757d; border-radius: 5px; padding: 10px">
                <h4 style="color: #6c757d; text-align: center">Number of countries</h4>
                <p style="text-align: center; font-size: 20px; color: black;">{num_countries}</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="background-color: #f5f3f6; border: 1px solid #6c757d; border-radius: 5px; padding: 10px">
                <h4 style="color: #6c757d; text-align: center">Number of articles</h4>
                <p style="text-align: center; font-size: 20px; color: black;">{num_articles}</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div style="background-color: #f5f3f6; border: 1px solid #6c757d; border-radius: 5px; padding: 10px">
                <h4 style="color: #6c757d; text-align: center">Number of media sources</h4>
                <p style="text-align: center; font-size: 20px; color: black;">{num_media_sources}</p>
            </div>
        """, unsafe_allow_html=True)

    # Initialize session_state variables if they do not exist
    if 'selected_countries' not in st.session_state:
        st.session_state.selected_countries = []

    # if 'selected_pairs' not in st.session_state:
    #     st.session_state.selected_pairs = []

    # Get unique countries
    available_countries = country_to_media['country'].unique().tolist()

    # Remove already selected countries from the list
    # available_countries = [country for country in available_countries if country not in st.session_state.selected_countries]

    ##################
    # Calculate total number of articles for each country
    # Calculate total number of articles for each country
    # Calculate total number of articles for each country
    # Calculate total number of articles for each country
    total_articles = country_article_counts_df.groupby('country').sum().reset_index()
    total_articles.columns = ['country', 'total_articles']
    
    # Create a dictionary that maps country names to total number of articles
    country_to_total_articles = total_articles.set_index('country')['total_articles'].to_dict()
    
    # Get unique countries
    available_countries = country_to_media['country'].unique().tolist()
    
    # Add the total number of articles per country
    for i in range(len(available_countries)):
        country = available_countries[i]
        if country in country_to_total_articles:
            number = str(country_to_total_articles[country])
            available_countries[i] = country + " (" + number + ")"
    
    # Now sort available_countries based on the total number of articles in descending order
    available_countries.sort(key=lambda x: int(x.split(' ')[-1].strip('()')), reverse=True)

    
    # If there are countries left to select
    if available_countries:
        country = st.selectbox('Select country', available_countries, key='country')
        index = available_countries.index(country)
        original_case_source = available_countries[index]
        if st.button('Add selection'):
            country = country[:country.index("(") - 1]
            if (country not in st.session_state.selected_countries):
                st.session_state.selected_countries.append(country)
            # Update the available countries list after a country has been added
            available_countries = [country for country in available_countries if country not in st.session_state.selected_countries]
    
    if st.button('Remove last selection', key='remove_countries') and st.session_state.selected_countries:
        last_removed = st.session_state.selected_countries.pop()
        # Add the removed country back to the available countries list
        available_countries.append(last_removed)
    
    selected_countries = st.session_state.selected_countries



    # rest of your code


    # Now you can use selected_countries and selected_sources for your plots.


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

    ########################################################################################

    ########################################################################################

    ########################################################################################

  
    # Calculate total number of articles for each country
    total_articles = country_article_counts_df.groupby('country').sum().reset_index()
    total_articles.columns = ['country', 'total_articles']
    # Filter data based on selected countries
    # if len(selected_countries) >0:
    #     for country in selected_countries:
    #         index = country.index(" ")
    #         selected_countries[selected_countries.index(country)]=country[:index]
    filtered_df = aggregated_df.loc[selected_countries]
        # Reset the index to make 'country' a column
    filtered_df = filtered_df.reset_index()

    # Merge this with your DataFrame
    filtered_df = pd.merge(filtered_df, total_articles, on='country', how='left')
    filtered_df = filtered_df.drop_duplicates()
    # Melt the DataFrame to have a separate row for each framing
    melted_df = filtered_df.melt(id_vars=['country', 'total_articles'], var_name='Framing', value_name='Percentage')

    # Calculate number of articles in each framing
    melted_df['number_of_articles'] = (melted_df['Percentage']) * melted_df['total_articles']
    melted_df['number_of_articles'] = np.ceil(melted_df['number_of_articles'])  # round to nearest whole number

    # Convert 'Percentage' from proportion to percentage
    melted_df['Percentage'] = melted_df['Percentage'] * 100

    # Sort the data by country and percentage (in descending order)
    # grouped_df = melted_df.sort_values(by=['country', 'Percentage'], ascending=[True, False])

    melted_df.sort_values(by=['total_articles', 'Percentage'], ascending=[True, False], inplace=True)

    # Ensure unique Framing labels
    melted_df['Framing'] = melted_df['Framing'].astype('category')
    melted_df['Framing'] = melted_df['Framing'].str.replace('_', ' ')

 
    
        # Combine country and total articles
    melted_df['country'] = melted_df['country'] + " (" + melted_df['total_articles'].astype(str) + ")"

    # Sort the data by total articles (in descending order)
    # melted_df = melted_df.sort_values(by=['total_articles'], ascending=True)
    frames_colors = {
        'Economic': '#6941BF',
        'External regulation and reputation': '#0468BF',
        'Capacity and resources': '#8FCEFF',
        'Political': '#FFD16A',
        'Security_and_defense': '#F22E2E',
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
    
    # Plotting the graph using Plotly Express
    # fig = px.bar(melted_df, x='Percentage', y='country', color='Framing', orientation='h', 
    #             color_discrete_map=frames_colors,
    #             title="Distribution of Framings by Country",
    #             hover_data={'number_of_articles': True},
    #             labels={'number_of_articles': 'Number of articles in framing'})

    # # Add axes lines and set x-axis range to [0, 100]
    # fig.update_xaxes(showline=True, linewidth=2, linecolor='black', range=[0, 100])
    # fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    # Increase the size of the figure
    # fig.update_layout(height=500, width=900)  # Adjust the height and width values as per your requirement

    # st.plotly_chart(fig, use_container_width=True)


    # Assuming melted_df is your dataframe with the data

    # Create the figure using Plotly Express
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
    # fig = px.bar(melted_df, 
    #          x='Percentage', 
    #          y='country', 
    #          color='Framing', 
    #          orientation='h',
    #          color_discrete_map=frames_colors,
    #          title="Distribution of Frames by Country",
    #          custom_data=['Framing', 'Percentage', 'number_of_articles'])

    # fig.update_traces(
    # hovertemplate="%{customdata[0]}: %{customdata[1]:.2f}%  (%{customdata[2]:,} times)<extra></extra>")

    # fig.update_layout(
    #     height=500, 
    #     width=900,
    #     xaxis_range=[0, 100],
    #     xaxis_title="Percentage",
    #     yaxis_title="Country"
    # )
    # st.plotly_chart(fig, use_container_width=True)
    # fig = px.bar(melted_df, 
    #          x='Percentage', 
    #          y='country', 
    #          color='Framing', 
    #          orientation='h',
    #          color_discrete_map=frames_colors,
    #          title="Distribution of Frames by Country",
    #          custom_data=['Framing', 'Percentage', 'number_of_articles', melted_df['Framing'].map(framing_explanations)])

    # fig.update_traces(
    # hovertemplate="<b>%{customdata[0]}</b>: %{customdata[1]:.2f}% (%{customdata[2]:,} times)<br><br>" +
    #               "<i>%{customdata[3]}</i><extra></extra>"
    # )

    # fig.update_layout(
    #     height=500, 
    #     width=900,
    #     xaxis_range=[0, 100],
    #     xaxis_title="Percentage",
    #     yaxis_title="Country"
    # )

    # st.plotly_chart(fig, use_container_width=True)
    fig = px.bar(melted_df, 
             x='Percentage', 
             y='country', 
             color='Framing', 
             orientation='h',
             color_discrete_map=frames_colors,
             title="Distribution of Frames by Country",
             custom_data=['Framing', 'Percentage', 'number_of_articles'])

    # Update hover template for the bars
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b>: %{customdata[1]:.2f}% (%{customdata[2]:,} times)<br><br>" +
                    "<i>%{customdata[0]|" + 
                    "|".join(f"{k}={framing_explanations[k]}" for k in framing_explanations) + 
                    "}</i><extra></extra>"
    )

    # Create a list to hold legend items
    legend_traces = []

    # Create invisible scatter plots for each framing category to use as legend items
    for frame, color in frames_colors.items():
        legend_traces.append(
            go.Scatter(
                x=[None], y=[None],  # No data, just for legend
                mode='markers',
                marker=dict(size=10, color=color),
                name=frame,
                legendgroup=frame,
                showlegend=True,
                hoverinfo='name',
                hovertemplate=f"<b>{frame}</b><br><br><i>{framing_explanations.get(frame, 'No explanation available')}</i><extra></extra>"
            )
        )

    # Add the legend traces to the figure
    for trace in legend_traces:
        fig.add_trace(trace)

    # Update layout
    fig.update_layout(
        height=500, 
        width=900,
        xaxis_range=[0, 100],
        xaxis_title="Percentage",
        yaxis_title="Country",
        legend_title="Framing",
        legend=dict(
            itemsizing='constant',
            itemwidth=30
        )
    )

    # Display the figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    ########################################################################################

    # Calculate total number of articles for each country
    total_articles = country_article_counts_df_subtask3.groupby('country').sum().reset_index()
    total_articles.columns = ['country', 'total_articles']

    # Filter data based on selected countries
    filtered_df = aggregated_df_subtask3.loc[selected_countries]
    # Add a column for the total frequency of persuasion techniques for each country
    filtered_df['total_frequency'] = filtered_df.sum(axis=1)

    # Reset the index to make 'country' a column
    filtered_df = filtered_df.reset_index()
    # Merge this with your DataFrame
    filtered_df = pd.merge(filtered_df, total_articles, on='country', how='left')
    filtered_df = filtered_df.drop_duplicates()

    # Melt the DataFrame to have a separate row for each framing
    melted_df = filtered_df.melt(id_vars=['country', 'total_articles', 'total_frequency'], var_name='Framing', value_name='Frequency')
    # Convert 'Frequency' to integer
    melted_df['Frequency'] = melted_df['Frequency'].astype(int)

    # Remove 'labels.' prefix from the 'Framing' column
    melted_df['Framing'] = melted_df['Framing'].str.replace('labels.', '')

    # Rename 'Framing' to 'Persuasion Techniques'
    melted_df.rename(columns={'Framing': 'Persuasion Techniques'}, inplace=True)
    melted_df['Persuasion Techniques'] = melted_df['Persuasion Techniques'].str.replace('_', ' ')
    melted_df['Persuasion Techniques'] = melted_df['Persuasion Techniques'].str.replace('-', ' ')
    

    # Create a new column for the percentage
    melted_df['Percentage'] = (melted_df['Frequency'] / melted_df['total_frequency']) * 100

    # Sort the data by country and frequency (in descending order)
    melted_df.sort_values(by=['country', 'Percentage'], ascending=[True, False], inplace=True)

    # Combine country and total articles
    melted_df['country'] = melted_df['country'] + " (" + melted_df['total_articles'].astype(str) + ")"

    # Sort the data by total articles (in descending order)
    melted_df.sort_values(by=['total_articles', 'Percentage'], ascending=[True, False], inplace=True)

    color_mapping = {
        'Loaded Language': '#FFD16A', #6941BF
        'Obfuscation Vagueness Confusion': '#0468BF',
        'Conversation Killer': '#8FCEFF',
        'Appeal to Time':'#6941BF',
        'Whataboutism': '#a9becf',
        'Red Herring':'#80F29D' ,
        'Straw Man': '#FFABAB',
        'Causal Oversimplification':'#9f7fe3' ,
        'Appeal to Values':'#b1cce3' ,
        'Appeal to Popularity': '#b39696',
        'Appeal to Hypocrisy': '#2f8dd5',
        'Appeal to Authority': '#bd7373',
        'Consequential Oversimplification': '#d4cdcd',
        'False Dilemma No Choice': '#5c9c6c',
        'Repetition': '#945454',
        'Slogans': '#af9cd6',
        'Doubt': '#edaa13',
        'Exaggeration Minimisation': '#958ca8',
        'Name Calling Labeling': '#77abd9',
        'Flag Waving': '#F22E2E', # 
        'Appeal to Fear Prejudice': '#9ad6ac',
        'Guilt by Association': '#6b0c0c',
        'Questioning the Reputation': '#ffdd91',
    }

    # Plotting the graph using Plotly Express
    # fig = px.bar(melted_df, x='Percentage', y='country', color='Persuasion Techniques', orientation='h', 
    #                 color_discrete_map = color_mapping,
    #                 title="Distribution of Persuasion Techniques by Country",
    #                 hover_data={
    #                     'Frequency': True,
    #                     'Percentage': ':.2f'  # Format as float with 2 decimal places
    #                 },
    #                 labels={
    #                     'Percentage': 'Percentage of Persuasion Technique',
    #                     'Frequency': 'Frequency of Persuasion Technique'
    #                 })

    # # Add axes lines
    # fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    # fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    # fig.update_layout(height=700, width=900) 
    # st.plotly_chart(fig, use_container_width=True)
    fig = px.bar(melted_df, 
             x='Percentage', 
             y='country', 
             color='Persuasion Techniques', 
             orientation='h',
             color_discrete_map=color_mapping,
             title="Distribution of Persuasion Techniques by Country",
             custom_data=['Persuasion Techniques', 'Percentage', 'Frequency'])

    fig.update_traces(
        hovertemplate="%{customdata[0]}: %{customdata[1]:.2f}%  (%{customdata[2]:,} times)<extra></extra>"
    )

    fig.update_layout(
        height=700, 
        width=900,
        xaxis_title="Percentage of Persuasion Technique",
        yaxis_title="Country",
        xaxis=dict(showline=True, linewidth=2, linecolor='black'),
        yaxis=dict(showline=True, linewidth=2, linecolor='black')
    )

    st.plotly_chart(fig, use_container_width=True)
    ###########################################################################################

    # Get list of unique countries
    countries = country_article_counts_df.index.get_level_values(0).unique()

    # Create buttons for selection
    option = st.selectbox('Choose an option', ('Aggregate', 'Compare'))

    # Add title and labels
    if option == 'Aggregate':
        fig = go.Figure(data=go.Scatter(x=article_counts_df.index, 
                                        y=article_counts_df['number_of_articles'],
                                        mode='lines'))
        fig.update_layout(title='Number of Articles Over Time (Aggregate)',
                        xaxis_title='Date',
                        yaxis_title='Number of Articles')

    else:
        # Filter the dataframe for selected countries
        filtered_df = country_article_counts_df.loc[selected_countries]
        
        fig = go.Figure()
        
        for country in selected_countries:
            country_df = filtered_df.loc[country]
            # Check if returned object is a Series or DataFrame
            if isinstance(country_df, pd.Series):
                x_data = [country_df.name]  # When it's a Series, name attribute gives the index
                y_data = [country_df.values[0]]
            else:
                x_data = country_df.index
                y_data = country_df['number_of_articles']
            fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=country))


        fig.update_layout(title='Number of Articles Over Time (Compare)',
                        xaxis_title='Date',
                        yaxis_title='Number of Articles')

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)


    ########################################################################################

    CONTINENT_MAP = {
        'AF': 'Africa',
        'AS': 'Asia',
        'EU': 'Europe',
        'NA': 'North America',
        'SA': 'South America',
        'OC': 'Oceania',
        'AN': 'Antarctica'
    }

    def get_region(country):
        try:
            country_code = country_name_to_country_alpha2(country)
            continent_code = country_alpha2_to_continent_code(country_code)
            return CONTINENT_MAP[continent_code]
        except KeyError:
            return "Unknown"

    st.title('News Analysis Dashboard')
    st.markdown("""
    Distribution of Framings in the Articles across Countries and Regions
    """)

    # Reset index of the dataframe
    aggregated_df.reset_index(level=0, inplace=True)

    # Convert wide data to long data
    long_format_df = pd.melt(aggregated_df, id_vars=['country'], var_name='framings', value_name='percentage')

    # Create new region column
    long_format_df['region'] = long_format_df['country'].apply(get_region)

    # Filter dataframe by selected countries
    filtered_df = long_format_df[long_format_df['country'].isin(selected_countries)]

    # Ensure unique Framing labels
    filtered_df['framings'] = filtered_df['framings'].astype('category')
    filtered_df['framings'] = filtered_df['framings'].str.replace('_', ' ')
    filtered_df['framings'] = filtered_df['framings'].str.replace('-', ' ')

    fig = px.sunburst(filtered_df, path=['region', 'country',  'framings'], values='percentage', color='percentage')

    fig.update_layout(width=1200, height=1000)

    st.plotly_chart(fig)


elif st.session_state.page  == "Framings: Countries and Sources":
    
    # option = 'Russo-Ukrainian War'
    # option = st.selectbox('Choose a topic', ('Climate Change', 'Russo-Ukrainian War', 'Combined'), index=1)

    #####loading data ######
    # if option == 'Russo-Ukrainian War':
    #     media_agg = RU_media_agg
    #     # print(media_agg)
    #     country_to_media = RU_country_to_media
    #     # print(country_to_media)
    #     aggregated_df = RU_aggregated_df
    #     # print(aggregated_df)
    #     article_counts_df = RU_article_counts_df
    #     # print(article_counts_df)
    #     country_article_counts_df = RU_country_article_counts_df
    #     # print(country_article_counts_df)

    #     media_agg_subtask3 = RU_media_agg_subtask3
    #     country_to_media_subtask3 = RU_country_to_media_subtask3
    #     aggregated_df_subtask3 = RU_aggregated_df_subtask3
    #     article_counts_df_subtask3 = RU_article_counts_df_subtask3
    #     country_article_counts_df_subtask3 = RU_country_article_counts_df_subtask3
    #     source_article_counts = RU_source_article_counts
    #     # print(source_article_counts)
    #     source_article_counts_subtask3 = RU_source_article_counts_subtask3
    # elif option == 'Climate Change':
    #     media_agg = CH_media_agg
    #     # print(media_agg)
    #     country_to_media = CH_country_to_media
    #     # print(country_to_media)
    #     aggregated_df = CH_aggregated_df
    #     # print(aggregated_df)
    #     article_counts_df = CH_article_counts_df
    #     # print(article_counts_df)
    #     country_article_counts_df = CH_country_article_counts_df
    #     # print(country_article_counts_df)

    #     media_agg_subtask3 = CH_media_agg_subtask3
    #     country_to_media_subtask3 = CH_country_to_media_subtask3
    #     aggregated_df_subtask3 = CH_aggregated_df_subtask3
    #     article_counts_df_subtask3 = CH_article_counts_df_subtask3
    #     country_article_counts_df_subtask3 = CH_country_article_counts_df_subtask3

    #     source_article_counts = CH_source_article_counts
    #     # print(source_article_counts)
    #     source_article_counts_subtask3 = CH_source_article_counts_subtask3
        
    # elif option == 'Combined':
    media_agg = both_media_agg
    # print(media_agg)
    country_to_media = both_country_to_media
    # print(country_to_media)
    aggregated_df = both_aggregated_df
    # print(aggregated_df)
    article_counts_df = both_article_counts_df
    # print(article_counts_df)
    country_article_counts_df = both_country_article_counts_df
    # print(country_article_counts_df)

    media_agg_subtask3 = both_media_agg_subtask3
    country_to_media_subtask3 = both_country_to_media_subtask3
    aggregated_df_subtask3 = both_aggregated_df_subtask3
    article_counts_df_subtask3 = both_article_counts_df_subtask3
    country_article_counts_df_subtask3 = both_country_article_counts_df_subtask3

    source_article_counts = both_source_article_counts
    # print(source_article_counts)
    source_article_counts_subtask3 = both_source_article_counts_subtask3        
    
    ###########################

        
    # Number of countries
    num_countries = len(country_to_media)
    # Number of articles
    num_articles = article_counts_df['number_of_articles'].sum()

    # Number of media sources
    num_media_sources = len(media_agg)

    # Create columns
    col1, col2, col3 = st.columns(3)

    # Display statistics in each column within a box
    with col1:
        st.markdown(f"""
            <div style="background-color: #f5f3f6; border: 1px solid #6c757d; border-radius: 5px; padding: 10px">
                <h4 style="color: #6c757d; text-align: center">Number of countries</h4>
                <p style="text-align: center; font-size: 20px; color: black;">{num_countries}</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="background-color: #f5f3f6; border: 1px solid #6c757d; border-radius: 5px; padding: 10px">
                <h4 style="color: #6c757d; text-align: center">Number of articles</h4>
                <p style="text-align: center; font-size: 20px; color: black;">{num_articles}</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div style="background-color: #f5f3f6; border: 1px solid #6c757d; border-radius: 5px; padding: 10px">
                <h4 style="color: #6c757d; text-align: center">Number of media sources</h4>
                <p style="text-align: center; font-size: 20px; color: black;">{num_media_sources}</p>
            </div>
        """, unsafe_allow_html=True)

    if 'selected_pairs' not in st.session_state:
        st.session_state.selected_pairs = []

# # Calculate total number of articles for each country
#     total_articles = country_article_counts_df.groupby('country').sum().reset_index()
#     total_articles.columns = ['country', 'total_articles']
    
#     # Create a dictionary that maps country names to total number of articles
#     country_to_total_articles = total_articles.set_index('country')['total_articles'].to_dict()
    
#     available_countries = country_to_media['country'].unique().tolist()
# #add the number of articles per country
#     total_articles = country_article_counts_df.groupby('country').sum().reset_index()
#     total_articles.columns = ['country', 'total_articles']
#     countries=country_to_media['country'].unique().tolist()
#     total_articles=  total_articles['total_articles'].tolist()
#     for country in countries:
#         if country in available_countries:
#             row_num=countries.index(country)
#             for i in range(len(available_countries)):
#                 if available_countries[i]==country:
#                     available_countries[i]+=" ("+str(total_articles[row_num])+")"
                    
#     available_countries.sort(key=lambda x: int(x.split(' ')[-1].strip('()')), reverse=True)
#     country = st.selectbox('Select country', available_countries, key='country')
#     country=country[:country.index("(")-1]
#     source_counts_list = country_to_media[country_to_media['country'] == country]['source_frequencies'].values[0]
#     source_counts_list = ast.literal_eval(source_counts_list)
#     selected_sources_dict = {item[0]: item[1] for item in source_counts_list}
#     selected_sources_dict = {k: (v if v is not None else 0) for k, v in selected_sources_dict.items()}
#     selected_sources_dict = {k: v for k, v in selected_sources_dict.items() if v > 0}
    
#     selected_sources = [pair[1] for pair in st.session_state.selected_pairs if pair[0] == country]
#     available_sources = selected_sources_dict.keys()
    
#     # Convert to lower case and create a mapping from lowercase to original
#     available_sources_lower_to_original = {source.lower(): source for source in available_sources}
    
#     source_articles_data = []
#     # Loop through each row in the country_to_media dataframe
#     for index, row in country_to_media.iterrows():
#         # Loop through each source_frequency list in the current row
#         for source_frequency in eval(row['source_frequencies']):
#             # Append a dictionary with the media source and its number of articles to the list
#             source_articles_data.append({'source': source_frequency[0], 'total_articles': source_frequency[1] if source_frequency[1] else 0})
#     # Convert the list of dictionaries into a dataframe
#     source_articles_df = pd.DataFrame(source_articles_data)
#     # Get the list of lowercase sources
#     available_sources_lower = [source.lower() for source in available_sources]
#     #add the total number of articles per source
#     for source in available_sources_lower:
#         if source in available_sources_lower_to_original.keys():
#             row_num = next((index for (index, d) in enumerate(source_articles_data) if d["source"].lower() == source), None)
#             if row_num is not None:
#                 number=source_articles_data[row_num]['total_articles']
#                 available_sources_lower[available_sources_lower.index(source)]=source+" ("+str(number)+")"
#             else:
#                 available_sources_lower[available_sources_lower.index(source)]=source+" (0)"
    
#     # Now sort available_sources_lower based on the total number of articles in descending order
#     available_sources_lower.sort(key=lambda x: int(x.split(' ')[-1].strip('()')), reverse=True)
    
#     source = st.selectbox('Select source', available_sources_lower, key='source')


    # Create a dictionary mapping sources to their correct countries
    correct_countries = {
        'europesun': 'Hungary',
        'menafn': 'Jordan',
        'northernirelandnews': 'Ireland',
        'urdupoint': 'Pakistan',
        'polandsun': 'Poland',
        'dailysabah': 'Turkey'
    }

    # Copy of the original DataFrame
    corrected_df = country_to_media.copy()
    # Iterate over the rows in the dataframe
    for index, row in corrected_df.iterrows():
        # Convert the source_frequencies to a list of tuples
        source_frequencies = literal_eval(row['source_frequencies'])

        # List to store the new source_frequencies for this row
        new_source_frequencies = []

        # Iterate over the source_frequencies
        for source_frequency in source_frequencies:
            source = source_frequency[0].lower()  # Get the source name

            # If the source is in the correct_countries dictionary
            if source in correct_countries:
                # Get the index of the correct country row in corrected_df
                correct_country_index = corrected_df[corrected_df['country'] == correct_countries[source]].index[0]

                # Get the source_frequencies for the correct country
                correct_country_sources = literal_eval(corrected_df.at[correct_country_index, 'source_frequencies'])

                # Add the source_frequency to correct_country_sources
                correct_country_sources.append(source_frequency)

                # Update the source_frequencies for the correct country
                corrected_df.at[correct_country_index, 'source_frequencies'] = str(correct_country_sources)
            else:
                # Add the source_frequency to new_source_frequencies
                new_source_frequencies.append(source_frequency)

        # Update the source_frequencies for this rowF
        corrected_df.at[index, 'source_frequencies'] = str(new_source_frequencies)

    # Append the corrected rows to country_to_media
    country_to_media = corrected_df
    # Calculate total number of articles for each country
    print(country_article_counts_df)
    total_articles = country_article_counts_df.groupby('country').sum().reset_index()
    total_articles.columns = ['country', 'total_articles']
    
    # Create a dictionary that maps country names to total number of articles
    country_to_total_articles = total_articles.set_index('country')['total_articles'].to_dict()
    
    available_countries = country_to_media['country'].unique().tolist()
    
    # Add the number of articles per country
    total_articles = country_article_counts_df.groupby('country').sum().reset_index()
    total_articles.columns = ['country', 'total_articles']
    countries = country_to_media['country'].unique().tolist()
    total_articles = total_articles['total_articles'].tolist()
    
    for country in countries:
        if country in available_countries:
            row_num = countries.index(country)
            for i in range(len(available_countries)):
                if available_countries[i] == country:
                    available_countries[i] += " (" + str(total_articles[row_num]) + ")"
    
    available_countries.sort(key=lambda x: int(x.split(' ')[-1].strip('()')), reverse=True)
    country = st.selectbox('Select country', available_countries, key='country')
    country = country[:country.index("(") - 1]
    source_counts_list = country_to_media[country_to_media['country'] == country]['source_frequencies'].values[0]
    # print(source_counts_list)
    source_counts_list = ast.literal_eval(source_counts_list)
    selected_sources_dict = {item[0]: item[1] for item in source_counts_list}
    selected_sources_dict = {k: (v if v is not None else 0) for k, v in selected_sources_dict.items()}
    selected_sources_dict = {k: v for k, v in selected_sources_dict.items() if v > 0}
    
    selected_sources = [pair[1] for pair in st.session_state.selected_pairs if pair[0] == country]
    available_sources = selected_sources_dict.keys()
    
    # Convert to lower case and create a mapping from lowercase to original
    available_sources_lower_to_original = {source.lower(): source for source in available_sources}
    
    source_articles_data = []
    # Loop through each row in the country_to_media dataframe
    for index, row in country_to_media.iterrows():
        # Loop through each source_frequency list in the current row
        for source_frequency in eval(row['source_frequencies']):
            # Append a dictionary with the media source and its number of articles to the list
            source_articles_data.append({'source': source_frequency[0], 'total_articles': source_frequency[1] if source_frequency[1] else 0})
            
    # Convert the list of dictionaries into a dataframe
    source_articles_df = pd.DataFrame(source_articles_data)
    
    # Get the list of lowercase sources
    available_sources_lower = [source.lower() for source in available_sources]
    
    # Add the total number of articles per source
    for source in available_sources_lower:
        if source in available_sources_lower_to_original.keys():
            row_num = next((index for (index, d) in enumerate(source_articles_data) if d["source"].lower() == source), None)
            if row_num is not None:
                number = source_articles_data[row_num]['total_articles']
                available_sources_lower[available_sources_lower.index(source)] = source + " (" + str(number) + ")"
            else:
                available_sources_lower[available_sources_lower.index(source)] = source + " (0)"
    
    # Now sort available_sources_lower based on the total number of articles in descending order
    available_sources_lower.sort(key=lambda x: int(x.split(' ')[-1].strip('()')), reverse=True)
    
    source = st.selectbox('Select source', available_sources_lower, key='source')




    ###############################################################################

    if st.button('Add selection'):
        # Retrieve the original casing version of the source
        index=source.index(' ')
        original_case_source = available_sources_lower_to_original[source[:index]]
        if ((country, original_case_source) not in st.session_state.selected_pairs):
            st.session_state.selected_pairs.append((country, original_case_source))

    if st.button('Remove last selection') and st.session_state.selected_pairs:
        st.session_state.selected_pairs.pop()

    selected_countries = [pair[0] for pair in st.session_state.selected_pairs]
    selected_sources = [pair[1] for pair in st.session_state.selected_pairs]


    # rest of your code

    # Now you can use selected_countries and selected_sources for your plots.


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

    ########################################################################################
    # First, create a list to hold all the media source data
    source_articles_data = []

    # Loop through each row in the country_to_media dataframe
    for index, row in country_to_media.iterrows():
        # Loop through each source_frequency list in the current row
        for source_frequency in eval(row['source_frequencies']):
            # Append a dictionary with the media source and its number of articles to the list
            source_articles_data.append({'source': source_frequency[0], 'total_articles': source_frequency[1]})

    # Convert the list of dictionaries into a dataframe
    source_articles_df = pd.DataFrame(source_articles_data)
    # Calculate total number of articles for each source
    total_articles = source_articles_df.groupby('source').sum().reset_index()

    # Filter media_agg DataFrame for the selected source
    filtered_df = media_agg.loc[selected_sources]

    # Merge this with your DataFrame
    filtered_df = pd.merge(filtered_df, total_articles, on='source', how='left')

    # Melt the DataFrame to have a separate row for each framing
    melted_df = filtered_df.melt(id_vars=['source', 'total_articles'], var_name='Framing', value_name='Percentage')

    # Calculate number of articles in each framing
    melted_df['number_of_articles'] = (melted_df['Percentage']) * melted_df['total_articles']
    melted_df['number_of_articles'] = np.ceil(melted_df['number_of_articles'])  # round to nearest whole number

    # Convert 'Percentage' from proportion to percentage
    melted_df['Percentage'] = melted_df['Percentage'] * 100

    # Sort the data by country and percentage (in descending order)
    melted_df = melted_df.sort_values(by=['source', 'Percentage'], ascending=[False, False])

    # Ensure unique Framing labels
    melted_df['Framing'] = melted_df['Framing'].astype('category')
    melted_df['Framing'] = melted_df['Framing'].str.replace('_', ' ')


                # Combine country and total articles
    melted_df['source'] = melted_df['source'] + " (" + melted_df['total_articles'].astype(str) + ")"


    melted_df.sort_values(by=['total_articles', 'Percentage'], ascending=[True, False], inplace=True)

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

    # Plotting the graph using Plotly Express
    # fig = px.bar(melted_df, x='Percentage', y='source', color='Framing', orientation='h', 
    #             color_discrete_map=frames_colors,  # use the color mapping
    #             title="Distribution of Framings by Source",
    #             hover_data={
    #                 'number_of_articles': True,
    #                 'total_articles': True
    #             },
    #             labels={
    #                 'number_of_articles': 'Number of articles in framing',
    #                 'total_articles': 'Total articles in source'
    #             })

    # # Add axes lines and set x-axis range to [0, 100]
    # fig.update_xaxes(showline=True, linewidth=2, linecolor='black', range=[0, 100])
    # fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    # fig.update_layout(height=500, width=900) 
    # st.plotly_chart(fig, use_container_width=True)
    


    fig = px.bar(melted_df, 
                x='Percentage', 
                y='source', 
                color='Framing', 
                orientation='h',
                color_discrete_map=frames_colors,
                title="Distribution of Framings by Source",
                custom_data=['Framing', 'Percentage', 'number_of_articles'])

    fig.update_traces(
        hovertemplate="%{customdata[0]}: %{customdata[1]:.2f}%  (%{customdata[2]:,} articles)<extra></extra>"
    )

    fig.update_layout(
        height=500, 
        width=900,
        xaxis_range=[0, 100],
        xaxis_title="Percentage",
        yaxis_title="Source",
        xaxis=dict(showline=True, linewidth=2, linecolor='black'),
        yaxis=dict(showline=True, linewidth=2, linecolor='black')
    )

    st.plotly_chart(fig, use_container_width=True)



    ########################################################################################

    ########################################################################################


    # Calculate total number of articles for each country
    total_articles = country_article_counts_df.groupby('country').sum().reset_index()
    total_articles.columns = ['country', 'total_articles']
# Convert selected_countries to set and back to list to remove duplicates
    selected_countries = list(set(selected_countries))
    # Filter data based on selected countries
    filtered_df = aggregated_df.loc[selected_countries]
    # Reset the index to make 'country' a column
    filtered_df = filtered_df.reset_index()

    # Merge this with your DataFrame
    filtered_df = pd.merge(filtered_df, total_articles, on='country', how='left')
    filtered_df = filtered_df.drop_duplicates()

    # Melt the DataFrame to have a separate row for each framing
    melted_df = filtered_df.melt(id_vars=['country', 'total_articles'], var_name='Framing', value_name='Percentage')

    # Calculate number of articles in each framing
    melted_df['number_of_articles'] = (melted_df['Percentage']) * melted_df['total_articles']
    melted_df['number_of_articles'] = np.ceil(melted_df['number_of_articles'])  # round to nearest whole number

    # Convert 'Percentage' from proportion to percentage
    melted_df['Percentage'] = melted_df['Percentage'] * 100

    # Sort the data by country and percentage (in descending order)
    melted_df = melted_df.sort_values(by=['country', 'Percentage'], ascending=[True, False])

    # Ensure unique Framing labels
    melted_df['Framing'] = melted_df['Framing'].astype('category')
    melted_df['Framing'] = melted_df['Framing'].str.replace('_', ' ')
    melted_df['Framing'] = melted_df['Framing'].str.replace('-', ' ')

                # Combine country and total articles
    melted_df['country'] = melted_df['country'] + " (" + melted_df['total_articles'].astype(str) + ")"

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

    # Plotting the graph using Plotly Express
    # fig = px.bar(melted_df, x='Percentage', y='country', color='Framing', orientation='h', 
    #             color_discrete_map=frames_colors,  # use the color mapping
    #             title="Distribution of Framings by Country",
    #             hover_data={
    #                 'number_of_articles': True,
    #                 'total_articles': True
    #             },
    #             labels={
    #                 'number_of_articles': 'Number of articles in framing',
    #                 'total_articles': 'Total articles in country'
    #             })

    # # Add axes lines and set x-axis range to [0, 100]
    # fig.update_xaxes(showline=True, linewidth=2, linecolor='black', range=[0, 100])
    # fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    # fig.update_layout(height=500, width=900) 
    # st.plotly_chart(fig, use_container_width=True)
    
    fig = px.bar(melted_df, 
                x='Percentage', 
                y='country', 
                color='Framing', 
                orientation='h',
                color_discrete_map=frames_colors,
                title="Distribution of Framings by Country",
                custom_data=['Framing', 'Percentage', 'number_of_articles'])

    fig.update_traces(
        hovertemplate="%{customdata[0]}: %{customdata[1]:.2f}%  (%{customdata[2]:,} times)<extra></extra>"
    )

    fig.update_layout(
        height=500, 
        width=900,
        xaxis_range=[0, 100],
        xaxis_title="Percentage",
        yaxis_title="Country"
    )

    st.plotly_chart(fig, use_container_width=True)
 ########################################################################################


    # Get list of unique countries
    countries = country_article_counts_df_subtask3.index.get_level_values(0).unique()

    # Create buttons for selection
    option = st.selectbox('Choose an option', ('Compare', 'Aggregate'))

    # Add title and labels
    if option == 'Aggregate':
        fig = go.Figure(data=go.Scatter(x=article_counts_df_subtask3.index, 
                                        y=article_counts_df_subtask3['number_of_articles'],
                                        mode='lines'))
        fig.update_layout(title='Number of Articles Over Time (Aggregate) by source',
                        xaxis_title='Date',
                        yaxis_title='Number of Articles')

    else:
        # Convert selected_countries to set and back to list to remove duplicates
        selected_sources = list(set(selected_sources))
        # Filter the dataframe for selected countries
        filtered_df = source_article_counts_subtask3.loc[selected_sources]
        
        fig = go.Figure()
        
        for source in selected_sources:
            source_df = filtered_df.loc[source]
            # Check if returned object is a Series or DataFrame
            if isinstance(source_df, pd.Series):
                x_data = [source_df.name]  # When it's a Series, name attribute gives the index
                y_data = [source_df.values[0]]
            else:
                x_data = source_df.index
                y_data = source_df['number_of_articles']
            fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=source))


        fig.update_layout(title='Number of Articles Over Time (Compare) by source',
                        xaxis_title='Date',
                        yaxis_title='Number of Articles')

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)


    ########################################################################################

        # Add title and labels
    if option == 'Aggregate':
        fig = go.Figure(data=go.Scatter(x=article_counts_df_subtask3.index, 
                                        y=article_counts_df_subtask3['number_of_articles'],
                                        mode='lines'))
        fig.update_layout(title='Number of Articles Over Time (Aggregate) by country',
                        xaxis_title='Date',
                        yaxis_title='Number of Articles')

    else:
        # Convert selected_countries to set and back to list to remove duplicates
        selected_countries = list(set(selected_countries))
        # Filter the dataframe for selected countries
        filtered_df = country_article_counts_df_subtask3.loc[selected_countries]
        
        fig = go.Figure()
        
        for country in selected_countries:
            country_df = filtered_df.loc[country]
            # Check if returned object is a Series or DataFrame
            if isinstance(country_df, pd.Series):
                x_data = [country_df.name]  # When it's a Series, name attribute gives the index
                y_data = [country_df.values[0]]
            else:
                x_data = country_df.index
                y_data = country_df['number_of_articles']
            fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=country))


        fig.update_layout(title='Number of Articles Over Time (Compare) by country',
                        xaxis_title='Date',
                        yaxis_title='Number of Articles')

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

    ########################################################################################


    CONTINENT_MAP = {
        'AF': 'Africa',
        'AS': 'Asia',
        'EU': 'Europe',
        'NA': 'North America',
        'SA': 'South America',
        'OC': 'Oceania',
        'AN': 'Antarctica'
    }

    def get_region(country):
        try:
            country_code = country_name_to_country_alpha2(country)
            continent_code = country_alpha2_to_continent_code(country_code)
            return CONTINENT_MAP[continent_code]
        except KeyError:
            return "Unknown"

    st.title('News Analysis Dashboard')
    st.markdown("""
    Distribution of Framings in the Articles across Countries and Regions
    """)

    # Reset index of the dataframe
    aggregated_df.reset_index(level=0, inplace=True)

    # Convert wide data to long data
    long_format_df = pd.melt(aggregated_df, id_vars=['country'], var_name='framings', value_name='percentage')

    # Create new region column
    long_format_df['region'] = long_format_df['country'].apply(get_region)

    # Filter dataframe by selected countries
    filtered_df = long_format_df[long_format_df['country'].isin(selected_countries)]

        # Ensure unique Framing labels
    filtered_df['framings'] = filtered_df['framings'].astype('category')
    filtered_df['framings'] = filtered_df['framings'].str.replace('_', ' ')
    filtered_df['framings'] = filtered_df['framings'].str.replace('-', ' ')

    fig = px.sunburst(filtered_df, path=['region', 'country',  'framings'], values='percentage', color='percentage')

    fig.update_layout(width=1200, height=1000)

    st.plotly_chart(fig)


#######################################################################################################################

    # import pandas as pd
    # import plotly.express as px
    # import ast

    # # Convert source_frequencies from string to list of lists
    # country_to_media['source_frequencies'] = country_to_media['source_frequencies'].apply(ast.literal_eval)

    # # Flatten the source_frequencies to have one row per source
    # country_source_df = pd.DataFrame([(row['country'], pair[0]) for _, row in country_to_media.iterrows() for pair in row['source_frequencies']], columns=['country', 'source'])

    # # Join with media_agg
    # merged_df = pd.merge(country_source_df, media_agg, left_on='source', right_index=True, how='inner')

    # # Convert wide data to long data
    # long_format_df = pd.melt(merged_df, id_vars=['country', 'source'], var_name='framings', value_name='percentage')

    # st.title('News Analysis Dashboard')
    # st.markdown("""
    # Distribution of Framings in the Articles across Countries and Sources
    # """)

    # # Filter dataframe by selected countries
    # filtered_df = long_format_df[long_format_df['country'].isin(selected_countries)]

    # fig = px.sunburst(filtered_df, path=['country', 'source', 'framings'], values='percentage', color='percentage')

    # fig.update_layout(width=1200, height=1000)

    # st.plotly_chart(fig)



#######################################################################################################################


elif st.session_state.page  == "Persuasion Techniques: Fine-Grained Propaganda":
    # option = 'Russo-Ukrainian War'
    # option = st.selectbox('Choose a topic', ('Climate Change', 'Russo-Ukrainian War', 'Combined'), index=1)

    # #####loading data ######
    # if option == 'Russo-Ukrainian War':
    #     media_agg = RU_media_agg
    #     # print(media_agg)
    #     country_to_media = RU_country_to_media
    #     # print(country_to_media)
    #     aggregated_df = RU_aggregated_df
    #     # print(aggregated_df)
    #     article_counts_df = RU_article_counts_df
    #     # print(article_counts_df)
    #     country_article_counts_df = RU_country_article_counts_df
    #     # print(country_article_counts_df)

    #     media_agg_subtask3 = RU_media_agg_subtask3
    #     country_to_media_subtask3 = RU_country_to_media_subtask3
    #     aggregated_df_subtask3 = RU_aggregated_df_subtask3
    #     article_counts_df_subtask3 = RU_article_counts_df_subtask3
    #     country_article_counts_df_subtask3 = RU_country_article_counts_df_subtask3
    #     source_article_counts = RU_source_article_counts
    #     # print(source_article_counts)
    #     source_article_counts_subtask3 = RU_source_article_counts_subtask3
    # elif option == 'Climate Change':
    #     media_agg = CH_media_agg
    #     # print(media_agg)
    #     country_to_media = CH_country_to_media
    #     # print(country_to_media)
    #     aggregated_df = CH_aggregated_df
    #     # print(aggregated_df)
    #     article_counts_df = CH_article_counts_df
    #     # print(article_counts_df)
    #     country_article_counts_df = CH_country_article_counts_df
    #     # print(country_article_counts_df)

    #     media_agg_subtask3 = CH_media_agg_subtask3
    #     country_to_media_subtask3 = CH_country_to_media_subtask3
    #     aggregated_df_subtask3 = CH_aggregated_df_subtask3
    #     article_counts_df_subtask3 = CH_article_counts_df_subtask3
    #     country_article_counts_df_subtask3 = CH_country_article_counts_df_subtask3

    #     source_article_counts = CH_source_article_counts
    #     # print(source_article_counts)
    #     source_article_counts_subtask3 = CH_source_article_counts_subtask3
        
    # elif option == 'Combined':
    media_agg = both_media_agg
    # print(media_agg)
    country_to_media = both_country_to_media
    # print(country_to_media)
    aggregated_df = both_aggregated_df
    # print(aggregated_df)
    article_counts_df = both_article_counts_df
    # print(article_counts_df)
    country_article_counts_df = both_country_article_counts_df
    # print(country_article_counts_df)

    media_agg_subtask3 = both_media_agg_subtask3
    country_to_media_subtask3 = both_country_to_media_subtask3
    aggregated_df_subtask3 = both_aggregated_df_subtask3
    article_counts_df_subtask3 = both_article_counts_df_subtask3
    country_article_counts_df_subtask3 = both_country_article_counts_df_subtask3

    source_article_counts = both_source_article_counts
    # print(source_article_counts)
    source_article_counts_subtask3 = both_source_article_counts_subtask3        
    
    ###########################

     # Number of countries
    num_countries = len(country_to_media)
    # Number of articles
    num_articles = article_counts_df['number_of_articles'].sum()

    # Number of media sources
    num_media_sources = len(media_agg)

    # Create columns
    col1, col2, col3 = st.columns(3)

    # Display statistics in each column within a box
    with col1:
        st.markdown(f"""
            <div style="background-color: #f5f3f6; border: 1px solid #6c757d; border-radius: 5px; padding: 10px">
                <h4 style="color: #6c757d; text-align: center">Number of countries</h4>
                <p style="text-align: center; font-size: 20px; color: black;">{num_countries}</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="background-color: #f5f3f6; border: 1px solid #6c757d; border-radius: 5px; padding: 10px">
                <h4 style="color: #6c757d; text-align: center">Number of articles</h4>
                <p style="text-align: center; font-size: 20px; color: black;">{num_articles}</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div style="background-color: #f5f3f6; border: 1px solid #6c757d; border-radius: 5px; padding: 10px">
                <h4 style="color: #6c757d; text-align: center">Number of media sources</h4>
                <p style="text-align: center; font-size: 20px; color: black;">{num_media_sources}</p>
            </div>
        """, unsafe_allow_html=True)

    if 'selected_pairs' not in st.session_state:
        st.session_state.selected_pairs = []


    # Create a dictionary mapping sources to their correct countries
    correct_countries = {
        'europesun': 'Hungary',
        'menafn': 'Jordan',
        'northernirelandnews': 'Ireland',
        'urdupoint': 'Pakistan',
        'polandsun': 'Poland',
        'dailysabah': 'Turkey'
    }

    # Copy of the original DataFrame
    corrected_df = country_to_media.copy()
    # Iterate over the rows in the dataframe
    for index, row in corrected_df.iterrows():
        # Convert the source_frequencies to a list of tuples
        source_frequencies = literal_eval(row['source_frequencies'])

        # List to store the new source_frequencies for this row
        new_source_frequencies = []

        # Iterate over the source_frequencies
        for source_frequency in source_frequencies:
            source = source_frequency[0].lower()  # Get the source name

            # If the source is in the correct_countries dictionary
            if source in correct_countries:
                # Get the index of the correct country row in corrected_df
                correct_country_index = corrected_df[corrected_df['country'] == correct_countries[source]].index[0]

                # Get the source_frequencies for the correct country
                correct_country_sources = literal_eval(corrected_df.at[correct_country_index, 'source_frequencies'])

                # Add the source_frequency to correct_country_sources
                correct_country_sources.append(source_frequency)

                # Update the source_frequencies for the correct country
                corrected_df.at[correct_country_index, 'source_frequencies'] = str(correct_country_sources)
            else:
                # Add the source_frequency to new_source_frequencies
                new_source_frequencies.append(source_frequency)

        # Update the source_frequencies for this row
        corrected_df.at[index, 'source_frequencies'] = str(new_source_frequencies)

    # Append the corrected rows to country_to_media
    country_to_media = corrected_df
    # Calculate total number of articles for each country
    print(country_article_counts_df)
    total_articles = country_article_counts_df.groupby('country').sum().reset_index()
    total_articles.columns = ['country', 'total_articles']
    
    # Create a dictionary that maps country names to total number of articles
    country_to_total_articles = total_articles.set_index('country')['total_articles'].to_dict()
    
    available_countries = country_to_media['country'].unique().tolist()
    
    # Add the number of articles per country
    total_articles = country_article_counts_df.groupby('country').sum().reset_index()
    total_articles.columns = ['country', 'total_articles']
    countries = country_to_media['country'].unique().tolist()
    total_articles = total_articles['total_articles'].tolist()
    
    for country in countries:
        if country in available_countries:
            row_num = countries.index(country)
            for i in range(len(available_countries)):
                if available_countries[i] == country:
                    available_countries[i] += " (" + str(total_articles[row_num]) + ")"
    
    available_countries.sort(key=lambda x: int(x.split(' ')[-1].strip('()')), reverse=True)
    country = st.selectbox('Select country', available_countries, key='country')
    country = country[:country.index("(") - 1]
    source_counts_list = country_to_media[country_to_media['country'] == country]['source_frequencies'].values[0]
    # print(source_counts_list)
    source_counts_list = ast.literal_eval(source_counts_list)
    selected_sources_dict = {item[0]: item[1] for item in source_counts_list}
    selected_sources_dict = {k: (v if v is not None else 0) for k, v in selected_sources_dict.items()}
    selected_sources_dict = {k: v for k, v in selected_sources_dict.items() if v > 0}
    
    selected_sources = [pair[1] for pair in st.session_state.selected_pairs if pair[0] == country]
    available_sources = selected_sources_dict.keys()
    
    # Convert to lower case and create a mapping from lowercase to original
    available_sources_lower_to_original = {source.lower(): source for source in available_sources}
    
    source_articles_data = []
    # Loop through each row in the country_to_media dataframe
    for index, row in country_to_media.iterrows():
        # Loop through each source_frequency list in the current row
        for source_frequency in eval(row['source_frequencies']):
            # Append a dictionary with the media source and its number of articles to the list
            source_articles_data.append({'source': source_frequency[0], 'total_articles': source_frequency[1] if source_frequency[1] else 0})
            
    # Convert the list of dictionaries into a dataframe
    source_articles_df = pd.DataFrame(source_articles_data)
    
    # Get the list of lowercase sources
    available_sources_lower = [source.lower() for source in available_sources]
    
    # Add the total number of articles per source
    for source in available_sources_lower:
        if source in available_sources_lower_to_original.keys():
            row_num = next((index for (index, d) in enumerate(source_articles_data) if d["source"].lower() == source), None)
            if row_num is not None:
                number = source_articles_data[row_num]['total_articles']
                available_sources_lower[available_sources_lower.index(source)] = source + " (" + str(number) + ")"
            else:
                available_sources_lower[available_sources_lower.index(source)] = source + " (0)"
    
    # Now sort available_sources_lower based on the total number of articles in descending order
    available_sources_lower.sort(key=lambda x: int(x.split(' ')[-1].strip('()')), reverse=True)
    
    source = st.selectbox('Select source', available_sources_lower, key='source')
    ###############################################################################

    if st.button('Add selection'):
        # Retrieve the original casing version of the source
        index=source.index(' ')
        original_case_source = available_sources_lower_to_original[source[:index]]
        if ((country, original_case_source) not in st.session_state.selected_pairs):
            st.session_state.selected_pairs.append((country, original_case_source))

    if st.button('Remove last selection') and st.session_state.selected_pairs:
        st.session_state.selected_pairs.pop()

    selected_countries = [pair[0] for pair in st.session_state.selected_pairs]
    selected_sources = [pair[1] for pair in st.session_state.selected_pairs]

    # rest of your code


    # Now you can use selected_countries and selected_sources for your plots.


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

    ########################################################################################

    # First, create a list to hold all the media source data
    source_articles_data = []

    # Loop through each row in the country_to_media dataframe
    for index, row in country_to_media_subtask3.iterrows():
        # Loop through each source_frequency list in the current row
        for source_frequency in eval(row['source_frequencies']):
            # Append a dictionary with the media source and its number of articles to the list
            source_articles_data.append({'source': source_frequency[0], 'total_articles': source_frequency[1]})

    # Convert the list of dictionaries into a dataframe
    source_articles_df = pd.DataFrame(source_articles_data)

    # Calculate total number of articles for each source
    total_articles = source_articles_df.groupby('source').sum()
    # Filter media_agg DataFrame for the selected source
    filtered_df = media_agg_subtask3.loc[selected_sources]

    # Add a column for the total frequency of persuasion techniques for each source
    filtered_df['total_frequency'] = filtered_df.sum(axis=1)

    # Merge this with your DataFrame
    filtered_df = pd.merge(filtered_df, total_articles, on='source', how='left')
    # Melt the DataFrame to have a separate row for each framing
    melted_df = filtered_df.reset_index().melt(id_vars=['source', 'total_articles', 'total_frequency'], var_name='Framing', value_name='Frequency')

    # Remove 'labels.' prefix from the 'Framing' column
    melted_df['Framing'] = melted_df['Framing'].str.replace('labels.', '')

    # Rename 'Framing' to 'Persuasion Techniques'
    melted_df.rename(columns={'Framing': 'Persuasion Techniques'}, inplace=True)
    melted_df['Persuasion Techniques'] = melted_df['Persuasion Techniques'].str.replace('_', ' ')
    melted_df['Persuasion Techniques'] = melted_df['Persuasion Techniques'].str.replace('-', ' ')

    # Create a new column for the percentage
    melted_df['Percentage'] = (melted_df['Frequency'] / melted_df['total_frequency']) * 100

    # Sort the DataFrame by 'source' and 'Frequency' in descending order
    melted_df.sort_values(by=['source', 'Percentage'], ascending=[True, False], inplace=True)

            # Combine country and total articles
    melted_df['source'] = melted_df['source'] + " (" + melted_df['total_articles'].astype(str) + ")"

    # Sort the data by total articles (in descending order)
    # melted_df = melted_df.sort_values(by=['total_articles'], ascending=True)

    melted_df.sort_values(by=['total_articles', 'Percentage'], ascending=[True, False], inplace=True)

    color_mapping = {
        'Loaded Language': '#FFD16A', #6941BF
        'Obfuscation Vagueness Confusion': '#0468BF',
        'Conversation Killer': '#8FCEFF',
        'Appeal to Time':'#6941BF',
        'Whataboutism': '#a9becf',
        'Red Herring':'#80F29D' ,
        'Straw Man': '#FFABAB',
        'Causal Oversimplification':'#9f7fe3' ,
        'Appeal to Values':'#b1cce3' ,
        'Appeal to Popularity': '#b39696',
        'Appeal to Hypocrisy': '#2f8dd5',
        'Appeal to Authority': '#bd7373',
        'Consequential Oversimplification': '#d4cdcd',
        'False Dilemma No Choice': '#5c9c6c',
        'Repetition': '#945454',
        'Slogans': '#af9cd6',
        'Doubt': '#edaa13',
        'Exaggeration Minimisation': '#958ca8',
        'Name Calling Labeling': '#77abd9',
        'Flag Waving': '#F22E2E', # 
        'Appeal to Fear Prejudice': '#9ad6ac',
        'Guilt by Association': '#6b0c0c',
        'Questioning the Reputation': '#ffdd91',
    }

    # Plotting the graph using Plotly Express
    # fig = px.bar(melted_df, x='Percentage', y='source', color='Persuasion Techniques', orientation='h', 
    #                 color_discrete_map=color_mapping,  # use the color mapping
    #                 title="Distribution of Persuasion Techniques by Source",
    #                 hover_data={
    #                     'Frequency': True,
    #                     'Percentage': ':.2f'  # Format as float with 2 decimal places
    #                 },
    #                 labels={
    #                     'Percentage': 'Percentage of Persuasion Technique',
    #                     'Frequency': 'Frequency of Persuasion Technique'
    #                 })

    # # Add axes lines
    # fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    # fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    # fig.update_layout(height=700, width=900) 
    # st.plotly_chart(fig, use_container_width=True)
    fig = px.bar(melted_df, 
             x='Percentage', 
             y='source',  # Make sure this matches your column name exactly
             color='Persuasion Techniques', 
             orientation='h',
             color_discrete_map=color_mapping,
             title="Distribution of Persuasion Techniques by Source",
             custom_data=['Persuasion Techniques', 'Percentage', 'Frequency'])

    fig.update_traces(
        hovertemplate="%{customdata[0]}: %{customdata[1]:.2f}%  (%{customdata[2]:,} times)<extra></extra>"
    )

    fig.update_layout(
        height=700, 
        width=900,
        xaxis_title="Percentage of Persuasion Technique",
        yaxis_title="Source",
        xaxis=dict(showline=True, linewidth=2, linecolor='black'),
        yaxis=dict(showline=True, linewidth=2, linecolor='black')
    )

    st.plotly_chart(fig, use_container_width=True)



    ########################################################################################

    ########################################################################################


    # Calculate total number of articles for each country
    total_articles = country_article_counts_df_subtask3.groupby('country').sum().reset_index()
    total_articles.columns = ['country', 'total_articles']

    # Filter data based on selected countries
    filtered_df = aggregated_df_subtask3.loc[selected_countries]
    # Add a column for the total frequency of persuasion techniques for each country
    filtered_df['total_frequency'] = filtered_df.sum(axis=1)

    # Reset the index to make 'country' a column
    filtered_df = filtered_df.reset_index()
    # Merge this with your DataFrame
    filtered_df = pd.merge(filtered_df, total_articles, on='country', how='left')
    filtered_df = filtered_df.drop_duplicates()

    # Melt the DataFrame to have a separate row for each framing
    melted_df = filtered_df.melt(id_vars=['country', 'total_articles', 'total_frequency'], var_name='Framing', value_name='Frequency')
    # Convert 'Frequency' to integer
    melted_df['Frequency'] = melted_df['Frequency'].astype(int)

    # Remove 'labels.' prefix from the 'Framing' column
    melted_df['Framing'] = melted_df['Framing'].str.replace('labels.', '')

    # Rename 'Framing' to 'Persuasion Techniques'
    melted_df.rename(columns={'Framing': 'Persuasion Techniques'}, inplace=True)
    melted_df['Persuasion Techniques'] = melted_df['Persuasion Techniques'].str.replace('_', ' ')
    melted_df['Persuasion Techniques'] = melted_df['Persuasion Techniques'].str.replace('-', ' ')

    # Create a new column for the percentage
    melted_df['Percentage'] = (melted_df['Frequency'] / melted_df['total_frequency']) * 100

    # Sort the data by country and frequency (in descending order)
    melted_df.sort_values(by=['country', 'Percentage'], ascending=[True, False], inplace=True)

            # Combine country and total articles
    melted_df['country'] = melted_df['country'] + " (" + melted_df['total_articles'].astype(str) + ")"

    # Sort the data by total articles (in descending order)

    melted_df.sort_values(by=['total_articles', 'Percentage'], ascending=[True, False], inplace=True)
    color_mapping = {
        'Loaded Language': '#FFD16A', #6941BF
        'Obfuscation Vagueness Confusion': '#0468BF',
        'Conversation Killer': '#8FCEFF',
        'Appeal to Time':'#6941BF',
        'Whataboutism': '#a9becf',
        'Red Herring':'#80F29D' ,
        'Straw Man': '#FFABAB',
        'Causal Oversimplification':'#9f7fe3' ,
        'Appeal to Values':'#b1cce3' ,
        'Appeal to Popularity': '#b39696',
        'Appeal to Hypocrisy': '#2f8dd5',
        'Appeal to Authority': '#bd7373',
        'Consequential Oversimplification': '#d4cdcd',
        'False Dilemma No Choice': '#5c9c6c',
        'Repetition': '#945454',
        'Slogans': '#af9cd6',
        'Doubt': '#edaa13',
        'Exaggeration Minimisation': '#958ca8',
        'Name Calling Labeling': '#77abd9',
        'Flag Waving': '#F22E2E', # 
        'Appeal to Fear Prejudice': '#9ad6ac',
        'Guilt by Association': '#6b0c0c',
        'Questioning the Reputation': '#ffdd91',
    }

    # Plotting the graph using Plotly Express
    # fig = px.bar(melted_df, x='Percentage', y='country', color='Persuasion Techniques', orientation='h', 
    #                 color_discrete_map=color_mapping,  # use the color mapping
    #                 title="Distribution of Persuasion Techniques by Country",
    #                 hover_data={
    #                     'Frequency': True,
    #                     'Percentage': ':.2f'  # Format as float with 2 decimal places
    #                 },
    #                 labels={
    #                     'Percentage': 'Percentage of Persuasion Technique',
    #                     'Frequency': 'Frequency of Persuasion Technique'
    #                 })

    # # Add axes lines
    # fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    # fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    # fig.update_layout(height=700, width=900) 
    # st.plotly_chart(fig, use_container_width=True)
    

    fig = px.bar(melted_df, 
                x='Percentage', 
                y='country', 
                color='Persuasion Techniques', 
                orientation='h',
                color_discrete_map=color_mapping,
                title="Distribution of Persuasion Techniques by Country",
                custom_data=['Persuasion Techniques', 'Percentage', 'Frequency'])

    fig.update_traces(
        hovertemplate="%{customdata[0]}: %{customdata[1]:.2f}%  (%{customdata[2]:,} times)<extra></extra>"
    )

    # Add axes lines
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', title='Percentage of Persuasion Technique')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', title='Country')

    fig.update_layout(height=700, width=900)

    st.plotly_chart(fig, use_container_width=True)
 ########################################################################################


    # Get list of unique countries
    countries = country_article_counts_df_subtask3.index.get_level_values(0).unique()

    # Create buttons for selection
    option = st.selectbox('Choose an option', ('Compare', 'Aggregate'))

    # Add title and labels
    if option == 'Aggregate':
        fig = go.Figure(data=go.Scatter(x=article_counts_df_subtask3.index, 
                                        y=article_counts_df_subtask3['number_of_articles'],
                                        mode='lines'))
        fig.update_layout(title='Number of Articles Over Time (Aggregate) by source',
                        xaxis_title='Date',
                        yaxis_title='Number of Articles')

    else:
        # Convert selected_countries to set and back to list to remove duplicates
        selected_sources = list(set(selected_sources))
        # Filter the dataframe for selected countries
        filtered_df = source_article_counts_subtask3.loc[selected_sources]
        
        fig = go.Figure()
        
        for source in selected_sources:
            source_df = filtered_df.loc[source]
            # Check if returned object is a Series or DataFrame
            if isinstance(source_df, pd.Series):
                x_data = [source_df.name]  # When it's a Series, name attribute gives the index
                y_data = [source_df.values[0]]
            else:
                x_data = source_df.index
                y_data = source_df['number_of_articles']
            fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=source))


        fig.update_layout(title='Number of Articles Over Time (Compare) by source',
                        xaxis_title='Date',
                        yaxis_title='Number of Articles')

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)


    ########################################################################################

        # Add title and labels
    if option == 'Aggregate':
        fig = go.Figure(data=go.Scatter(x=article_counts_df_subtask3.index, 
                                        y=article_counts_df_subtask3['number_of_articles'],
                                        mode='lines'))
        fig.update_layout(title='Number of Articles Over Time (Aggregate) by country',
                        xaxis_title='Date',
                        yaxis_title='Number of Articles')

    else:
        # Convert selected_countries to set and back to list to remove duplicates
        selected_countries = list(set(selected_countries))
        # Filter the dataframe for selected countries
        filtered_df = country_article_counts_df_subtask3.loc[selected_countries]
        
        fig = go.Figure()
        
        for country in selected_countries:
            country_df = filtered_df.loc[country]
            # Check if returned object is a Series or DataFrame
            if isinstance(country_df, pd.Series):
                x_data = [country_df.name]  # When it's a Series, name attribute gives the index
                y_data = [country_df.values[0]]
            else:
                x_data = country_df.index
                y_data = country_df['number_of_articles']
            fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=country))


        fig.update_layout(title='Number of Articles Over Time (Compare) by country',
                        xaxis_title='Date',
                        yaxis_title='Number of Articles')

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

    ########################################################################################


    CONTINENT_MAP = {
        'AF': 'Africa',
        'AS': 'Asia',
        'EU': 'Europe',
        'NA': 'North America',
        'SA': 'South America',
        'OC': 'Oceania',
        'AN': 'Antarctica'
    }

    def get_region(country):
        try:
            country_code = country_name_to_country_alpha2(country)
            continent_code = country_alpha2_to_continent_code(country_code)
            return CONTINENT_MAP[continent_code]
        except KeyError:
            return "Unknown"

    st.title('News Analysis Dashboard')
    st.markdown("""
    Distribution of Persuation Techniques in the Articles across Countries and Regions
    """)

    # Reset index of the dataframe
    aggregated_df_subtask3.reset_index(level=0, inplace=True)

    # Convert wide data to long data
    long_format_df = pd.melt(aggregated_df_subtask3, id_vars=['country'], var_name='framings', value_name='percentage')

    # Create new region column
    long_format_df['region'] = long_format_df['country'].apply(get_region)

    # Filter dataframe by selected countries
    filtered_df = long_format_df[long_format_df['country'].isin(selected_countries)]
    # Calculate total frequency of all persuasion techniques for each country
    total_frequencies = filtered_df.groupby('country')['percentage'].sum().reset_index()
    total_frequencies.columns = ['country', 'total_frequency']

    # Merge this with your DataFrame
    filtered_df = pd.merge(filtered_df, total_frequencies, on='country', how='left')

    # Calculate relative frequency (percentage) for each persuasion technique
    filtered_df['percentage'] = (filtered_df['percentage'] / filtered_df['total_frequency']) * 100

        # Remove 'labels.' prefix from the 'framings' column
    filtered_df['framings'] = filtered_df['framings'].str.replace('labels.', '')

        # Remove rows with zero percentage
    filtered_df = filtered_df[filtered_df['percentage'] > 0]

    # Ensure unique Framing labels
    filtered_df['framings'] = filtered_df['framings'].astype('category')
    filtered_df['framings'] = filtered_df['framings'].str.replace('_', ' ')
    filtered_df['framings'] = filtered_df['framings'].str.replace('-', ' ')

    fig = px.sunburst(filtered_df, path=['region', 'country',  'framings'], values='percentage', color='percentage')

    fig.update_layout(width=1200, height=1000)

    st.plotly_chart(fig)

elif st.session_state.page  == "Persuasion Techniques: Coarse-Grained Propaganda":
    # option = 'Russo-Ukrainian War'
    # option = st.selectbox('Choose a topic', ('Climate Change', 'Russo-Ukrainian War', 'Combined'), index=1)

    # #####loading data ######
    # if option == 'Russo-Ukrainian War':
    #     media_agg = RU_media_agg
    #     # print(media_agg)
    #     country_to_media = RU_country_to_media
    #     # print(country_to_media)
    #     aggregated_df = RU_aggregated_df
    #     # print(aggregated_df)
    #     article_counts_df = RU_article_counts_df
    #     # print(article_counts_df)
    #     country_article_counts_df = RU_country_article_counts_df
    #     # print(country_article_counts_df)

    #     media_agg_subtask3 = RU_media_agg_subtask3
    #     country_to_media_subtask3 = RU_country_to_media_subtask3
    #     aggregated_df_subtask3 = RU_aggregated_df_subtask3
    #     article_counts_df_subtask3 = RU_article_counts_df_subtask3
    #     country_article_counts_df_subtask3 = RU_country_article_counts_df_subtask3
    #     source_article_counts = RU_source_article_counts
    #     # print(source_article_counts)
    #     source_article_counts_subtask3 = RU_source_article_counts_subtask3
    # elif option == 'Climate Change':
    #     media_agg = CH_media_agg
    #     # print(media_agg)
    #     country_to_media = CH_country_to_media
    #     # print(country_to_media)
    #     aggregated_df = CH_aggregated_df
    #     # print(aggregated_df)
    #     article_counts_df = CH_article_counts_df
    #     # print(article_counts_df)
    #     country_article_counts_df = CH_country_article_counts_df
    #     # print(country_article_counts_df)

    #     media_agg_subtask3 = CH_media_agg_subtask3
    #     country_to_media_subtask3 = CH_country_to_media_subtask3
    #     aggregated_df_subtask3 = CH_aggregated_df_subtask3
    #     article_counts_df_subtask3 = CH_article_counts_df_subtask3
    #     country_article_counts_df_subtask3 = CH_country_article_counts_df_subtask3

    #     source_article_counts = CH_source_article_counts
    #     # print(source_article_counts)
    #     source_article_counts_subtask3 = CH_source_article_counts_subtask3
        
    # elif option == 'Combined':
    media_agg = both_media_agg
    # print(media_agg)
    country_to_media = both_country_to_media
    # print(country_to_media)
    aggregated_df = both_aggregated_df
    # print(aggregated_df)
    article_counts_df = both_article_counts_df
    # print(article_counts_df)
    country_article_counts_df = both_country_article_counts_df
    # print(country_article_counts_df)

    media_agg_subtask3 = both_media_agg_subtask3
    country_to_media_subtask3 = both_country_to_media_subtask3
    aggregated_df_subtask3 = both_aggregated_df_subtask3
    article_counts_df_subtask3 = both_article_counts_df_subtask3
    country_article_counts_df_subtask3 = both_country_article_counts_df_subtask3

    source_article_counts = both_source_article_counts
    # print(source_article_counts)
    source_article_counts_subtask3 = both_source_article_counts_subtask3        
    
    ###########################

       # Number of countries
    num_countries = len(country_to_media)
    # Number of articles
    num_articles = article_counts_df['number_of_articles'].sum()

    # Number of media sources
    num_media_sources = len(media_agg)

    # Create columns
    col1, col2, col3 = st.columns(3)

    # Display statistics in each column within a box
    with col1:
        st.markdown(f"""
            <div style="background-color: #f5f3f6; border: 1px solid #6c757d; border-radius: 5px; padding: 10px">
                <h4 style="color: #6c757d; text-align: center">Number of countries</h4>
                <p style="text-align: center; font-size: 20px; color: black;">{num_countries}</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="background-color: #f5f3f6; border: 1px solid #6c757d; border-radius: 5px; padding: 10px">
                <h4 style="color: #6c757d; text-align: center">Number of articles</h4>
                <p style="text-align: center; font-size: 20px; color: black;">{num_articles}</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div style="background-color: #f5f3f6; border: 1px solid #6c757d; border-radius: 5px; padding: 10px">
                <h4 style="color: #6c757d; text-align: center">Number of media sources</h4>
                <p style="text-align: center; font-size: 20px; color: black;">{num_media_sources}</p>
            </div>
        """, unsafe_allow_html=True)

    if 'selected_pairs' not in st.session_state:
        st.session_state.selected_pairs = []
        
        import pandas as pd
    from ast import literal_eval

    # Create a dictionary mapping sources to their correct countries
    correct_countries = {
        'europesun': 'Hungary',
        'menafn': 'Jordan',
        'northernirelandnews': 'Ireland',
        'urdupoint': 'Pakistan',
        'polandsun': 'Poland',
        'dailysabah': 'Turkey'
    }

    # Copy of the original DataFrame
    corrected_df = country_to_media.copy()
    # Iterate over the rows in the dataframe
    for index, row in corrected_df.iterrows():
        # Convert the source_frequencies to a list of tuples
        source_frequencies = literal_eval(row['source_frequencies'])

        # List to store the new source_frequencies for this row
        new_source_frequencies = []

        # Iterate over the source_frequencies
        for source_frequency in source_frequencies:
            source = source_frequency[0].lower()  # Get the source name

            # If the source is in the correct_countries dictionary
            if source in correct_countries:
                # Get the index of the correct country row in corrected_df
                correct_country_index = corrected_df[corrected_df['country'] == correct_countries[source]].index[0]

                # Get the source_frequencies for the correct country
                correct_country_sources = literal_eval(corrected_df.at[correct_country_index, 'source_frequencies'])

                # Add the source_frequency to correct_country_sources
                correct_country_sources.append(source_frequency)

                # Update the source_frequencies for the correct country
                corrected_df.at[correct_country_index, 'source_frequencies'] = str(correct_country_sources)
            else:
                # Add the source_frequency to new_source_frequencies
                new_source_frequencies.append(source_frequency)

        # Update the source_frequencies for this row
        corrected_df.at[index, 'source_frequencies'] = str(new_source_frequencies)

    # Append the corrected rows to country_to_media
    country_to_media = corrected_df
    # Calculate total number of articles for each country
    print(country_article_counts_df)
    total_articles = country_article_counts_df.groupby('country').sum().reset_index()
    total_articles.columns = ['country', 'total_articles']
    
    # Create a dictionary that maps country names to total number of articles
    country_to_total_articles = total_articles.set_index('country')['total_articles'].to_dict()
    
    available_countries = country_to_media['country'].unique().tolist()
    
    # Add the number of articles per country
    total_articles = country_article_counts_df.groupby('country').sum().reset_index()
    total_articles.columns = ['country', 'total_articles']
    countries = country_to_media['country'].unique().tolist()
    total_articles = total_articles['total_articles'].tolist()
    
    for country in countries:
        if country in available_countries:
            row_num = countries.index(country)
            for i in range(len(available_countries)):
                if available_countries[i] == country:
                    available_countries[i] += " (" + str(total_articles[row_num]) + ")"
    
    available_countries.sort(key=lambda x: int(x.split(' ')[-1].strip('()')), reverse=True)
    country = st.selectbox('Select country', available_countries, key='country')
    country = country[:country.index("(") - 1]
    source_counts_list = country_to_media[country_to_media['country'] == country]['source_frequencies'].values[0]
    # print(source_counts_list)
    source_counts_list = ast.literal_eval(source_counts_list)
    selected_sources_dict = {item[0]: item[1] for item in source_counts_list}
    selected_sources_dict = {k: (v if v is not None else 0) for k, v in selected_sources_dict.items()}
    selected_sources_dict = {k: v for k, v in selected_sources_dict.items() if v > 0}
    
    selected_sources = [pair[1] for pair in st.session_state.selected_pairs if pair[0] == country]
    available_sources = selected_sources_dict.keys()
    
    # Convert to lower case and create a mapping from lowercase to original
    available_sources_lower_to_original = {source.lower(): source for source in available_sources}
    
    source_articles_data = []
    # Loop through each row in the country_to_media dataframe
    for index, row in country_to_media.iterrows():
        # Loop through each source_frequency list in the current row
        for source_frequency in eval(row['source_frequencies']):
            # Append a dictionary with the media source and its number of articles to the list
            source_articles_data.append({'source': source_frequency[0], 'total_articles': source_frequency[1] if source_frequency[1] else 0})
            
    # Convert the list of dictionaries into a dataframe
    source_articles_df = pd.DataFrame(source_articles_data)
    
    # Get the list of lowercase sources
    available_sources_lower = [source.lower() for source in available_sources]
    
    # Add the total number of articles per source
    for source in available_sources_lower:
        if source in available_sources_lower_to_original.keys():
            row_num = next((index for (index, d) in enumerate(source_articles_data) if d["source"].lower() == source), None)
            if row_num is not None:
                number = source_articles_data[row_num]['total_articles']
                available_sources_lower[available_sources_lower.index(source)] = source + " (" + str(number) + ")"
            else:
                available_sources_lower[available_sources_lower.index(source)] = source + " (0)"
    
    # Now sort available_sources_lower based on the total number of articles in descending order
    available_sources_lower.sort(key=lambda x: int(x.split(' ')[-1].strip('()')), reverse=True)
    
    source = st.selectbox('Select source', available_sources_lower, key='source')

    ###############################################################################

    if st.button('Add selection'):
        # Retrieve the original casing version of the source
        index=source.index(' ')
        original_case_source = available_sources_lower_to_original[source[:index]]
        if ((country, original_case_source) not in st.session_state.selected_pairs):
            st.session_state.selected_pairs.append((country, original_case_source))

    if st.button('Remove last selection') and st.session_state.selected_pairs:
        st.session_state.selected_pairs.pop()

    selected_countries = [pair[0] for pair in st.session_state.selected_pairs]
    selected_sources = [pair[1] for pair in st.session_state.selected_pairs]

    # rest of your code


    # Now you can use selected_countries and selected_sources for your plots.


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

    ########################################################################################

    # First, create a list to hold all the media source data
    source_articles_data = []

    # Loop through each row in the country_to_media dataframe
    for index, row in country_to_media_subtask3.iterrows():
        # Loop through each source_frequency list in the current row
        for source_frequency in eval(row['source_frequencies']):
            # Append a dictionary with the media source and its number of articles to the list
            source_articles_data.append({'source': source_frequency[0], 'total_articles': source_frequency[1]})

    # Convert the list of dictionaries into a dataframe
    source_articles_df = pd.DataFrame(source_articles_data)

    # Calculate total number of articles for each source
    total_articles = source_articles_df.groupby('source').sum()
    # Filter media_agg DataFrame for the selected source
    filtered_df = media_agg_subtask3.loc[selected_sources]

    # Add a column for the total frequency of persuasion techniques for each source
    filtered_df['total_frequency'] = filtered_df.sum(axis=1)

    # Merge this with your DataFrame
    filtered_df = pd.merge(filtered_df, total_articles, on='source', how='left')
    # Melt the DataFrame to have a separate row for each framing
    melted_df = filtered_df.reset_index().melt(id_vars=['source', 'total_articles', 'total_frequency'], var_name='Framing', value_name='Frequency')

    # Remove 'labels.' prefix from the 'Framing' column
    melted_df['Framing'] = melted_df['Framing'].str.replace('labels.', '')

    # Rename 'Framing' to 'Persuasion Techniques'
    melted_df.rename(columns={'Framing': 'Persuasion Techniques'}, inplace=True)

    # Create a new column for the percentage
    melted_df['Percentage'] = (melted_df['Frequency'] / melted_df['total_frequency']) * 100

    # Sort the DataFrame by 'source' and 'Frequency' in descending order
    melted_df.sort_values(by=['source', 'Percentage'], ascending=[True, False], inplace=True)

            # Combine country and total articles
    melted_df['source'] = melted_df['source'] + " (" + melted_df['total_articles'].astype(str) + ")"

    # Sort the data by total articles (in descending order)
    # melted_df = melted_df.sort_values(by=['total_articles'], ascending=True)

    melted_df.sort_values(by=['total_articles', 'Percentage'], ascending=[True, False], inplace=True)

    # Mapping of techniques to categories
    technique_to_category = {
        'Flag_Waving': 'JUSTIFICATION',
        'Appeal_to_Authority': 'JUSTIFICATION',
        'Appeal_to_Popularity': 'JUSTIFICATION',
        'Appeal_to_Values': 'JUSTIFICATION',
        'Appeal_to_Fear-Prejudice': 'JUSTIFICATION',
        'Straw_Man': 'DISTRACTION',
        'Red_Herring': 'DISTRACTION',
        'Whataboutism': 'DISTRACTION',
        'Causal_Oversimplification': 'SIMPLIFICATION',
        'False_Dilemma-No_Choice': 'SIMPLIFICATION',
        'Consequential_Oversimplification': 'SIMPLIFICATION',
        'Slogans': 'CALL',
        'Conversation_Killer': 'CALL',
        'Appeal_to_Time': 'CALL',
        'Loaded_Language': 'MANIPULATIVE WORDING',
        'Obfuscation-Vagueness-Confusion': 'MANIPULATIVE WORDING',
        'Exaggeration-Minimisation': 'MANIPULATIVE WORDING',
        'Repetition': 'MANIPULATIVE WORDING',
        'Name_Calling-Labeling': 'ATTACK ON REPUTATION',
        'Guilt_by_Association': 'ATTACK ON REPUTATION',
        'Casting_Doubt': 'ATTACK ON REPUTATION',
        'Appeal_to_Hypocrisy': 'ATTACK ON REPUTATION',
        'Questioning_the_Reputation': 'ATTACK ON REPUTATION'
    }

    grouped_df = melted_df
    # Add a new column 'Category' to the dataframe
    grouped_df['Category'] = melted_df['Persuasion Techniques'].map(technique_to_category)
    
    # # Group by 'source' and 'Category' and sum the 'Frequency'
    # grouped_df = melted_df.groupby(['source', 'Category']).sum().reset_index()

    # grouped_df['Percentage'] = (grouped_df['Frequency'] / grouped_df['total_frequency']) * 100

        # Group by 'source' and 'Category' and sum the 'Frequency'
    grouped_freq_df = grouped_df.groupby(['source', 'Category'])['Frequency'].sum().reset_index()
    
    # Calculate the total frequency for each source after the techniques are grouped into categories
    total_category_frequency = grouped_freq_df.groupby('source')['Frequency'].sum().reset_index()
    total_category_frequency.columns = ['source', 'total_category_frequency']
    
    # Merge the total frequency dataframe with the grouped frequency dataframe
    grouped_freq_df = pd.merge(grouped_freq_df, total_category_frequency, on='source', how='left')
    
    # Calculate the percentage
    grouped_freq_df['Percentage'] = (grouped_freq_df['Frequency'] / grouped_freq_df['total_category_frequency']) * 100

    grouped_df = grouped_freq_df

    grouped_df.sort_values(by=['source', 'Percentage'], ascending=[True, False], inplace=True)

    grouped_df['total_articles'] = grouped_df['source'].str.extract(r'\((.*?)\)', expand=False).astype(int)

    grouped_df.sort_values(by=['total_articles', 'Percentage'], ascending=[True, False], inplace=True)

    # Plot the graph using Plotly Express
    # fig = px.bar(grouped_df, x='Percentage', y='source', color='Category', orientation='h', 
    #                 title="Distribution of Coarse Grained Propaganda by Source",
    #                 hover_data={
    #                     'Frequency': True,
    #                     'Percentage': ':.2f'  # Format as float with 2 decimal places
    #                 },
    #                 labels={
    #                     'Percentage': 'Percentage of Coarse Grained Propaganda Technique',
    #                     'Frequency': 'Frequency of Coarse Grained Propaganda Technique'
    #                 })

    # # Add axes lines
    # fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    # fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    # st.plotly_chart(fig, use_container_width=True)
    
    fig = px.bar(grouped_df, 
             x='Percentage', 
             y='source', 
             color='Category', 
             orientation='h',
             title="Distribution of Coarse Grained Propaganda by Source",
             custom_data=['Category', 'Percentage', 'Frequency'])

    fig.update_traces(
        hovertemplate="%{customdata[0]}: %{customdata[1]:.2f}%  (%{customdata[2]:,} times)<extra></extra>"
    )

    # Add axes lines
    fig.update_xaxes(showline=True, 
                    linewidth=2, 
                    linecolor='black', 
                    title='Percentage of Coarse Grained Propaganda Technique')
    fig.update_yaxes(showline=True, 
                    linewidth=2, 
                    linecolor='black', 
                    title='Source')

    st.plotly_chart(fig, use_container_width=True)


    ########################################################################################

    ########################################################################################


    # Calculate total number of articles for each country
    total_articles = country_article_counts_df_subtask3.groupby('country').sum().reset_index()
    total_articles.columns = ['country', 'total_articles']

    # Filter data based on selected countries
    filtered_df = aggregated_df_subtask3.loc[selected_countries]
    # Add a column for the total frequency of persuasion techniques for each country
    filtered_df['total_frequency'] = filtered_df.sum(axis=1)

    # Reset the index to make 'country' a column
    filtered_df = filtered_df.reset_index()
    # Merge this with your DataFrame
    filtered_df = pd.merge(filtered_df, total_articles, on='country', how='left')
    filtered_df = filtered_df.drop_duplicates()

    # Melt the DataFrame to have a separate row for each framing
    melted_df = filtered_df.melt(id_vars=['country', 'total_articles', 'total_frequency'], var_name='Framing', value_name='Frequency')
    # Convert 'Frequency' to integer
    melted_df['Frequency'] = melted_df['Frequency'].astype(int)

    # Remove 'labels.' prefix from the 'Framing' column
    melted_df['Framing'] = melted_df['Framing'].str.replace('labels.', '')

    # Rename 'Framing' to 'Persuasion Techniques'
    melted_df.rename(columns={'Framing': 'Persuasion Techniques'}, inplace=True)

    # Create a new column for the percentage
    # melted_df['Percentage'] = (melted_df['Frequency'] / melted_df['total_frequency']) * 100

    melted_df['Percentage'] = (melted_df['Frequency'] / melted_df['total_articles']) * 100

    # Sort the data by country and frequency (in descending order)
    melted_df.sort_values(by=['country', 'Percentage'], ascending=[True, False], inplace=True)

            # Combine country and total articles
    melted_df['country'] = melted_df['country'] + " (" + melted_df['total_articles'].astype(str) + ")"

    # Sort the data by total articles (in descending order)

    melted_df.sort_values(by=['total_articles', 'Percentage'], ascending=[True, False], inplace=True)

    # Mapping of techniques to categories
    technique_to_category = {
        'Flag_Waving': 'JUSTIFICATION',
        'Appeal_to_Authority': 'JUSTIFICATION',
        'Appeal_to_Popularity': 'JUSTIFICATION',
        'Appeal_to_Values': 'JUSTIFICATION',
        'Appeal_to_Fear-Prejudice': 'JUSTIFICATION',
        'Straw_Man': 'DISTRACTION',
        'Red_Herring': 'DISTRACTION',
        'Whataboutism': 'DISTRACTION',
        'Causal_Oversimplification': 'SIMPLIFICATION',
        'False_Dilemma-No_Choice': 'SIMPLIFICATION',
        'Consequential_Oversimplification': 'SIMPLIFICATION',
        'Slogans': 'CALL',
        'Conversation_Killer': 'CALL',
        'Appeal_to_Time': 'CALL',
        'Loaded_Language': 'MANIPULATIVE WORDING',
        'Obfuscation-Vagueness-Confusion': 'MANIPULATIVE WORDING',
        'Exaggeration-Minimisation': 'MANIPULATIVE WORDING',
        'Repetition': 'MANIPULATIVE WORDING',
        'Name_Calling-Labeling': 'ATTACK ON REPUTATION',
        'Guilt_by_Association': 'ATTACK ON REPUTATION',
        'Casting_Doubt': 'ATTACK ON REPUTATION',
        'Appeal_to_Hypocrisy': 'ATTACK ON REPUTATION',
        'Questioning_the_Reputation': 'ATTACK ON REPUTATION'
    }

    grouped_df = melted_df
    grouped_df['Category'] = melted_df['Persuasion Techniques'].map(technique_to_category)
    # Add a new column 'Category' to the dataframe    # # Group by 'source' and 'Category' and sum the 'Frequency'
    # grouped_df = melted_df.groupby(['source', 'Category']).sum().reset_index()

    # grouped_df['Percentage'] = (grouped_df['Frequency'] / grouped_df['total_frequency']) * 100

        # Group by 'source' and 'Category' and sum the 'Frequency'
    grouped_freq_df = grouped_df.groupby(['country', 'Category'])['Frequency'].sum().reset_index()
    
    # Calculate the total frequency for each source after the techniques are grouped into categories
    total_category_frequency = grouped_freq_df.groupby('country')['Frequency'].sum().reset_index()
    total_category_frequency.columns = ['country', 'total_category_frequency']
    
    # Merge the total frequency dataframe with the grouped frequency dataframe
    grouped_freq_df = pd.merge(grouped_freq_df, total_category_frequency, on='country', how='left')
    
    # Calculate the percentage
    grouped_freq_df['Percentage'] = (grouped_freq_df['Frequency'] / grouped_freq_df['total_category_frequency']) * 100

    grouped_df = grouped_freq_df

    grouped_df['total_articles'] = grouped_df['country'].str.extract(r'\((.*?)\)', expand=False).astype(int)

    grouped_df.sort_values(by=['total_articles', 'Percentage'], ascending=[True, False], inplace=True)

    # Plot the graph using Plotly Express
    # fig = px.bar(grouped_df, x='Percentage', y='country', color='Category', orientation='h', 
    #                 title="Distribution of Coarse Grained Propaganda by Country",
    #                 hover_data={
    #                     'Frequency': True,
    #                     'Percentage': ':.2f'  # Format as float with 2 decimal places
    #                 },
    #                 labels={
    #                     'Percentage': 'Percentage of Coarse Grained Propaganda Technique',
    #                     'Frequency': 'Frequency of Coarse Grained Propaganda Technique'
    #                 })

    # # Add axes lines
    # fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    # fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    # st.plotly_chart(fig, use_container_width=True)
    fig = px.bar(grouped_df, 
             x='Percentage', 
             y='country', 
             color='Category', 
             orientation='h',
             title="Distribution of Coarse Grained Propaganda by Country",
             custom_data=['Category', 'Percentage', 'Frequency'])

    fig.update_traces(
        hovertemplate="%{customdata[0]}: %{customdata[1]:.2f}%  (%{customdata[2]:,} times)<extra></extra>"
    )

    # Add axes lines
    fig.update_xaxes(showline=True, 
                    linewidth=2, 
                    linecolor='black', 
                    title='Percentage of Coarse Grained Propaganda Technique')
    fig.update_yaxes(showline=True, 
                    linewidth=2, 
                    linecolor='black', 
                    title='Country')

    st.plotly_chart(fig, use_container_width=True)


    ########################################################################################

    # Get list of unique countries
    countries = country_article_counts_df_subtask3.index.get_level_values(0).unique()

    # Create buttons for selection
    option = st.selectbox('Choose an option', ('Compare', 'Aggregate'))

    # Add title and labels
    if option == 'Aggregate':
        fig = go.Figure(data=go.Scatter(x=article_counts_df_subtask3.index, 
                                        y=article_counts_df_subtask3['number_of_articles'],
                                        mode='lines'))
        fig.update_layout(title='Number of Articles Over Time (Aggregate) by source',
                        xaxis_title='Date',
                        yaxis_title='Number of Articles')

    else:
        # Convert selected_countries to set and back to list to remove duplicates
        selected_sources = list(set(selected_sources))
        # Filter the dataframe for selected countries
        filtered_df = source_article_counts_subtask3.loc[selected_sources]
        
        fig = go.Figure()
        
        for source in selected_sources:
            source_df = filtered_df.loc[source]
            # Check if returned object is a Series or DataFrame
            if isinstance(source_df, pd.Series):
                x_data = [source_df.name]  # When it's a Series, name attribute gives the index
                y_data = [source_df.values[0]]
            else:
                x_data = source_df.index
                y_data = source_df['number_of_articles']
            fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=source))


        fig.update_layout(title='Number of Articles Over Time (Compare) by source',
                        xaxis_title='Date',
                        yaxis_title='Number of Articles')

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)


    ########################################################################################

        # Add title and labels
    if option == 'Aggregate':
        fig = go.Figure(data=go.Scatter(x=article_counts_df_subtask3.index, 
                                        y=article_counts_df_subtask3['number_of_articles'],
                                        mode='lines'))
        fig.update_layout(title='Number of Articles Over Time (Aggregate) by country',
                        xaxis_title='Date',
                        yaxis_title='Number of Articles')

    else:
        # Convert selected_countries to set and back to list to remove duplicates
        selected_countries = list(set(selected_countries))
        # Filter the dataframe for selected countries
        filtered_df = country_article_counts_df_subtask3.loc[selected_countries]
        
        fig = go.Figure()
        
        for country in selected_countries:
            country_df = filtered_df.loc[country]
            # Check if returned object is a Series or DataFrame
            if isinstance(country_df, pd.Series):
                x_data = [country_df.name]  # When it's a Series, name attribute gives the index
                y_data = [country_df.values[0]]
            else:
                x_data = country_df.index
                y_data = country_df['number_of_articles']
            fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=country))


        fig.update_layout(title='Number of Articles Over Time (Compare) by country',
                        xaxis_title='Date',
                        yaxis_title='Number of Articles')

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

    ########################################################################################

 
    CONTINENT_MAP = {
        'AF': 'Africa',
        'AS': 'Asia',
        'EU': 'Europe',
        'NA': 'North America',
        'SA': 'South America',
        'OC': 'Oceania',
        'AN': 'Antarctica'
    }

    def get_region(country):
        try:
            country_code = country_name_to_country_alpha2(country)
            continent_code = country_alpha2_to_continent_code(country_code)
            return CONTINENT_MAP[continent_code]
        except KeyError:
            return "Unknown"

    st.title('News Analysis Dashboard')
    st.markdown("""
    Distribution of Persuation Techniques in the Articles across Countries and Regions
    """)


    # Mapping of techniques to categories
    technique_to_category = {
        'labels.Flag_Waving': 'JUSTIFICATION',
        'labels.Appeal_to_Authority': 'JUSTIFICATION',
        'labels.Appeal_to_Popularity': 'JUSTIFICATION',
        'labels.Appeal_to_Values': 'JUSTIFICATION',
        'labels.Appeal_to_Fear-Prejudice': 'JUSTIFICATION',
        'labels.Straw_Man': 'DISTRACTION',
        'labels.Red_Herring': 'DISTRACTION',
        'labels.Whataboutism': 'DISTRACTION',
        'labels.Causal_Oversimplification': 'SIMPLIFICATION',
        'labels.False_Dilemma-No_Choice': 'SIMPLIFICATION',
        'labels.Consequential_Oversimplification': 'SIMPLIFICATION',
        'labels.Slogans': 'CALL',
        'labels.Conversation_Killer': 'CALL',
        'labels.Appeal_to_Time': 'CALL',
        'labels.Loaded_Language': 'MANIPULATIVE WORDING',
        'labels.Obfuscation-Vagueness-Confusion': 'MANIPULATIVE WORDING',
        'labels.Exaggeration-Minimisation': 'MANIPULATIVE WORDING',
        'labels.Repetition': 'MANIPULATIVE WORDING',
        'labels.Name_Calling-Labeling': 'ATTACK ON REPUTATION',
        'labels.Guilt_by_Association': 'ATTACK ON REPUTATION',
        'labels.Doubt': 'ATTACK ON REPUTATION',
        'labels.Appeal_to_Hypocrisy': 'ATTACK ON REPUTATION',
        'labels.Questioning_the_Reputation': 'ATTACK ON REPUTATION'
    }

    # Reset index of the dataframe
    aggregated_df_subtask3.reset_index(level=0, inplace=True)

    # Convert wide data to long data
    long_format_df = pd.melt(aggregated_df_subtask3, id_vars=['country'], var_name='framings', value_name='percentage')

    # Create new 'Category' column
    long_format_df['Category'] = long_format_df['framings'].map(technique_to_category)

    # Create new region column
    long_format_df['region'] = long_format_df['country'].apply(get_region)

    long_format_df = long_format_df.drop_duplicates(subset=['region', 'country', 'Category'])

    # Filter dataframe by selected countries
    filtered_df = long_format_df[long_format_df['country'].isin(selected_countries)]
    # Calculate total frequency of all persuasion techniques for each country
    total_frequencies = filtered_df.groupby('country')['percentage'].sum().reset_index()
    total_frequencies.columns = ['country', 'total_frequency']

    # Merge this with your DataFrame
    filtered_df = pd.merge(filtered_df, total_frequencies, on='country', how='left')

    # Calculate relative frequency (percentage) for each persuasion technique
    filtered_df['percentage'] = (filtered_df['percentage'] / filtered_df['total_frequency']) * 100

    # Remove rows with zero percentage
    filtered_df = filtered_df[filtered_df['percentage'] > 0]

    fig = px.sunburst(filtered_df, path=['region', 'country',  'Category'], values='percentage', color='percentage')

    fig.update_layout(width=1200, height=1000)

    st.plotly_chart(fig)


 #######################################   
elif st.session_state.page  == "Persuasion Techniques: Ethos, Logos, Pathos":
    # option = 'Russo-Ukrainian War'
    # option = st.selectbox('Choose a topic', ('Climate Change', 'Russo-Ukrainian War', 'Combined'), index=1)

    # #####loading data ######
    # if option == 'Russo-Ukrainian War':
    #     media_agg = RU_media_agg
    #     # print(media_agg)
    #     country_to_media = RU_country_to_media
    #     # print(country_to_media)
    #     aggregated_df = RU_aggregated_df
    #     # print(aggregated_df)
    #     article_counts_df = RU_article_counts_df
    #     # print(article_counts_df)
    #     country_article_counts_df = RU_country_article_counts_df
    #     # print(country_article_counts_df)

    #     media_agg_subtask3 = RU_media_agg_subtask3
    #     country_to_media_subtask3 = RU_country_to_media_subtask3
    #     aggregated_df_subtask3 = RU_aggregated_df_subtask3
    #     article_counts_df_subtask3 = RU_article_counts_df_subtask3
    #     country_article_counts_df_subtask3 = RU_country_article_counts_df_subtask3
    #     source_article_counts = RU_source_article_counts
    #     # print(source_article_counts)
    #     source_article_counts_subtask3 = RU_source_article_counts_subtask3
    # elif option == 'Climate Change':
    #     media_agg = CH_media_agg
    #     # print(media_agg)
    #     country_to_media = CH_country_to_media
    #     # print(country_to_media)
    #     aggregated_df = CH_aggregated_df
    #     # print(aggregated_df)
    #     article_counts_df = CH_article_counts_df
    #     # print(article_counts_df)
    #     country_article_counts_df = CH_country_article_counts_df
    #     # print(country_article_counts_df)

    #     media_agg_subtask3 = CH_media_agg_subtask3
    #     country_to_media_subtask3 = CH_country_to_media_subtask3
    #     aggregated_df_subtask3 = CH_aggregated_df_subtask3
    #     article_counts_df_subtask3 = CH_article_counts_df_subtask3
    #     country_article_counts_df_subtask3 = CH_country_article_counts_df_subtask3

    #     source_article_counts = CH_source_article_counts
    #     # print(source_article_counts)
    #     source_article_counts_subtask3 = CH_source_article_counts_subtask3
        
    # elif option == 'Combined':
    media_agg = both_media_agg
    # print(media_agg)
    country_to_media = both_country_to_media
    # print(country_to_media)
    aggregated_df = both_aggregated_df
    # print(aggregated_df)
    article_counts_df = both_article_counts_df
    # print(article_counts_df)
    country_article_counts_df = both_country_article_counts_df
    # print(country_article_counts_df)

    media_agg_subtask3 = both_media_agg_subtask3
    country_to_media_subtask3 = both_country_to_media_subtask3
    aggregated_df_subtask3 = both_aggregated_df_subtask3
    article_counts_df_subtask3 = both_article_counts_df_subtask3
    country_article_counts_df_subtask3 = both_country_article_counts_df_subtask3

    source_article_counts = both_source_article_counts
    # print(source_article_counts)
    source_article_counts_subtask3 = both_source_article_counts_subtask3        
    
    ###########################

    
      # Number of countries
    num_countries = len(country_to_media)
    # Number of articles
    num_articles = article_counts_df['number_of_articles'].sum()

    # Number of media sources
    num_media_sources = len(media_agg)

    # Create columns
    col1, col2, col3 = st.columns(3)

    # Display statistics in each column within a box
    with col1:
        st.markdown(f"""
            <div style="background-color: #f5f3f6; border: 1px solid #6c757d; border-radius: 5px; padding: 10px">
                <h4 style="color: #6c757d; text-align: center">Number of countries</h4>
                <p style="text-align: center; font-size: 20px; color: black;">{num_countries}</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="background-color: #f5f3f6; border: 1px solid #6c757d; border-radius: 5px; padding: 10px">
                <h4 style="color: #6c757d; text-align: center">Number of articles</h4>
                <p style="text-align: center; font-size: 20px; color: black;">{num_articles}</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div style="background-color: #f5f3f6; border: 1px solid #6c757d; border-radius: 5px; padding: 10px">
                <h4 style="color: #6c757d; text-align: center">Number of media sources</h4>
                <p style="text-align: center; font-size: 20px; color: black;">{num_media_sources}</p>
            </div>
        """, unsafe_allow_html=True)

    if 'selected_pairs' not in st.session_state:
        st.session_state.selected_pairs = []
        
 
    # Create a dictionary mapping sources to their correct countries
    correct_countries = {
        'europesun': 'Hungary',
        'menafn': 'Jordan',
        'northernirelandnews': 'Ireland',
        'urdupoint': 'Pakistan',
        'polandsun': 'Poland',
        'dailysabah': 'Turkey'
    }

    # Copy of the original DataFrame
    corrected_df = country_to_media.copy()
    # Iterate over the rows in the dataframe
    for index, row in corrected_df.iterrows():
        # Convert the source_frequencies to a list of tuples
        source_frequencies = literal_eval(row['source_frequencies'])

        # List to store the new source_frequencies for this row
        new_source_frequencies = []

        # Iterate over the source_frequencies
        for source_frequency in source_frequencies:
            source = source_frequency[0].lower()  # Get the source name

            # If the source is in the correct_countries dictionary
            if source in correct_countries:
                # Get the index of the correct country row in corrected_df
                correct_country_index = corrected_df[corrected_df['country'] == correct_countries[source]].index[0]

                # Get the source_frequencies for the correct country
                correct_country_sources = literal_eval(corrected_df.at[correct_country_index, 'source_frequencies'])

                # Add the source_frequency to correct_country_sources
                correct_country_sources.append(source_frequency)

                # Update the source_frequencies for the correct country
                corrected_df.at[correct_country_index, 'source_frequencies'] = str(correct_country_sources)
            else:
                # Add the source_frequency to new_source_frequencies
                new_source_frequencies.append(source_frequency)

        # Update the source_frequencies for this row
        corrected_df.at[index, 'source_frequencies'] = str(new_source_frequencies)

    # Append the corrected rows to country_to_media
    country_to_media = corrected_df
    # Calculate total number of articles for each country
    print(country_article_counts_df)
    total_articles = country_article_counts_df.groupby('country').sum().reset_index()
    total_articles.columns = ['country', 'total_articles']
    
    # Create a dictionary that maps country names to total number of articles
    country_to_total_articles = total_articles.set_index('country')['total_articles'].to_dict()
    
    available_countries = country_to_media['country'].unique().tolist()
    
    # Add the number of articles per country
    total_articles = country_article_counts_df.groupby('country').sum().reset_index()
    total_articles.columns = ['country', 'total_articles']
    countries = country_to_media['country'].unique().tolist()
    total_articles = total_articles['total_articles'].tolist()
    
    for country in countries:
        if country in available_countries:
            row_num = countries.index(country)
            for i in range(len(available_countries)):
                if available_countries[i] == country:
                    available_countries[i] += " (" + str(total_articles[row_num]) + ")"
    
    available_countries.sort(key=lambda x: int(x.split(' ')[-1].strip('()')), reverse=True)
    country = st.selectbox('Select country', available_countries, key='country')
    country = country[:country.index("(") - 1]
    source_counts_list = country_to_media[country_to_media['country'] == country]['source_frequencies'].values[0]
    # print(source_counts_list)
    source_counts_list = ast.literal_eval(source_counts_list)
    selected_sources_dict = {item[0]: item[1] for item in source_counts_list}
    selected_sources_dict = {k: (v if v is not None else 0) for k, v in selected_sources_dict.items()}
    selected_sources_dict = {k: v for k, v in selected_sources_dict.items() if v > 0}
    
    selected_sources = [pair[1] for pair in st.session_state.selected_pairs if pair[0] == country]
    available_sources = selected_sources_dict.keys()
    
    # Convert to lower case and create a mapping from lowercase to original
    available_sources_lower_to_original = {source.lower(): source for source in available_sources}
    
    source_articles_data = []
    # Loop through each row in the country_to_media dataframe
    for index, row in country_to_media.iterrows():
        # Loop through each source_frequency list in the current row
        for source_frequency in eval(row['source_frequencies']):
            # Append a dictionary with the media source and its number of articles to the list
            source_articles_data.append({'source': source_frequency[0], 'total_articles': source_frequency[1] if source_frequency[1] else 0})
            
    # Convert the list of dictionaries into a dataframe
    source_articles_df = pd.DataFrame(source_articles_data)
    
    # Get the list of lowercase sources
    available_sources_lower = [source.lower() for source in available_sources]
    
    # Add the total number of articles per source
    for source in available_sources_lower:
        if source in available_sources_lower_to_original.keys():
            row_num = next((index for (index, d) in enumerate(source_articles_data) if d["source"].lower() == source), None)
            if row_num is not None:
                number = source_articles_data[row_num]['total_articles']
                available_sources_lower[available_sources_lower.index(source)] = source + " (" + str(number) + ")"
            else:
                available_sources_lower[available_sources_lower.index(source)] = source + " (0)"
    
    # Now sort available_sources_lower based on the total number of articles in descending order
    available_sources_lower.sort(key=lambda x: int(x.split(' ')[-1].strip('()')), reverse=True)
    
    source = st.selectbox('Select source', available_sources_lower, key='source')

    ###############################################################################

    if st.button('Add selection'):
        # Retrieve the original casing version of the source
        index=source.index(' ')
        original_case_source = available_sources_lower_to_original[source[:index]]
        if ((country, original_case_source) not in st.session_state.selected_pairs):
            st.session_state.selected_pairs.append((country, original_case_source))

    if st.button('Remove last selection') and st.session_state.selected_pairs:
        st.session_state.selected_pairs.pop()

    selected_countries = [pair[0] for pair in st.session_state.selected_pairs]
    selected_sources = [pair[1] for pair in st.session_state.selected_pairs]


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

    ########################################################################################

    # First, create a list to hold all the media source data
    source_articles_data = []

    # Loop through each row in the country_to_media dataframe
    for index, row in country_to_media_subtask3.iterrows():
        # Loop through each source_frequency list in the current row
        for source_frequency in eval(row['source_frequencies']):
            # Append a dictionary with the media source and its number of articles to the list
            source_articles_data.append({'source': source_frequency[0], 'total_articles': source_frequency[1]})

    # Convert the list of dictionaries into a dataframe
    source_articles_df = pd.DataFrame(source_articles_data)

    # Calculate total number of articles for each source
    total_articles = source_articles_df.groupby('source').sum()
    # Filter media_agg DataFrame for the selected source
    filtered_df = media_agg_subtask3.loc[selected_sources]

    # Add a column for the total frequency of persuasion techniques for each source
    filtered_df['total_frequency'] = filtered_df.sum(axis=1)

    # Merge this with your DataFrame
    filtered_df = pd.merge(filtered_df, total_articles, on='source', how='left')
    # Melt the DataFrame to have a separate row for each framing
    melted_df = filtered_df.reset_index().melt(id_vars=['source', 'total_articles', 'total_frequency'], var_name='Framing', value_name='Frequency')

    # Remove 'labels.' prefix from the 'Framing' column
    melted_df['Framing'] = melted_df['Framing'].str.replace('labels.', '')

    # Rename 'Framing' to 'Persuasion Techniques'
    melted_df.rename(columns={'Framing': 'Persuasion Techniques'}, inplace=True)

    # Create a new column for the percentage
    # melted_df['Percentage'] = (melted_df['Frequency'] / melted_df['total_frequency']) * 100

    # Create a new column for the percentage
    melted_df['Percentage'] = (melted_df['Frequency'] / melted_df['total_articles']) * 100

    # Sort the DataFrame by 'source' and 'Frequency' in descending order
    melted_df.sort_values(by=['source', 'Percentage'], ascending=[True, False], inplace=True)

            # Combine country and total articles
    melted_df['source'] = melted_df['source'] + " (" + melted_df['total_articles'].astype(str) + ")"

    # Sort the data by total articles (in descending order)
    # melted_df = melted_df.sort_values(by=['total_articles'], ascending=True)

    melted_df.sort_values(by=['total_articles', 'Percentage'], ascending=[True, False], inplace=True)

    # Mapping of techniques to categories
    technique_to_category = {
        'Flag_Waving': 'Pathos',
        'Appeal_to_Authority': 'Ethos',
        'Appeal_to_Popularity': 'Ethos',
        'Appeal_to_Values': 'Pathos',
        'Appeal_to_Fear-Prejudice': 'Pathos',
        'Straw_Man': 'Logos',
        'Red_Herring': 'Logos',
        'Whataboutism': 'Logos',
        'Causal_Oversimplification': 'Logos',
        'False_Dilemma-No_Choice': 'Logos',
        'Consequential_Oversimplification': 'Logos',
        'Slogans': 'Other',
        'Conversation_Killer': 'Other',
        'Appeal_to_Time': 'Other',
        'Loaded_Language': 'Pathos',
        'Obfuscation-Vagueness-Confusion': 'Logos',
        'Exaggeration-Minimisation': 'Other',
        'Repetition': 'Other',
        'Name_Calling-Labeling': 'Pathos',
        'Guilt_by_Association': 'Ethos',
        'Casting_Doubt': 'Ethos',
        'Appeal_to_Hypocrisy': 'Ethos',
        'Questioning_the_Reputation': 'Ethos'
    }

    grouped_df = melted_df
    # Add a new column 'Category' to the dataframe
    grouped_df['Category'] = melted_df['Persuasion Techniques'].map(technique_to_category)
    
        # Group by 'source' and 'Category' and sum the 'Frequency'
    grouped_freq_df = grouped_df.groupby(['source', 'Category'])['Frequency'].sum().reset_index()
    
    # Calculate the total frequency for each source after the techniques are grouped into categories
    total_category_frequency = grouped_freq_df.groupby('source')['Frequency'].sum().reset_index()
    total_category_frequency.columns = ['source', 'total_category_frequency']
    
    # Merge the total frequency dataframe with the grouped frequency dataframe
    grouped_freq_df = pd.merge(grouped_freq_df, total_category_frequency, on='source', how='left')
    
    # Calculate the percentage
    grouped_freq_df['Percentage'] = (grouped_freq_df['Frequency'] / grouped_freq_df['total_category_frequency']) * 100

    grouped_df = grouped_freq_df
    grouped_df['total_articles'] = grouped_df['source'].str.extract(r'\((.*?)\)', expand=False).astype(int)

    grouped_df.sort_values(by=['total_articles', 'Percentage'], ascending=[True, False], inplace=True)

    # Plot the graph using Plotly Express
    # fig = px.bar(grouped_df, x='Percentage', y='source', color='Category', orientation='h', 
    #                 title="Distribution of Rhetorical dimension by Source",
    #                 hover_data={
    #                     'Frequency': True,
    #                     'Percentage': ':.2f'  # Format as float with 2 decimal places
    #                 },
    #                 labels={
    #                     'Percentage': 'Percentage of Rhetorical dimension',
    #                     'Frequency': 'Frequency of Rhetorical dimension'
    #                 })

    # # Add axes lines
    # fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    # fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    # st.plotly_chart(fig, use_container_width=True)
    fig = px.bar(grouped_df, 
             x='Percentage', 
             y='source', 
             color='Category', 
             orientation='h',
             title="Distribution of Rhetorical dimension by Source",
             custom_data=['Category', 'Percentage', 'Frequency'])

    fig.update_traces(
        hovertemplate="%{customdata[0]}: %{customdata[1]:.2f}%  (%{customdata[2]:,} times)<extra></extra>"
    )

    # Add axes lines
    fig.update_xaxes(showline=True, 
                    linewidth=2, 
                    linecolor='black', 
                    title='Percentage of Rhetorical dimension')
    fig.update_yaxes(showline=True, 
                    linewidth=2, 
                    linecolor='black', 
                    title='Source')

    st.plotly_chart(fig, use_container_width=True)


    ########################################################################################

    ########################################################################################


    # Calculate total number of articles for each country
    total_articles = country_article_counts_df_subtask3.groupby('country').sum().reset_index()
    total_articles.columns = ['country', 'total_articles']

    # Filter data based on selected countries
    filtered_df = aggregated_df_subtask3.loc[selected_countries]
    # Add a column for the total frequency of persuasion techniques for each country
    filtered_df['total_frequency'] = filtered_df.sum(axis=1)

    # Reset the index to make 'country' a column
    filtered_df = filtered_df.reset_index()
    # Merge this with your DataFrame
    filtered_df = pd.merge(filtered_df, total_articles, on='country', how='left')
    filtered_df = filtered_df.drop_duplicates()

    # Melt the DataFrame to have a separate row for each framing
    melted_df = filtered_df.melt(id_vars=['country', 'total_articles', 'total_frequency'], var_name='Framing', value_name='Frequency')
    # Convert 'Frequency' to integer
    melted_df['Frequency'] = melted_df['Frequency'].astype(int)

    # Remove 'labels.' prefix from the 'Framing' column
    melted_df['Framing'] = melted_df['Framing'].str.replace('labels.', '')

    # Rename 'Framing' to 'Persuasion Techniques'
    melted_df.rename(columns={'Framing': 'Persuasion Techniques'}, inplace=True)
    # Create a new column for the percentage
    # melted_df['Percentage'] = (melted_df['Frequency'] / melted_df['total_frequency']) * 100

    melted_df['Percentage'] = (melted_df['Frequency'] / melted_df['total_articles']) * 100

    # Sort the data by country and frequency (in descending order)
    melted_df.sort_values(by=['country', 'Percentage'], ascending=[True, False], inplace=True)

            # Combine country and total articles
    melted_df['country'] = melted_df['country'] + " (" + melted_df['total_articles'].astype(str) + ")"

    # Sort the data by total articles (in descending order)

    melted_df.sort_values(by=['total_articles', 'Percentage'], ascending=[True, False], inplace=True)

    # Mapping of techniques to categories
    technique_to_category = {
        'Flag_Waving': 'Pathos',
        'Appeal_to_Authority': 'Ethos',
        'Appeal_to_Popularity': 'Ethos',
        'Appeal_to_Values': 'Pathos',
        'Appeal_to_Fear-Prejudice': 'Pathos',
        'Straw_Man': 'Logos',
        'Red_Herring': 'Logos',
        'Whataboutism': 'Logos',
        'Causal_Oversimplification': 'Logos',
        'False_Dilemma-No_Choice': 'Logos',
        'Consequential_Oversimplification': 'Logos',
        'Slogans': 'Other',
        'Conversation_Killer': 'Other',
        'Appeal_to_Time': 'Other',
        'Loaded_Language': 'Pathos',
        'Obfuscation-Vagueness-Confusion': 'Logos',
        'Exaggeration-Minimisation': 'Other',
        'Repetition': 'Other',
        'Name_Calling-Labeling': 'Pathos',
        'Guilt_by_Association': 'Ethos',
        'Casting_Doubt': 'Ethos',
        'Appeal_to_Hypocrisy': 'Ethos',
        'Questioning_the_Reputation': 'Ethos'
    }

    grouped_df = melted_df
    # Add a new column 'Category' to the dataframe
    grouped_df['Category'] = melted_df['Persuasion Techniques'].map(technique_to_category)
        # Group by 'source' and 'Category' and sum the 'Frequency'
    grouped_freq_df = grouped_df.groupby(['country', 'Category'])['Frequency'].sum().reset_index()
    
    # Calculate the total frequency for each source after the techniques are grouped into categories
    total_category_frequency = grouped_freq_df.groupby('country')['Frequency'].sum().reset_index()
    total_category_frequency.columns = ['country', 'total_category_frequency']
    
    # Merge the total frequency dataframe with the grouped frequency dataframe
    grouped_freq_df = pd.merge(grouped_freq_df, total_category_frequency, on='country', how='left')
    
    # Calculate the percentage
    grouped_freq_df['Percentage'] = (grouped_freq_df['Frequency'] / grouped_freq_df['total_category_frequency']) * 100

    grouped_df = grouped_freq_df

    grouped_df['total_articles'] = grouped_df['country'].str.extract(r'\((.*?)\)', expand=False).astype(int)

    grouped_df.sort_values(by=['total_articles', 'Percentage'], ascending=[True, False], inplace=True)

    # Plot the graph using Plotly Express
    # fig = px.bar(grouped_df, x='Percentage', y='country', color='Category', orientation='h', 
    #                 title="Distribution of Rhetorical dimension by Country",
    #                 hover_data={
    #                     'Frequency': True,
    #                     'Percentage': ':.2f'  # Format as float with 2 decimal places
    #                 },
    #                 labels={
    #                     'Percentage': 'Percentage of Rhetorical dimension',
    #                     'Frequency': 'Frequency of Rhetorical dimension'
    #                 })

    # # Add axes lines
    # fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    # fig.update_yaxes(showline=True, linewidth=2, linecolor='black')

    # st.plotly_chart(fig, use_container_width=True)
    
    fig = px.bar(grouped_df, 
             x='Percentage', 
             y='country', 
             color='Category', 
             orientation='h',
             title="Distribution of Rhetorical dimension by Country",
             custom_data=['Category', 'Percentage', 'Frequency'])

    fig.update_traces(
        hovertemplate="%{customdata[0]}: %{customdata[1]:.2f}%  (%{customdata[2]:,} times)<extra></extra>"
    )

    # Add axes lines
    fig.update_xaxes(showline=True, 
                    linewidth=2, 
                    linecolor='black', 
                    title='Percentage of Rhetorical dimension')
    fig.update_yaxes(showline=True, 
                    linewidth=2, 
                    linecolor='black', 
                    title='Country')

    st.plotly_chart(fig, use_container_width=True)


    ########################################################################################


    # Get list of unique countries
    countries = country_article_counts_df_subtask3.index.get_level_values(0).unique()

    # Create buttons for selection
    option = st.selectbox('Choose an option', ('Compare', 'Aggregate'))

    # Add title and labels
    if option == 'Aggregate':
        fig = go.Figure(data=go.Scatter(x=article_counts_df_subtask3.index, 
                                        y=article_counts_df_subtask3['number_of_articles'],
                                        mode='lines'))
        fig.update_layout(title='Number of Articles Over Time (Aggregate) by source',
                        xaxis_title='Date',
                        yaxis_title='Number of Articles')

    else:
        # Convert selected_countries to set and back to list to remove duplicates
        selected_sources = list(set(selected_sources))
        # Filter the dataframe for selected countries
        filtered_df = source_article_counts_subtask3.loc[selected_sources]
        
        fig = go.Figure()
        
        for source in selected_sources:
            source_df = filtered_df.loc[source]
            # Check if returned object is a Series or DataFrame
            if isinstance(source_df, pd.Series):
                x_data = [source_df.name]  # When it's a Series, name attribute gives the index
                y_data = [source_df.values[0]]
            else:
                x_data = source_df.index
                y_data = source_df['number_of_articles']
            fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=source))


        fig.update_layout(title='Number of Articles Over Time (Compare) by source',
                        xaxis_title='Date',
                        yaxis_title='Number of Articles')

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)


    ########################################################################################

        # Add title and labels
    if option == 'Aggregate':
        fig = go.Figure(data=go.Scatter(x=article_counts_df_subtask3.index, 
                                        y=article_counts_df_subtask3['number_of_articles'],
                                        mode='lines'))
        fig.update_layout(title='Number of Articles Over Time (Aggregate) by country',
                        xaxis_title='Date',
                        yaxis_title='Number of Articles')

    else:
        # Convert selected_countries to set and back to list to remove duplicates
        selected_countries = list(set(selected_countries))
        # Filter the dataframe for selected countries
        filtered_df = country_article_counts_df_subtask3.loc[selected_countries]
        
        fig = go.Figure()
        
        for country in selected_countries:
            country_df = filtered_df.loc[country]
            # Check if returned object is a Series or DataFrame
            if isinstance(country_df, pd.Series):
                x_data = [country_df.name]  # When it's a Series, name attribute gives the index
                y_data = [country_df.values[0]]
            else:
                x_data = country_df.index
                y_data = country_df['number_of_articles']
            fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=country))


        fig.update_layout(title='Number of Articles Over Time (Compare) by country',
                        xaxis_title='Date',
                        yaxis_title='Number of Articles')

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

    ########################################################################################


    CONTINENT_MAP = {
        'AF': 'Africa',
        'AS': 'Asia',
        'EU': 'Europe',
        'NA': 'North America',
        'SA': 'South America',
        'OC': 'Oceania',
        'AN': 'Antarctica'
    }

    def get_region(country):
        try:
            country_code = country_name_to_country_alpha2(country)
            continent_code = country_alpha2_to_continent_code(country_code)
            return CONTINENT_MAP[continent_code]
        except KeyError:
            return "Unknown"

    st.title('News Analysis Dashboard')
    st.markdown("""
    Distribution of Persuation Techniques in the Articles across Countries and Regions
    """)


    # Mapping of techniques to categories
    technique_to_category = {
        'labels.Flag_Waving': 'Pathos',
        'labels.Appeal_to_Authority': 'Ethos',
        'labels.Appeal_to_Popularity': 'Ethos',
        'labels.Appeal_to_Values': 'Pathos',
        'labels.Appeal_to_Fear-Prejudice': 'Pathos',
        'labels.Straw_Man': 'Logos',
        'labels.Red_Herring': 'Logos',
        'labels.Whataboutism': 'Logos',
        'labels.Causal_Oversimplification': 'Logos',
        'labels.False_Dilemma-No_Choice': 'Logos',
        'labels.Consequential_Oversimplification': 'Logos',
        'labels.Slogans': 'Other',
        'labels.Conversation_Killer': 'Other',
        'labels.Appeal_to_Time': 'Other',
        'labels.Loaded_Language': 'Pathos',
        'labels.Obfuscation-Vagueness-Confusion': 'Logos',
        'labels.Exaggeration-Minimisation': 'Other',
        'labels.Repetition': 'Other',
        'labels.Name_Calling-Labeling': 'Pathos',
        'labels.Guilt_by_Association': 'Ethos',
        'labels.Doubt': 'Ethos',
        'labels.Appeal_to_Hypocrisy': 'Ethos',
        'labels.Questioning_the_Reputation': 'Ethos'
    }

    # Reset index of the dataframe
    aggregated_df_subtask3.reset_index(level=0, inplace=True)

    # Convert wide data to long data
    long_format_df = pd.melt(aggregated_df_subtask3, id_vars=['country'], var_name='framings', value_name='percentage')

    # Create new 'Category' column
    long_format_df['Category'] = long_format_df['framings'].map(technique_to_category)

    # Group by 'country' and 'Category' and sum the 'percentage' column
    grouped_df = long_format_df.groupby(['country', 'Category']).agg({'percentage': 'sum'}).reset_index()

    # Calculate the cumulative sum of 'percentage' within each country
    grouped_df['percentage_accumulated'] = grouped_df.groupby('country')['percentage'].cumsum()

    # Create new region column
    grouped_df['region'] = grouped_df['country'].apply(get_region)

    grouped_df = grouped_df.drop_duplicates(subset=['region', 'country', 'Category'])

    # Filter dataframe by selected countries
    grouped_df = grouped_df[grouped_df['country'].isin(selected_countries)]

    # # Calculate total frequency of all persuasion techniques for each country
    # total_frequencies = filtered_df.groupby('country')['percentage'].sum().reset_index()
    # total_frequencies.columns = ['country', 'total_frequency']

    # # Merge this with your DataFrame
    # filtered_df = pd.merge(filtered_df, total_frequencies, on='country', how='left')


    # Calculate relative frequency (percentage) for each persuasion technique
    grouped_df['percentage'] = (grouped_df['percentage'] / grouped_df['percentage_accumulated'].max()) * 100

    # Remove rows with zero percentage
    grouped_df = grouped_df[grouped_df['percentage'] > 0]

    fig = px.sunburst(grouped_df, path=['region', 'country',  'Category'], values='percentage', color='percentage')

    fig.update_layout(width=1200, height=1000)

    st.plotly_chart(fig)
