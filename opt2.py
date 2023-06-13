import pandas as pd
import json
import pyarrow.parquet as pq
import pyarrow as pa

# Function to read the data in chunks
def chunked_dataframe(file, chunk_size=10000):
    data = []
    with open(file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            if len(data) == chunk_size:
                yield pd.json_normalize(data)  # Normalize the nested JSON structure
                data = []
        if data:
            yield pd.json_normalize(data)  # Normalize the remaining data

# Use the function to process the data
file = 'merged_subtask3.jsonl'  # Update the filename
df_chunks = chunked_dataframe(file)

# Get the schema from the first chunk
first_chunk = next(df_chunks)

table = pa.Table.from_pandas(first_chunk)
parquet_schema = table.schema

# Create a ParquetWriter
writer = pq.ParquetWriter('subtask3.parquet', parquet_schema)  # Update the output filename

# Write the first chunk
first_chunk.drop_duplicates(subset='id', inplace=True)
table = pa.Table.from_pandas(first_chunk, schema=parquet_schema)
writer.write_table(table)

# Iterate over the remaining chunks, remove duplicates and write to the parquet file
for df in df_chunks:
    df.drop_duplicates(subset='id', inplace=True)
    table = pa.Table.from_pandas(df, schema=parquet_schema)
    writer.write_table(table)

# Close the ParquetWriter
writer.close()



# import pandas as pd
# import json
# from tqdm import tqdm

# # Create a list to hold the JSON data
# json_data = []

# # Read JSONL file line by line
# with open('subtask3.jsonl', 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         json_data.append(data)

# # Convert the list of dictionaries to DataFrame
# json_df = pd.DataFrame(json_data)

# # Define a chunk size (adjust based on your memory availability)
# chunk_size = 100000

# # Calculate total number of chunks for tqdm progress bar
# total_chunks = sum(1 for row in open('eu_data.csv')) // chunk_size + 1

# # Read the CSV data in chunks
# csv_chunk = pd.read_csv('eu_data.csv', chunksize=chunk_size, usecols=['id', 'source', 'country', 'pubDate'])

# # Create a list to hold chunks of merged data
# merged_data = []

# # Iterate over chunks with tqdm for progress bar
# for chunk in tqdm(csv_chunk, total=total_chunks):
#     # Merge the current chunk with the JSON data
#     merged_chunk = pd.merge(json_df, chunk, on='id')

#     # Append the merged chunk to the merged_data list
#     merged_data.append(merged_chunk)

# # Concatenate all the chunks into a DataFrame
# merged_df = pd.concat(merged_data)

# # Save merged DataFrame to a new JSONL file
# merged_df.to_json('merged_subtask3.jsonl', orient='records', lines=True)

