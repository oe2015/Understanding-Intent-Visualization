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
                yield pd.DataFrame(data)
                data = []
        if data:
            yield pd.DataFrame(data)

# Use the function to process the data
file = 'newfile.jsonl'
df_chunks = chunked_dataframe(file)

# Get the schema from the first chunk
first_chunk = next(df_chunks)

# Convert labels to string
first_chunk['labels'] = first_chunk['labels'].apply(json.dumps)

table = pa.Table.from_pandas(first_chunk)
parquet_schema = table.schema

# Create a ParquetWriter
writer = pq.ParquetWriter('newfile.parquet', parquet_schema)

# Write the first chunk
first_chunk.drop_duplicates(subset='id', inplace=True)
table = pa.Table.from_pandas(first_chunk, schema=parquet_schema)
writer.write_table(table)

# Iterate over the remaining chunks, remove duplicates and write to the parquet file
for df in df_chunks:
    # Convert labels to string
    df['labels'] = df['labels'].apply(json.dumps)
    
    df.drop_duplicates(subset='id', inplace=True)
    table = pa.Table.from_pandas(df, schema=parquet_schema)
    writer.write_table(table)

# Close the ParquetWriter
writer.close()

