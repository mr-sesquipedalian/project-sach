# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np

# Data collection and cleaning
splits1 = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}
df1 = pd.read_parquet("hf://datasets/Yashaswat/Indian-Legal-Text-ABS/" + splits1["train"])
df1.rename(columns={'judgement': 'case', 'summary': 'judgement'}, inplace=True)
print(df1.columns)

splits2 = {'train': 'train.jsonl.xz', 'test': 'test.jsonl.xz'}
df2 = pd.read_json("hf://datasets/joelniklaus/legal_case_document_summarization/" + splits2["train"], lines=True)
df2 = df2[df2['dataset_name'] != 'UK-Abs']
df2 = df2.drop('dataset_name', axis = 1)
df2.rename(columns={'judgement': 'case', 'summary': 'judgement'}, inplace=True)
print(df2.columns)

df3 = pd.read_json("hf://datasets/Sahil2507/Indian_Legal_Dataset/legal_DataSet.jsonl", lines=True)
df3['case'] = df3['instruction']+df3['input']
df3 = df3.drop(['prompt', 'text', 'instruction', 'input'], axis = 1)
df3 = df3[['case', 'output']]
df3.rename(columns={'output': 'judgement'}, inplace=True)
print(df3.columns)

# Step 1: Concatenate the datasets
df_combined = pd.concat([df1, df2, df3], ignore_index=True)

# Step 2: Remove duplicates
df_combined.drop_duplicates(inplace=True)

# Step 3: Convert all text columns to lowercase
for col in df_combined.select_dtypes(include=['object']).columns:
    df_combined[col] = df_combined[col].str.lower()

# Step 4: Remove non-ASCII characters, `\n`, `\t`, and other garbage values
def clean_text(text):
    if isinstance(text, str):
        text = text.replace('\n', ' ').replace('\t', ' ')  # Remove newlines and tabs
        text = ''.join(char for char in text if char.isascii())  # Remove non-ASCII characters
        text = ' '.join(text.split())  # Remove extra spaces
    return text

df_combined = df_combined.applymap(clean_text)

# Step 5: Drop rows with missing values
df_combined.dropna(inplace=True)
df_combined.drop_duplicates(inplace=True)
print(df_combined.shape)


# Breaking data into chunks

def split_text_by_word_count(text, max_words=7000):
    words = text.split()  # Split text into words
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks

# Apply to columns
df_chunks = pd.DataFrame()
# Apply the splitting function to the columns
df_chunks['case_chunks'] = df_combined['case'].apply(split_text_by_word_count)
df_chunks['judgement_chunks'] = df_combined['judgement'].apply(split_text_by_word_count)

def flatten_chunks(row, column):
    return [{'original_index': row.name, 'chunk': chunk} for chunk in row[column]]

flattened_data = []
for column in ['case_chunks', 'judgement_chunks']:
    flattened_data.extend(df_chunks.apply(flatten_chunks, column=column, axis=1).sum())

# Create a new DataFrame with flattened data for easier processing
chunked_df = pd.DataFrame(flattened_data)

# Display the result
chunked_df.shape

# Gemini API running on chunked data
import os
import re
from openai import OpenAI
import google.generativeai as palm
import time
from tqdm import tqdm


# Configure the API key for PaLM
API_KEY = "<API KEY>"  # Replace with your valid API key
palm.configure(api_key=API_KEY)

generation_config = palm.GenerationConfig(temperature=0.5)
model = palm.GenerativeModel("gemini-pro") #change here for Gemini Flash

# Function to send a prompt to the Gemini (PaLM) API
def get_gemini_response(prompt):
    start = '''You are given a prompt that take a chunk from the case+judgement given by a judicial court in india. Your 
    task is to read through the entire prompt and give the citations from that judgement those citations can be 
    1. In case you are putting a section/article like article 15 also put the source like section 15 of constitution of India.
    2. references of past cases
    3. do not repeat any citations, give me a unique python list only
    You need to put all these in a python list and give an output that will look like this ['Section 120 B, Indian Penal Code']
    if it is not like this just return 0 do not throw any error. Here article 10 is given as an example do not put it in all of your responses'''
    try:
        # Generate a response using the text generation model
        response = model.generate_content(start+prompt, generation_config=generation_config)
        return response.text  # Retrieve the text result from the response
    except Exception as e:
        return f"Error interacting with Gemini API: {e}"

# Initialize the 'citations' column in the DataFrame if not already present
if "citations" not in chunked_df.columns:
    chunked_df["citations"] = None


# Assuming chunked_df is already created and available
batch_size = 5
skip_batch = 4919
till = 5838

# Function to process rows in batches
def process_batches(chunked_df, batch_size, output_prefix="rows"):
    # Determine the number of batches
    num_batches = (len(chunked_df) + batch_size - 1) // batch_size

    for batch_num in tqdm(range(num_batches),desc="Processing Batches"):

        if batch_num <= skip_batch:
            pass
        elif batch_num > skip_batch and  batch_num < till:
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(chunked_df))
            
            # Slice the DataFrame for the current batch
            batch_df = chunked_df.iloc[start_idx:end_idx].copy()
            
            # Process each row in the batch
            for index, row in batch_df.iterrows():
                user_input = row["chunk"]
                citation_output = get_gemini_response(user_input)
                
                # Clean the response text to get citations
                #citation_output = [line.strip('- ').strip() for line in citation_output.split('\n') if line.strip()]
                start = citation_output.find('[')
                end = citation_output.find(']')
                n = len(citation_output)
                print(citation_output[start:end+1])
                batch_df.at[index, "citations"] = citation_output[start:end+1] if n > 1 else None
                
                time.sleep(5)  # Respect API rate limits
                
            
            # Remove rows where the 'citations' column is empty
            #batch_df = batch_df[batch_df["citations"].notnull()]
            
            # Save the processed batch to a CSV file
            output_file = f"{output_prefix}_{start_idx + 1}_{end_idx}.csv"
            
            # Use a relative path for citations directory
            output_dir = '<put your own path>'
            
            # Ensure the directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save the file to the citations directory
            batch_df.to_csv(os.path.join(output_dir, output_file), index=False)        
            print(f"Saved batch {batch_num + 1} to {output_file}")

# Call the function to process the DataFrame in batches
process_batches(chunked_df, batch_size, output_prefix="rows")
# -








