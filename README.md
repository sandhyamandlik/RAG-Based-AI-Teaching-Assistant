# RAG-Based-AI-Teaching-Assistant
The RAG-Based AI Teaching Assistant is an intelligent system that allows users to query and interact with their own educational video content using advanced Retrieval-Augmented Generation (RAG) techniques. It processes videos into searchable knowledge, enabling a personalized AI learning experience.


# How to use this RAG AI Teaching assistant on your own data
## Step 1 - Collect your videos
Move all your video files to the videos folder

## Step 2 - Convert to mp3
Convert all the video files to mp3 by ruunning video_to_mp3

## Step 3 - Convert mp3 to json 
Convert all the mp3 files to json by ruunning mp3_to_json

## Step 4 - Convert the json files to Vectors
Use the file preprocess_json to convert the json files to a dataframe with Embeddings and save it as a joblib pickle

## Step 5 - Prompt generation and feeding to LLM

Read the joblib file and load it into the memory. Then create a relevant prompt as per the user query and feed it to the LLM
