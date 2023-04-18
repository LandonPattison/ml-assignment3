import pandas as pd
import re

# Read the CSV file
data = pd.read_csv("Health-Tweets/usnewshealth.txt", sep='|', header=None, names=["tweet_id", "timestamp", "text"])

# Remove the tweet id and timestamp
data = data.drop(columns=["tweet_id", "timestamp"])

def preprocess_text(text):
    # Remove any word that starts with the symbol @
    text = re.sub(r'@\w+', '', text)
    
    # Remove any URL
    text = re.sub(r'http\S+', '', text)
    
    # Convert every word to lowercase
    text = text.lower()
    
    # Remove any hashtag symbols
    text = re.sub(r'#', '', text)
    
    return text

# Apply the preprocessing function to each row
data['text'] = data['text'].apply(preprocess_text)

# Save the processed data to a new CSV file
data.to_csv("preprocessed_data.csv", index=False)
