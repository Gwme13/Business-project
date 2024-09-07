import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import contractions
from fpdf import FPDF
from PIL import Image
import nltk
import google.generativeai as genai
import filters

nltk.download('punkt')

from nltk.corpus import stopwords


stopwords_en = stopwords.words('english')

nltk.download('wordnet')

from nltk.stem.wordnet import WordNetLemmatizer
    

def process_dataset_and_get_results(dataset_path, output_pdf, api_key=None, company_name=None):
    
    
    
    df = {}
    
    # try to read the dataset with the 'utf-8' encoding
    try:
        df = pd.read_csv(dataset_path, encoding='utf-8')
    
    except Exception as e:
        # if an error occurs, try reading the dataset with 'latin1' encoding
        try:
            
            df = pd.read_csv(dataset_path, encoding='latin1')
        except Exception as e:
            print("Error reading the dataset:", e)
            return
        
    

    # check if the 'review' and 'rating' columns are present in the dataset
    if 'review' not in df.columns or 'rating' not in df.columns:
        print("Error: 'review' or 'rating' columns not found in the dataset.")
        return
    
    # check if the 'company_name' is provided
    if company_name is None:
        print("Error: 'company_name' is required.")
        return
    
    # check if the 'api_key' is provided
    if api_key is None:
        print("Error: 'api_key' is required.")
        
        
    # Category distribution
    plt.figure(figsize=(8, 8))
    sns.countplot(x='rating', data=df, order=df['rating'].value_counts().index, stat='percent', palette='viridis')
    plt.title('Rating Distribution')
    plt.ylabel('Percentage')
    plt.xlabel('Rating')
    plt.tight_layout()
    plt.savefig('rating_distribution.png', format='png', bbox_inches='tight')
    plt.show()
    
    df['review'] = df['review'].str.lower()

    df['review'] = df['review'].apply(filters.filter_string)
    
    # fix contractions
    df['review'] = df['review'].apply(contractions.fix)
    
    # Tokenization of the whole text
    # the output is a list, where each element is a token of the original text

    df['tokenized_review'] = df['review'].apply(lambda text: nltk.word_tokenize(text))

    # remove the stopwords
    df['tokenized_review'] = df['tokenized_review'].apply(lambda tokens: [token for token in tokens if token not in stopwords_en])

    # lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    
    df['tokenized_review'] = df['tokenized_review'].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

    # Wordcloud
    
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords_en, 
                min_font_size = 10).generate(' '.join(df['tokenized_review'].apply(lambda tokens: ' '.join(tokens))))
    
    # Plot the word cloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in Reviews')
    
    plt.savefig('wordcloud.png', format='png', bbox_inches='tight')
  
    plt.show()
        
    



if __name__ == "__main__":
   
    process_dataset_and_get_results("./dataset/MacDonalds_Reviews_Cleaned.csv", "output.pdf", api_key="API_KEY", company_name="Company Name")

