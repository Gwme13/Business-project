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
        
    
    print("Dataset loaded successfully.")
    
    print("Processing the dataset...")    
    
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
    
    print("GeminAI API Integration")
    
    prompt = f""" 
    you are a virtual assistant to the CEO of {company_name}. I will provide you with reviews in text format. Analyze them and I want you to provide:
        1) Main Topics.
        2) Assign a rank of positive or negative to each.
        3) For the formulation of a new strategy in business identify my company's main problems (worst topic) and propose technical-economic feedback.
        
        Provide a detailed analysis.
    """
        
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # write the reviews to a text file
    with open('reviews.txt', 'w') as f:
        f.write('\n'.join(df['review']))
        
      
    sample_file = genai.upload_file(path='reviews.txt', display_name='reviews.txt')

    print(f"Uploaded file '{sample_file.display_name}'")
    
    # delete the uploaded file
    os.remove('reviews.txt')
    
    # Generate content using the uploaded document
    response = model.generate_content([sample_file, prompt])

    # Save the response to a text file
    with open('output.txt', 'w') as f:
        f.write(response.text)
        
    print("Analysis completed.")
    
    print("Generating PDF report...")
    
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
   
    pdf.add_page()
    
   
    pdf.set_font("Arial", size=11)

    # Read the text from the output file and write it to the PDF
    with open('output.txt', 'r', encoding='utf-8') as f:
        for line in f:
            pdf.multi_cell(0, 10, line)  # Allow for multi-line text
            
    
    # delete the output file
    os.remove('output.txt')

    # Next, add the images to the PDF
    pdf.add_page()

    # Add the rating distribution image to the PDF
    image = Image.open('rating_distribution.png')
    image_width, image_height = image.size
    max_width, max_height = 100, 100  # Dimensioni massime nel PDF (in mm)

    
    width_ratio = max_width / image_width
    height_ratio = max_height / image_height
    scale_ratio = min(width_ratio, height_ratio)

    new_width = int(image_width * scale_ratio)
    new_height = int(image_height * scale_ratio)

    
    pdf.image('rating_distribution.png', x=10, y=None, w=new_width, h=new_height)
    
    # Add the word cloud image to the PDF
    image = Image.open('wordcloud.png')
    image_width, image_height = image.size
    max_width, max_height = 100, 100  # Dimensioni massime nel PDF (in mm)
    
    width_ratio = max_width / image_width
    height_ratio = max_height / image_height
    scale_ratio = min(width_ratio, height_ratio)
    
    new_width = int(image_width * scale_ratio)
    new_height = int(image_height * scale_ratio)
    
    pdf.image('wordcloud.png', x=10, y=None, w=new_width, h=new_height)
    
    # Salva il PDF finale
    pdf.output(output_pdf)
        
    



if __name__ == "__main__":
   
    process_dataset_and_get_results("./dataset/MacDonalds_Reviews_Cleaned.csv", "output.pdf", api_key="AIzaSyBzjTSU97Yedj0yo5GDLxuUQVxxCWDunVk", company_name="McDonald's")

