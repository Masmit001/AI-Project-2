import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import numpy as np
import string
import os


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) 
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

def load_faqs(filepath='faqs.json'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FAQs file not found at {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess_text(text):
    text = text.lower()
    
    text = "".join([char for char in text if char not in string.punctuation])
    
    tokens = nltk.word_tokenize(text)
    
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized)

class FAQChatbot:
    def __init__(self, faqs_path='faqs.json'):
        self.faqs = load_faqs(faqs_path)
        self.questions = [faq['question'] for faq in self.faqs]
        self.answers = [faq['answer'] for faq in self.faqs]
        
        
        self.processed_questions = [preprocess_text(q) for q in self.questions]
        
        
        self.vectorizer = TfidfVectorizer()
        
        
        if self.processed_questions:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_questions)
            
    def get_response(self, user_query, threshold=0.2):
        processed_query = preprocess_text(user_query)
        
        query_vec = self.vectorizer.transform([processed_query])
        
        
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)
        
        
        best_match_idx = np.argmax(similarities)
        best_score = similarities[0, best_match_idx]
        
        
        print(f"Query: '{user_query}' | Best match: '{self.questions[best_match_idx]}' (Score: {best_score:.2f})")
        
        if best_score > threshold:
            return self.answers[best_match_idx]
        else:
            return "I'm sorry, I couldn't understand your question. Could you try rephrasing it or asking something else?"

if __name__ == "__main__":
    # Test
    bot = FAQChatbot()
    print("Bot is ready. Testing query 'How do I return something?'")
    print("Response:", bot.get_response("How do I return something?"))
