import re
import string
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Remove negation words from stopwords as they're important for mental health
        self.stop_words -= {'no', 'not', 'nor', 'don', "don't", 'ain', 'aren', "aren't", 
                           'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
                           'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
                           'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't",
                           'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
                           'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
                           'wouldn', "wouldn't"}
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags (but keep the word)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove numbers (but keep if part of words like "24/7")
        text = re.sub(r'\b\d+\b', '', text)
        
        # Keep punctuation that matters for sentiment
        # Remove other punctuation
        text = re.sub(r'[^\w\s!?.,-]', '', text)
        
        # Handle repeated characters (but keep some emphasis)
        # e.g., "soooo" -> "soo", "noooo" -> "noo"
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        return text.strip()
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text while preserving important words"""
        # Tokenize
        words = text.split()
        
        # Lemmatize and filter stopwords
        filtered_words = []
        for word in words:
            # Skip very short words unless they're important
            if len(word) < 2 and word not in ['i']:
                continue
            # Remove stopwords but keep negations and emotional words
            if word not in self.stop_words or word in ['i', 'me', 'my', 'myself']:
                lemma = self.lemmatizer.lemmatize(word)
                filtered_words.append(lemma)
        
        return ' '.join(filtered_words)
    
    def preprocess(self, text):
        """Full preprocessing pipeline"""
        cleaned = self.clean_text(text)
        processed = self.tokenize_and_lemmatize(cleaned)
        return processed if processed else cleaned  # Fallback to cleaned if empty


def load_and_preprocess_data(file_path, preprocess_text=True):
    """
    Load and preprocess the mental health dataset
    
    Args:
        file_path: Path to the CSV file
        preprocess_text: Whether to apply text preprocessing
        
    Returns:
        texts: Processed text data
        labels: Encoded labels
        label_encoder: Fitted label encoder
        df: Original dataframe with preprocessing
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Remove any rows with missing values
    df.dropna(subset=['statement', 'status'], inplace=True)
    
    # Remove duplicates
    df.drop_duplicates(subset=['statement'], inplace=True)
    
    # Convert to string
    df['statement'] = df['statement'].astype(str)
    df['status'] = df['status'].astype(str)
    
    # Apply preprocessing if requested
    if preprocess_text:
        print("Preprocessing text...")
        preprocessor = TextPreprocessor()
        df['processed_text'] = df['statement'].apply(preprocessor.preprocess)
        # Use processed text
        texts = df['processed_text']
    else:
        texts = df['statement']
    
    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(df['status'])
    
    # Print class distribution
    print("\nClass Distribution:")
    class_counts = pd.Series(labels).value_counts().sort_index()
    for idx, count in class_counts.items():
        print(f"{le.classes_[idx]}: {count} samples ({count/len(labels)*100:.2f}%)")
    
    return texts, labels, le, df


def augment_minority_classes(texts, labels, label_encoder, augment_to='mean', max_augment_ratio=2):
    """
    Augment minority classes by creating variations of existing texts
    
    Args:
        texts: Text data (pandas Series or list)
        labels: Label array
        label_encoder: Fitted label encoder
        augment_to: 'max' (to match largest class) or 'mean' (to match average)
        max_augment_ratio: Maximum times to augment any single class
        
    Returns:
        augmented_texts (pandas Series), augmented_labels (numpy array)
    """
    from collections import Counter
    import random
    
    # Convert to list for easier manipulation
    texts_list = list(texts) if not isinstance(texts, list) else texts
    labels_list = list(labels) if not isinstance(labels, list) else labels
    
    # Count samples per class
    class_counts = Counter(labels_list)
    
    if augment_to == 'max':
        target_count = max(class_counts.values())
    else:  # mean
        target_count = int(np.mean(list(class_counts.values())))
    
    print("\nAugmenting minority classes...")
    print(f"Target count per class: {target_count}")
    
    # Start with copies of original data
    augmented_texts = texts_list.copy()
    augmented_labels = labels_list.copy()
    
    for class_idx, count in class_counts.items():
        # Only augment if below target
        if count < target_count:
            # Don't augment more than max_augment_ratio times
            samples_needed = min(target_count - count, count * max_augment_ratio)
            
            # Get all texts from this class
            class_texts = [text for text, label in zip(texts_list, labels_list) if label == class_idx]
            
            # Randomly sample with replacement
            augmented_samples = random.choices(class_texts, k=samples_needed)
            
            # Add to lists
            augmented_texts.extend(augmented_samples)
            augmented_labels.extend([class_idx] * samples_needed)
            
            print(f"{label_encoder.classes_[class_idx]}: {count} -> {count + samples_needed}")
    
    # Return as pandas Series and numpy array to avoid memory issues
    return pd.Series(augmented_texts), np.array(augmented_labels)