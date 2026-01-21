"""Text preprocessing for data preparation."""

import re
from typing import List, Optional
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class TextPreprocessor:
    """Preprocessor for text data."""
    
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
    
    def clean(self, text: str) -> str:
        """Clean text data."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens."""
        return [token for token in tokens if token not in self.stopwords]
    
    def preprocess(self, text: str) -> List[str]:
        """Complete preprocessing pipeline."""
        text = self.clean(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        return tokens
