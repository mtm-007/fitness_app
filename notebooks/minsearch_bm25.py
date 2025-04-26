import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np
import re

class Index:
    """
    A simple search index using BM25 ranking for text fields and exact matching for keyword fields.

    Attributes:
        text_fields (list): List of text field names to index.
        keyword_fields (list): List of keyword field names to index.
        bm25_models (dict): Dictionary of BM25Okapi instances for each text field.
        corpus_tokens (dict): Dictionary of tokenized documents for each text field.
        keyword_df (pd.DataFrame): DataFrame containing keyword field data.
        docs (list): List of documents indexed.
    """

    def __init__(self, text_fields, keyword_fields):
        """
        Initializes the Index with specified text and keyword fields.

        Args:
            text_fields (list): List of text field names to index.
            keyword_fields (list): List of keyword field names to index.
        """
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields
        self.bm25_models = {}  # BM25Okapi instances per field
        self.corpus_tokens = {}  # Tokenized documents per field
        self.keyword_df = None
        self.docs = []
    
    def tokenize(self, text):
        """
        Tokenize text by splitting on whitespace and removing punctuation.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            list: List of tokens
        """
        if not text:
            return []
        # Convert to lowercase and split on non-alphanumeric characters
        return re.findall(r'\w+', text.lower())
        
    def fit(self, docs):
        """
        Fits the index with the provided documents.

        Args:
            docs (list of dict): List of documents to index. Each document is a dictionary.
            
        Returns:
            Index: The fitted index instance
        """
        self.docs = docs
        keyword_data = {field: [] for field in self.keyword_fields}
        
        # Process text fields with BM25
        for field in self.text_fields:
            texts = [doc.get(field, '') for doc in docs]
            tokens = [self.tokenize(text) for text in texts]
            self.corpus_tokens[field] = tokens
            
            # Only create BM25 model if we have valid tokens
            if any(len(t) > 0 for t in tokens):
                self.bm25_models[field] = BM25Okapi(tokens)
        
        # Process keyword fields
        for doc in docs:
            for field in self.keyword_fields:
                keyword_data[field].append(doc.get(field, ''))
                
        self.keyword_df = pd.DataFrame(keyword_data)
        return self
        
    def search(self, query, filter_dict={}, boost_dict={}, num_results=10):
        """
        Searches the index with the given query, filters, and boost parameters.

        Args:
            query (str): The search query string.
            filter_dict (dict): Dictionary of keyword fields to filter by. Keys are field names and values are the values to filter by.
            boost_dict (dict): Dictionary of boost scores for text fields. Keys are field names and values are the boost scores.
            num_results (int): The number of top results to return. Defaults to 10.

        Returns:
            list of dict: List of documents matching the search criteria, ranked by relevance.
        """
        if not query or not self.docs:
            return []
            
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []
            
        scores = np.zeros(len(self.docs))
        
        # Calculate BM25 scores for each text field and apply boost
        for field in self.text_fields:
            if field in self.bm25_models and query_tokens:
                sim = self.bm25_models[field].get_scores(query_tokens)
                boost = boost_dict.get(field, 1.0)
                scores += sim * boost
        
        # Apply keyword filters
        for field, value in filter_dict.items():
            if field in self.keyword_fields:
                mask = self.keyword_df[field] == value
                scores = scores * mask.to_numpy()
        
        # Get top results with scores > 0
        non_zero_indices = np.where(scores > 0)[0]
        if len(non_zero_indices) <= num_results:
            # If we have fewer results than requested, return all non-zero ones
            top_indices = non_zero_indices[np.argsort(-scores[non_zero_indices])]
        else:
            # Otherwise get the top num_results
            top_indices = np.argpartition(scores, -num_results)[-num_results:]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
        
        top_docs = [self.docs[i] for i in top_indices]
        return top_docs