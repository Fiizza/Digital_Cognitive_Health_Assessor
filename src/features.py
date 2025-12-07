import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle


class MentalHealthFeatureExtractor:
    """
    Enhanced feature extraction for mental health text classification
    """
    
    def __init__(self, 
                 use_tfidf=True,
                 ngram_range=(1, 3),  # Increased to capture phrases
                 max_features=50000,
                 min_df=2,  # Reduced to capture rare mental health terms
                 max_df=0.8,  # Reduced to remove very common words
                 use_svd=True,
                 svd_components=400):  # Increased for better representation
        
        self.use_tfidf = use_tfidf
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.use_svd = use_svd
        self.svd_components = svd_components
        
        # Initialize vectorizer
        if use_tfidf:
            self.vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                sublinear_tf=True,  # Apply sublinear scaling
                strip_accents='unicode',
                analyzer='word',
                token_pattern=r'\w{1,}',  # Keep single letters like 'i'
                use_idf=True,
                smooth_idf=True
            )
        else:
            self.vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_df,
                max_df=max_df
            )
        
        # Initialize SVD if needed
        self.svd = None
        if use_svd and svd_components:
            self.svd = TruncatedSVD(
                n_components=svd_components,
                random_state=42,
                n_iter=10
            )
    
    def fit_transform(self, texts):
        """
        Fit the vectorizer and transform texts
        
        Args:
            texts: List or array of text strings
            
        Returns:
            Transformed feature matrix (sparse or dense)
        """
        # Vectorize
        X = self.vectorizer.fit_transform(texts)
        
        print(f"Vectorized shape: {X.shape}")
        print(f"Features: {len(self.vectorizer.get_feature_names_out())} terms")
        
        # Apply SVD if enabled
        if self.svd is not None:
            X = self.svd.fit_transform(X)
            explained_var = self.svd.explained_variance_ratio_.sum()
            print(f"SVD reduced to {X.shape[1]} components")
            print(f"Explained variance: {explained_var:.4f}")
        
        return X
    
    def transform(self, texts):
        """
        Transform texts using fitted vectorizer
        
        Args:
            texts: List or array of text strings
            
        Returns:
            Transformed feature matrix
        """
        X = self.vectorizer.transform(texts)
        
        if self.svd is not None:
            X = self.svd.transform(X)
        
        return X
    
    def save(self, vectorizer_path, svd_path=None):
        """Save the fitted vectorizer and SVD"""
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        if self.svd is not None and svd_path:
            with open(svd_path, 'wb') as f:
                pickle.dump(self.svd, f)
    
    def load(self, vectorizer_path, svd_path=None):
        """Load a fitted vectorizer and SVD"""
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        if svd_path:
            with open(svd_path, 'rb') as f:
                self.svd = pickle.load(f)
    
    def get_top_features_per_class(self, X, y, label_encoder, top_n=20):
        """
        Get top features for each class
        
        Args:
            X: Feature matrix
            y: Labels
            label_encoder: Fitted label encoder
            top_n: Number of top features to return per class
            
        Returns:
            Dictionary mapping class names to top features
        """
        if self.svd is not None:
            print("Cannot extract feature names after SVD transformation")
            return {}
        
        feature_names = self.vectorizer.get_feature_names_out()
        top_features = {}
        
        for class_idx, class_name in enumerate(label_encoder.classes_):
            # Get class mask
            mask = y == class_idx
            
            # Calculate mean TF-IDF for this class
            if hasattr(X, 'toarray'):
                class_tfidf = X[mask].mean(axis=0).A1
            else:
                class_tfidf = X[mask].mean(axis=0)
            
            # Get top features
            top_indices = np.argsort(class_tfidf)[-top_n:][::-1]
            top_features[class_name] = [
                (feature_names[i], class_tfidf[i]) 
                for i in top_indices
            ]
        
        return top_features


def extract_features(texts_train, texts_test=None, 
                    use_tfidf=True, 
                    use_svd=True,
                    ngram_range=(1, 3),
                    max_features=50000,
                    svd_components=400):
    """
    Extract features from text data
    
    Args:
        texts_train: Training texts
        texts_test: Test texts (optional)
        use_tfidf: Use TF-IDF (True) or Count vectorization (False)
        use_svd: Apply SVD dimensionality reduction
        ngram_range: N-gram range for vectorization
        max_features: Maximum number of features
        svd_components: Number of SVD components
        
    Returns:
        X_train, X_test (if provided), feature_extractor
    """
    extractor = MentalHealthFeatureExtractor(
        use_tfidf=use_tfidf,
        ngram_range=ngram_range,
        max_features=max_features,
        use_svd=use_svd,
        svd_components=svd_components
    )
    
    X_train = extractor.fit_transform(texts_train)
    
    if texts_test is not None:
        X_test = extractor.transform(texts_test)
        return X_train, X_test, extractor
    
    return X_train, extractor