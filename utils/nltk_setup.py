"""NLTK data setup utility."""

def setup_nltk_data():
    """Download required NLTK data."""
    import nltk
    import ssl
    
    # Fix SSL certificate issue if needed
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Download required data
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✓ NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"Warning: Could not download NLTK data: {e}")
        return False


if __name__ == "__main__":
    setup_nltk_data()
