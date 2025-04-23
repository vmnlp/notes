import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
# --- POS Tagging using NLTK ---
def pos_tag_nltk(text):
    words = word_tokenize(text)
    tagged = pos_tag(words)
    return tagged

# --- TF-IDF Vectorization ---
def get_tfidf_embeddings(sentence):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sentence])
    return vectorizer, tfidf_matrix

# --- Main Program ---
if __name__ == "__main__":
    user_text = input("Enter a sentence: ")

    print("\nPOS Tagging with NLTK:")
    print(pos_tag_nltk(user_text))

    # Get TF-IDF embeddings
    vectorizer, tfidf_matrix = get_tfidf_embeddings(user_text)

    # Example word lookup
    word = input("\nEnter a word from the sentence to get its TF-IDF value: ").strip().lower()

    # Get the index of the word and its TF-IDF value
    words = vectorizer.get_feature_names_out()
    if word in words:
        word_index = words.tolist().index(word)
        word_tfidf = tfidf_matrix[0, word_index]
        print(f"\nTF-IDF value for '{word}': {word_tfidf}")
    else:
        print(f"\n'{word}' not found in the sentence. Try a different word.")
