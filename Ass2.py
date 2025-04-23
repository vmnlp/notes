from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

eng_stemmer = SnowballStemmer("english")
ger_stemmer = SnowballStemmer("german")

eng_word = ["running", "ran", "eats", "eating", "laughing", "easily"]

eng_stemmed_words = [eng_stemmer.stem(word) for word in eng_word]
print("English stemmed words:")
print(eng_stemmed_words)

german_word = ["laufst", "gelaufen", "essen", "gegessen"]
german_stemmed_words = [ger_stemmer.stem(word) for word in german_word]
print("German stemmed words:")
print(german_stemmed_words)