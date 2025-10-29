import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation

# one-time setup
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop = set(stopwords.words('english')) - {"no", "not", "nor", "n't"}
wnl = WordNetLemmatizer()

def clean_lemmatize(text):
    if not isinstance(text, str):
        return ""
    
    # lowercase + remove non-letters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # tokenize
    tokens = word_tokenize(text)
    
    # remove stopwords + punctuation
    tokens = [t for t in tokens if t not in stop and t not in punctuation]
    
    # lemmatize
    lemmas = [wnl.lemmatize(t) for t in tokens]
    
    # return joined string
    return " ".join(lemmas)



sample = "The cats aren't running quickly, but they're not lazy either!"
print(clean_lemmatize(sample))


