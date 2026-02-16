import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Downlaoad resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

df = pd.read_csv("data/merged/merged_dataset.csv")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word)
              for word in tokens]

    # Join back into string
    return ' '.join(tokens)


# Apply to a DataFrame column
df["clean_content"] = df["content"].apply(clean_text)

# Count words by class
fake_words = Counter(" ".join(df[df["label"] == 0]["clean_content"]).split())
real_words = Counter(" ".join(df[df["label"] == 1]["clean_content"]).split())

# Identify exclusive words
exclusive_fake = {w: c for w, c in fake_words.items() if w not in real_words}
exclusive_real = {w: c for w, c in real_words.items() if w not in fake_words}

print("Exclusive Fake sample:", list(exclusive_fake)[:20])
print("Exclusive Real sample:", list(exclusive_real)[:20])


# Remove exclusive and rare words
def remove_exclusive_and_rare_words(text, rare_threshold=2):
    words = text.split()
    words = [w for w in words if w not in exclusive_fake and w not in exclusive_real]
    counts = Counter(words)
    words = [w for w in words if counts[w] >= rare_threshold]
    return " ".join(words)


df["clean_content"] = df["clean_content"].apply(
    remove_exclusive_and_rare_words)

# Save processed data for model training
df.to_csv("data/preprocessed/preprocessed_dataset.csv", index=False)
