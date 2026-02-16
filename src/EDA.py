import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
from wordcloud import WordCloud

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Load Dataset
df = pd.read_csv("data/merged/merged_dataset.csv")
print("Dataset shape:", df.shape)
print(df.head())

# Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df)
plt.xticks([0, 1], ['Fake', 'Real'])
plt.title("Class Distribution")
plt.xlabel("Label (0=Fake, 1=Real)")
plt.ylabel("Count")

total = len(df)
for p in plt.gca().patches:
    count = int(p.get_height())
    percent = 100 * count / total
    plt.text(p.get_x() + p.get_width()/2., p.get_height()/2,
             f'{count}\n({percent:.1f}%)',
             ha="center", va="center", color="black", fontsize=10, fontweight="bold")
plt.show()

print("Class counts:\n", df['label'].value_counts())

# Text Length Analysis
df['text_length'] = df['content'].astype(str).apply(len)
print("\nText length stats:\n", df['text_length'].describe())

plt.figure(figsize=(6, 4))
sns.histplot(df['text_length'], bins=50, kde=True)
plt.title("Distribution of Text Lengths")
plt.xlabel("Number of characters")
plt.show()

# Word Count Analysis
df['word_count'] = df['content'].astype(str).apply(lambda x: len(x.split()))
print("\nWord count stats:\n", df['word_count'].describe())

plt.figure(figsize=(6, 4))
sns.histplot(df['word_count'], bins=50, kde=True)
plt.title("Distribution of Word Counts")
plt.xlabel("Number of words")
plt.show()

# Top Words in Entire Dataset
all_text = ' '.join(df['content'].astype(str))
tokens = re.findall(r'\b\w+\b', all_text.lower())
tokens = [t for t in tokens if t not in stop_words]

top_words = Counter(tokens).most_common(20)
print("\nTop 20 words:\n", top_words)

words, counts = zip(*top_words)
plt.figure(figsize=(8, 4))
sns.barplot(x=list(counts), y=list(words))
plt.title("Top 20 Words in Dataset")
plt.show()

# Overall Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color='white',
                      stopwords=stop_words, max_words=100).generate(all_text)

plt.figure(figsize=(8, 4))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Dataset")
plt.show()

# Comparison: Fake vs Real
df_fake = df[df['label'] == 0]
df_real = df[df['label'] == 1]

# Text length comparison
plt.figure(figsize=(8, 4))
sns.kdeplot(df_fake['text_length'], label='Fake', fill=True)
sns.kdeplot(df_real['text_length'], label='Real', fill=True)
plt.title("Text Length Distribution: Fake vs Real")
plt.xlabel("Number of characters")
plt.legend()
plt.show()

# Word count comparison
plt.figure(figsize=(8, 4))
sns.kdeplot(df_fake['word_count'], label='Fake', fill=True)
sns.kdeplot(df_real['word_count'], label='Real', fill=True)
plt.title("Word Count Distribution: Fake vs Real")
plt.xlabel("Number of words")
plt.legend()
plt.show()


# Class-specific Top Words
def top_words_class(df_class, stop_words, n=20):
    text = ' '.join(df_class['content'].astype(str))
    tokens = re.findall(r'\b\w+\b', text.lower())
    tokens = [t for t in tokens if t not in stop_words]
    return Counter(tokens).most_common(n)


print("\nTop 20 words in Fake news:\n", top_words_class(df_fake, stop_words))
print("\nTop 20 words in Real news:\n", top_words_class(df_real, stop_words))


# Class-specific Word Clouds
def generate_wordcloud(text_series, stop_words, title):
    text = ' '.join(text_series.astype(str))
    tokens = re.findall(r'\b\w+\b', text.lower())
    tokens = [t for t in tokens if t not in stop_words]
    cleaned_text = ' '.join(tokens)

    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          stopwords=stop_words, max_words=100).generate(cleaned_text)

    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=20)
    plt.show()


# Fake vs Real Word Clouds
generate_wordcloud(df_fake['content'], stop_words, "Top Words in Fake News")
generate_wordcloud(df_real['content'], stop_words, "Top Words in Real News")
