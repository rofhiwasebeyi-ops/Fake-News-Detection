import pandas as pd

# Fake and Real News Dataset
df_fake_real_fake = pd.read_csv("data/raw/Fake.csv")
df_fake_real_real = pd.read_csv("data/raw/True.csv")

df_fake_real_fake = df_fake_real_fake[['text']].rename(
    columns={'text': 'content'})
df_fake_real_real = df_fake_real_real[['text']].rename(
    columns={'text': 'content'})

df_fake_real_fake['label'] = 0  # Fake
df_fake_real_real['label'] = 1  # Real

df_fake_real = pd.concat(
    [df_fake_real_fake, df_fake_real_real], ignore_index=True)


# LIAR Dataset
df_liar_train = pd.read_csv("data/raw/train.tsv", sep='\t', header=None)
df_liar_test = pd.read_csv("data/raw/test.tsv", sep='\t', header=None)
df_liar_valid = pd.read_csv("data/raw/valid.tsv", sep='\t', header=None)

df_liar = pd.concat([df_liar_train, df_liar_test,
                    df_liar_valid], ignore_index=True)

df_liar = df_liar[[1, 2]]
df_liar.columns = ['raw_label', 'content']

real_labels = {'half-true', 'mostly-true', 'true'}
fake_labels = {'pants-fire', 'false', 'barely-true'}


def map_liar_label(x):
    if x in real_labels:
        return 1
    elif x in fake_labels:
        return 0
    else:
        return None


df_liar['label'] = df_liar['raw_label'].apply(map_liar_label)
df_liar = df_liar[['content', 'label']].dropna()


# Misinformation Dataset
df_misinfo_fake = pd.read_csv("data/raw/DataSet_Misinfo_FAKE.csv", header=None)
df_misinfo_true = pd.read_csv("data/raw/DataSet_Misinfo_True.csv", header=None)

# Combine all numbered columns into a single text column
df_misinfo_fake['content'] = df_misinfo_fake.astype(str).agg(' '.join, axis=1)
df_misinfo_true['content'] = df_misinfo_true.astype(str).agg(' '.join, axis=1)

# Keep only 'content' column
df_misinfo_fake = df_misinfo_fake[['content']]
df_misinfo_true = df_misinfo_true[['content']]

# Assign labels
df_misinfo_fake['label'] = 0  # Fake
df_misinfo_true['label'] = 1  # Real

# Merge
df_misinfo = pd.concat([df_misinfo_fake, df_misinfo_true], ignore_index=True)


# FakeNewsNet Dataset
df_fn_politifact_real = pd.read_csv("data/raw/politifact_real.csv")
df_fn_politifact_fake = pd.read_csv("data/raw/politifact_fake.csv")
df_fn_gossipcop_real = pd.read_csv("data/raw/gossipcop_real.csv")
df_fn_gossipcop_fake = pd.read_csv("data/raw/gossipcop_fake.csv")

df_fn_politifact_real = df_fn_politifact_real[[
    'title']].rename(columns={'title': 'content'})
df_fn_politifact_fake = df_fn_politifact_fake[[
    'title']].rename(columns={'title': 'content'})
df_fn_gossipcop_real = df_fn_gossipcop_real[[
    'title']].rename(columns={'title': 'content'})
df_fn_gossipcop_fake = df_fn_gossipcop_fake[[
    'title']].rename(columns={'title': 'content'})

df_fn_politifact_real['label'] = 1
df_fn_politifact_fake['label'] = 0
df_fn_gossipcop_real['label'] = 1
df_fn_gossipcop_fake['label'] = 0

df_fakenewsnet = pd.concat([
    df_fn_politifact_real,
    df_fn_politifact_fake,
    df_fn_gossipcop_real,
    df_fn_gossipcop_fake
], ignore_index=True)


# Merge All Datasets
df_all = pd.concat([df_fake_real, df_liar, df_misinfo,
                   df_fakenewsnet], ignore_index=True)

# Drop duplicates & NaNs
df_all.drop_duplicates(subset=['content'], inplace=True)
df_all.dropna(subset=['content', 'label'], inplace=True)

print("Dataset size after merging:", df_all.shape)
print(df_all['label'].value_counts())

# Save Final Dataset
df_all.to_csv("data/merged/merged_dataset.csv", index=False)
print("Merged dataset Saved.")
