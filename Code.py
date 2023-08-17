import nltk
import pandas as pd
# nltk.download()
import re
import gensim #the library for Topic modelling
from gensim.models.ldamulticore import LdaMulticore
from gensim import corpora, models

import pyLDAvis.gensim_models as gensimvis
import pickle
import pyLDAvis


# merging all tweets based on various hashtags concerning the world cup into a single csv file
df_tweets1 = pd.read_csv("C:\\Users\\Jedidiah Kristilere\\Documents\\Enoch\\#worldcup2022_output_search_tweets.csv")

df_tweets2 = pd.read_csv("C:\\Users\\Jedidiah Kristilere\\Documents\\Enoch\\"
                         "#qatarworldcup2022_output_search_tweets.csv")
df_tweets3 = pd.read_csv("C:\\Users\\Jedidiah Kristilere\\Documents\\Enoch\\"
                         "#fifaworldcup2022qatar_output_search_tweets.csv")
comb_tweets = [df_tweets1, df_tweets2, df_tweets3]
master_tweets = pd.concat(comb_tweets)


# Extracting the comments to prep for analysis
master_tweets = master_tweets['raw_value'].str.split(",", expand=True)
for i in master_tweets.columns:
    if i != 3:
        del master_tweets[i]
    else:
        master_tweets2 = master_tweets[i].str.split("':", expand=True)
del master_tweets2[0]
master_tweets2.reset_index(drop=True, inplace=True)


# Convert all strings to lowercase characters
master_tweets2.columns = ['tweets']
master_tweets2 = master_tweets2['tweets'].str.lower()

# Remove https and all /n characters
master_tweets2 = master_tweets2.apply(lambda x: re.split('https:\/\/.*', str(x))[0])
master_tweets2 = master_tweets2.apply(lambda x: re.sub(r'\n', '', str(x)))
master_tweets2.to_csv("C:\\Users\\Jedidiah Kristilere\\Documents\\Enoch\\""cleaned_tweets.csv")


# Remove all punctuation, numbers, special characters and non-english words
master_tweets2 = pd.read_csv("C:\\Users\\Jedidiah Kristilere\\Documents\\Enoch\\cleaned_tweets.csv")
master_tweets2 = master_tweets2.replace('\W', ' ', regex=True)

# Removing non-english words
def remove_non_english(text):
    english_words = set(nltk.corpus.words.words())
    text = " ".join(w for w in nltk.wordpunct_tokenize(text)
                    if w.lower() in english_words or not w.isalpha())
    text = re.sub('\s+', ' ', text).strip()
    return text
master_tweets2 = master_tweets2.drop(209)
master_tweets2['tweets'] = master_tweets2['tweets'].apply(remove_non_english)

# Removing numbers
def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
master_tweets2['tweets'] = master_tweets2['tweets'].apply(lambda x: cleaning_numbers(x))


# SENTIMENT ANALYSIS
# Using the Nltk package to remove stop words, lemmatization, and tokenize
# Tokenization
from nltk.tokenize import RegexpTokenizer

regexp = RegexpTokenizer('\w+')
master_tweets2['tweets_tokenize'] = master_tweets2['tweets'].apply(regexp.tokenize)

# Removing Stop Words
from nltk.corpus import stopwords

stopwords = nltk.corpus.stopwords.words("english")
master_tweets2['tweets_tokenize'] = master_tweets2['tweets_tokenize'].apply(lambda x: [item for item in x if item not
                                                                                       in stopwords])
# Removing infrequent words
master_tweets2['tweet_strings'] = master_tweets2['tweets_tokenize'].apply(lambda x: ' '.join([item for item in x if
                                                                                              len(item) > 2]))

# Applying lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
wnl = WordNetLemmatizer()
def lemmatize_text(text):
    return " ".join([wnl.lemmatize(word) for word in text.split()])
master_tweets2['lemm_tweets'] = master_tweets2['tweet_strings'].apply(lambda x: lemmatize_text(x))


# Sentiment Analysis Using VADER Lexicon
from nltk.sentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
master_tweets2['polarity'] = master_tweets2['lemm_tweets'].apply(lambda x: analyzer.polarity_scores(x))
master_tweets2 = pd.concat([master_tweets2.drop(['polarity'], axis=1), master_tweets2['polarity'].apply(pd.Series)],
                           axis=1)


# Create a new column with sentiment "neutral", "positive" and "negative"
master_tweets2['sentiment'] = master_tweets2['compound'].apply(lambda x: "positive" if x > 0 else "neutral" if x == 0
else "negative")


# TOPIC MODELLING
# TOPIC MODELLING using LDA gensim
dictionary = corpora.Dictionary(master_tweets2['tweets_tokenize'])
doc_term_matrix = [dictionary.doc2bow(doc) for doc in master_tweets2['tweets_tokenize']]
lda = gensim.models.ldamodel.LdaModel

num_topics = 7
lda_model = lda(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=50, minimum_probability=0)

for topic_id, topic in lda_model.print_topics():
    print(f'Topic ID: {topic_id}')
    print(f'Top Keywords: {topic}\n')
master_tweets2['topic'] = [max(lda_model[doc_term_matrix[i]], key=lambda x: x[1])[0] for i in range(len(doc_term_matrix))]


# Visualize the LDA Model
lda_display = gensimvis.prepare(lda_model, doc_term_matrix, dictionary, sort_topics=False, mds='mmds')
pyLDAvis.display(lda_display)
pyLDAvis.save_html(lda_display, "C:\\Users\\Jedidiah Kristilere\\Documents\\Enoch\\lda_display" + str(num_topics) + ".html")


# Save the data into a csv file
master_tweets2.to_csv("C:\\Users\\Jedidiah Kristilere\\Documents\\Enoch\\""analysed_tweets.csv", index=False)
