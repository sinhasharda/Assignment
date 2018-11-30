# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 17:27:27 2018

@author: Sharda.sinha
"""

"""
importing all libraries
"""
import pandas as pd
import nltk
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from nltk.stem.snowball import SnowballStemmer

#import spacy for lemmatization
import spacy


# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  
import matplotlib.pyplot as plt
#%matplotlib inline

def gen_word_cloud(filepath, column):
    """
    Takes file path and relevant column name as input and generates
    word cloud and returns reverse sorted word tokens with associated
    frequency.
    """
    data_df= pd.read_csv(full_path)
    
    corpus_text = data_df[column].str.lower().tolist()
    makeitastring = " ".join(map(str, corpus_text))
    
    STOPWORDS = nltk.corpus.stopwords.words('english')
    #print STOPWORDS
    wordcloud = WordCloud(width = 1000, height = 500, stopwords=set(STOPWORDS))
    wordcloud.generate(makeitastring)
    word_freq = wordcloud.process_text(makeitastring)
    word_freq= sorted(word_freq.items(), key = lambda x: x[1], reverse= True)
    plt.imshow(wordcloud,interpolation="bilinear")
    plt.axis("off")
    plt.show()
    return word_freq



def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" %(topic_idx))
        print (" ".join([feature_names[i]
            for i in topic.argsort()[:-no_top_words - 1:-1]]))

    
def identify_topics_lda_sklearn(filepath, column):
    """
    Takes file path and relevant column name as input and identifies topic for 
    each document.
    """
    #parameters for Count Vectorizer
    no_features= 1000
    STOPWORDS = nltk.corpus.stopwords.words('english')
    
    # read data from path and extract corpus as iterable 
    data_df= pd.read_csv(full_path)    
    corpus_text = data_df[column].str.lower().tolist()
    
    
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words=STOPWORDS)
    tf = tf_vectorizer.fit_transform(corpus_text)
    tf_feature_names = tf_vectorizer.get_feature_names()
    
    #parametere for LDA
    no_topics= 5
    
    #Run LDA
    lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    no_top_words= 10
    display_topics(lda, tf_feature_names, no_top_words)
    

def sent_to_words(sentences):
    """
    Function to tokenize sentences given an iterable
    """
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def remove_stopwords(texts):
    """
    Function to remove stopwords given an iterable
    """
    STOPWORDS = nltk.corpus.stopwords.words('english')
    return [[word for word in simple_preprocess(str(doc)) if word not in STOPWORDS] for doc in texts]


def make_bigrams(texts, bigram_mod):
    """
    Function to generate bigrams given an iterable
    """
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts, bigram_mod, trigram_mod):
    """
    Function to generate trigrams given an iterable
    """
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def stem_text(texts):
    """
    Function to generate stemmed output of given text
    """
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    stm_text=[]
    stems = [stm_text.extend(stemmer.stem(t))  for item in texts for t in item]
    print ("type....", type(stems))
    print ("texts......", stems)
    return stems

def lemmatization(nlp, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def text_preprocess_lda_gensim(filepath, column):
    """
    Takes file path and relevant column name as input and returns text prepared 
    (bow) as input for LDA.
    """
    # read data from path and extract corpus as iterable  
    data_df= pd.read_csv(full_path)
    corpus_text = data_df[column].str.lower().tolist()
    
    #tokenizing the sentences
    tokenized_text = list(sent_to_words(corpus_text))
    
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(tokenized_text, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[tokenized_text], threshold=1) 
    
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    #remove stopwords
    text_nostop= remove_stopwords(tokenized_text)
#    print (text_nostop[0])
     
    #create bigrams
    text_with_bigrams= make_bigrams(text_nostop, bigram_mod )
    
    
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
#    python -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])
    
    # Do lemmatization keeping only noun, adj, vb, adv
    text_lemmatized = lemmatization(nlp, text_with_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
#    print(text_lemmatized[:1])
    
    #create a Gensim dictionary from the processed text
    dictionary = corpora.Dictionary(text_lemmatized)
    
    #Creating corpus for training LDA model
    texts = text_lemmatized
    
    #convert the dictionary to a bag of words corpus for reference
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    return text_lemmatized, corpus, dictionary

    
def identify_topics_lda_gensim(corpus):
    """
    Takes processed text as input and identifies topic for 
    each document.
    """
            
    
    #Build LDA model
    lda = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, 
                            id2word=dictionary, 
                            random_state=100,
                            update_every=1,
                            chunksize=100,
                            passes=10,
                            alpha='auto',
                            per_word_topics=True)
    
    perplexity= lda.log_perplexity(corpus)
    
    return lda, perplexity

def visualize_topics(lda_model, corpus, dictionary):
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    vis

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
#        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, 
                            id2word=dictionary, 
                            random_state=100,
                            update_every=1,
                            chunksize=100,
                            passes=10,
                            alpha='auto',
                            per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

    
if __name__ == "__main__":
    INPUT_PATH=  "D:\\Personal\\Interviews\\Racetrack\\"
    FILE_NAME= "issue_resolution.csv"
    full_path= INPUT_PATH + FILE_NAME
    
    word_freq= gen_word_cloud(full_path, "ISSUE")
#    identify_topics_lda_sklearn(full_path, "ISSUE")
    text_lemmatized, corpus_for_lda, dictionary= text_preprocess_lda_gensim(full_path, "ISSUE")
    lda_model, perplexity= identify_topics_lda_gensim(corpus_for_lda)
    
    # Print the Keyword in the topics
    print(lda_model.print_topics())
    
    
    # Compute Perplexity
    print('\nPerplexity: ', perplexity)  # a measure of how good the model is. lower the better.
    
    visualize_topics(lda_model, corpus_for_lda, dictionary)
    
    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus_for_lda, texts=text_lemmatized, start=2, limit=20, step=1)