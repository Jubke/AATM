import re
import numpy as np
import pandas as pd
import nltk
import nltk.data
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
nltk.download('tagsets')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def tokenize(text:str,word_tokenizer:object, sent_tokenizer: object , get_tokens:str='text',word_strain:str='lemma', filter_length:int=(),handle_stopwords:str='remove', lower_stop_words:list=set(stopwords.words('english')),uncapitalized=True, get_sentences = False, punctuation=False):
        # Zahlen

        # Get tokens grouped by text
        if get_sentences is True:
            return sent_tokenizer.tokenize(text)

        # Get tokens grouped by texts
        if get_tokens=='text':
            tokens = [word_tokenizer.tokenize(text)]

        # Get Tokens grouped by sentences and texts
        elif get_tokens=='sentence':
            # Sentence based tokens
            sent_tokens = sent_tokenizer.tokenize(text)
            tokens = [word_tokenizer.tokenize(w) for w in sent_tokens]

        # Replace chars
        if len(punctuation) > 0:
            tokens = [[re.sub(punctuation,'',w) for w in s] for s in tokens]

        # Remove stopwords
        if handle_stopwords == 'remove':
            tokens = [[w for w in s if not w.lower() in lower_stop_words] for s in tokens]


        # Extract stopwords
        elif handle_stopwords == 'get':
            tokens = [[w.lower() for w in s if w.lower() in lower_stop_words] for s in tokens]

        # Lemmatize words
        if word_strain == 'lemma':
            lemmatizer = WordNetLemmatizer()
            tokens = [[lemmatizer.lemmatize(y) for y in sent] for sent in tokens]

        # Stem words
        elif word_strain == 'stem':
            stemmer = PorterStemmer()
            tokens = [[stemmer.stem(y) for y in sent] for sent in tokens]


        # Extract long words
        if filter_length[0]=='long':
            tokens=[[w for w in sent if len(w)>= filter_length[1]] for sent in tokens]


        # Extract short words
        elif filter_length[0]=='short':
            tokens = [[w for w in sent if len(w)<= filter_length[1]] for sent in tokens]

        # uncapitalize words
        if uncapitalized:
            tokens = [[w.lower() for w in sent] for sent in tokens]

        if get_tokens=='text':
            return tokens[0]
        if get_tokens=='sentence':
            return tokens

###############################################################################################################################################################################################

# Normalization: Dividing by total number
def divide_by_number_of_tokens(df):
    #Number of tokens per text
    num = df.sum(axis=1)
    # Divide by total word number
    df = df.divide(num, axis='index', level=None, fill_value=None)
    # Return relative frequencies
    return df

# Get the most common tokens
def get_most_common_tokens(df_tokens, percentage, how_to_select):
    # Percentage of total number of occurences
    if how_to_select=='quantile':
        summed_values=df_tokens.sum(axis=0)
        quantile=summed_values.quantile(q= 1 - percentage, interpolation='higher')
        return summed_values.loc[summed_values > quantile].index
    # Percentage of total number of tokens
    elif how_to_select== 'relative':
        sorted_values=df_tokens.sum(axis=0).sort_values(ascending=False)
        most_frequent_columns= sorted_values.iloc[0:int(percentage * len(sorted_values))]
        return most_frequent_columns.index

# Select features which have a high variance
def feature_selection(df:pd.DataFrame, feature_selection_type, threshold):
    if feature_selection_type=='VarianceThreshold':
        df_variances=pd.DataFrame(df.var(axis=0),index=df.columns).T
        columns_with_highest_variances=get_most_common_tokens(df_tokens=df_variances, percentage=threshold,how_to_select='quantile')
        return columns_with_highest_variances

        #sel = VarianceThreshold(threshold=threshold)
        #sel.fit(df)
        #return sel.get_support()
    else: return df.columns


#Converts a collection of text documents to a matrix of token/ngrams occuracies. Therefore all texts are considered. Filter the tokens/ngrams to get the most relevant features.
def n_grams(ngram_range, perc_most_common, threshold, analyzer, tokenizer, fit_transform, filter_freq, filter_var, tfidf, how_to_select,feature_selection_type):
    # TF-IDF or TF
    if tfidf:
        c = TfidfVectorizer(tokenizer=tokenizer,ngram_range=ngram_range,lowercase=False,analyzer=analyzer)
    else:
        c = CountVectorizer(tokenizer=tokenizer,ngram_range=ngram_range,lowercase=False,analyzer=analyzer)
    # Transform ngrams to vectors
    df_frequencies = pd.DataFrame(c.fit_transform(fit_transform).toarray(),columns=c.get_feature_names())

    # Only use features which occurs often in all texts.
    if filter_freq:
        # Filter the ngrams regarding their frequencies
        most_common_ngrams = get_most_common_tokens(pd.DataFrame( df_frequencies, columns=c.get_feature_names()), perc_most_common, how_to_select)
        df_frequencies =  df_frequencies.loc[:, most_common_ngrams]

    # Normalization by dividing the counts of tokens by the total number of tokens
    if tfidf is False:
        df_frequencies = divide_by_number_of_tokens(df_frequencies)


    # Filter the ngrams regarding their variances
    if filter_var:
        mask=feature_selection(df=df_frequencies, feature_selection_type=feature_selection_type, threshold=threshold)
        # Return DF which includes the selected features
        return df_frequencies.loc[:,mask]
    else:
        return df_frequencies


# Get relevant n-grams und tags to use them as vocabularies for the CountVectorizer-Objects in the feature engineering step
def calc_features(corpus_series, features_to_calc, token_params_1, threshold_word_ngrams, threshold_char_ngrams,threshold_pos_ngrams, word_perc_most_common, char_perc_most_common, pos_perc_most_common):

    dict_df_features={}

    # Serie with tokens grouped by sentences and texts( text 1 -> [[<tokens in sentence 1>],[...]...])
    series_tokens_grouped_by_sentences = corpus_series.apply(lambda row: tokenize(row, **token_params_1))

    # Total number of tokens per text
    series_num_of_tokens = series_tokens_grouped_by_sentences.apply(lambda row: np.array([len(w) for w in row]).sum())

    # Text lengths per text
    series_text_length = corpus_series.apply(lambda row: len(row))


    # Calculate (number of chars in text)/(number of sentences in text)
    if features_to_calc['avg_sent_len']:
        series_num_of_sent=series_tokens_grouped_by_sentences.apply(lambda row: len(row))
        series_avg_sent_len=series_text_length.divide(series_num_of_sent)
        dict_df_features['avg_sent_len']=(series_avg_sent_len,series_avg_sent_len.index)

    # Calculate (number of chars in text)/(number of chars in text)
    if features_to_calc['avg_word_len']:
        series_avg_token_len=series_text_length.divide(series_num_of_tokens)
        dict_df_features['avg_word_len'] = (series_avg_token_len, series_avg_token_len.index)

    # Number of tokens per sentence
    if features_to_calc['token_per_sent']:
        # Calculate number of sentences
        series_num_of_sent = series_tokens_grouped_by_sentences.apply(lambda row: len(row))
        # Number of tokens/number of sentences
        series_token_per_sent = series_num_of_tokens.divide(series_num_of_sent)
        dict_df_features['token_per_sent'] = (series_token_per_sent, series_token_per_sent.index)

    # Vocabulary_richness = (Number of different tokens)/(Total word number)
    if features_to_calc['vocabulary_richness']:
        # Define the number of unique tokens
        series_set_of_tokens = series_tokens_grouped_by_sentences.apply(lambda row: len(set(sum([sent for sent in row],[]))))
        # Calculate the Vocabulary_richness of each author
        series_vocabulary_richness = series_set_of_tokens.divide(series_num_of_tokens)
        # Add to features
        dict_df_features['vocabulary_richness'] = (series_vocabulary_richness, series_vocabulary_richness.index)

    # Series to dict to pass it to CountVectorizer
    dict_tokens = series_tokens_grouped_by_sentences.apply(lambda row: sum([sent for sent in row], [])).to_dict()

    # Create word ngrams and filter them regarding frequencies and variances
    if len(features_to_calc['word_n_grams'])>0:
        # Empty DataFrame to save ngrams
        df_word_ngrams=pd.DataFrame()
        for index,i in enumerate(features_to_calc['word_n_grams']):
            # Get word ngrams
            df_frequencies = n_grams(tokenizer=lambda key: dict_tokens[key], fit_transform=dict_tokens.keys(),
                                     ngram_range=(i, i), perc_most_common=word_perc_most_common[2][index], how_to_select=word_perc_most_common[1],
                                     threshold=threshold_word_ngrams[2][index],
                                     feature_selection_type=threshold_word_ngrams[1], analyzer='word', filter_freq=word_perc_most_common[0], filter_var=threshold_word_ngrams[0],tfidf=word_perc_most_common[3])
            # Concatenate values to a dataframe
            df_word_ngrams=pd.concat([df_word_ngrams, df_frequencies], axis=1)
        # Add to features to the feature dictionary
        dict_df_features['word_n_grams'] = (df_word_ngrams, df_word_ngrams.index)

    # Create word ngrams and filter them regarding frequencies and variances.
    if len(features_to_calc['char_n_grams']) > 0:
        # Empty DataFrame to save ngrams
        df_char_ngrams = pd.DataFrame()
        for index,i in enumerate(features_to_calc['char_n_grams']):
            # Get filtered char i-grams
            df_frequencies = n_grams(tokenizer=None, fit_transform=corpus_series,
                                     ngram_range=(i, i), perc_most_common=char_perc_most_common[2][index], how_to_select=char_perc_most_common[1],
                                     threshold=threshold_char_ngrams[2][index],
                                     feature_selection_type=threshold_char_ngrams[1], analyzer='char', filter_freq=char_perc_most_common[0], filter_var=threshold_char_ngrams[0],tfidf=char_perc_most_common[3])
            # Concatenate values to a dataframe
            df_char_ngrams = pd.concat([df_char_ngrams, df_frequencies], axis=1)
        # Add to features
        dict_df_features['char_n_grams'] = (df_char_ngrams, df_char_ngrams.index)

    # Create pos ngrams and filter them regarding frequencies and variances
    if len(features_to_calc['POS_n_grams']) > 0:
        # Get pos-tags grouped by texts
        series_pos= series_tokens_grouped_by_sentences.apply(lambda row: nltk.pos_tag(sum([sent for sent in row],[])))
        # Create dict to pass it to the CountVectorizer object
        dict_pos=series_pos.apply(lambda row: [w[1] for w in row])
        # Empty DataFrame to save ngrams
        df_pos_ngrams=pd.DataFrame()
        for index, i in enumerate(features_to_calc['POS_n_grams']):
            # Get pos filtered n-grams
            df_frequencies = n_grams(tokenizer=lambda key: dict_pos[key], fit_transform=dict_pos.keys(),
                                     ngram_range=(i, i), perc_most_common=pos_perc_most_common[2][index], how_to_select=pos_perc_most_common[1],
                                     threshold=threshold_word_ngrams[2][index],
                                     feature_selection_type=threshold_pos_ngrams[1], analyzer='word', filter_freq=pos_perc_most_common[0], filter_var=threshold_pos_ngrams[0],tfidf=pos_perc_most_common[3])
            # Concatenate values to a dataframe
            df_pos_ngrams = pd.concat([df_pos_ngrams, df_frequencies], axis=1)
        # Add to features dictionary
        dict_df_features['pos_n_grams'] = (df_pos_ngrams, df_pos_ngrams.index)

    # View features
    for i in dict_df_features.keys():
        print(i)
        print('DF:\n',dict_df_features[i][0])
        print('Vocabulary:\n',dict_df_features[i][1])

    return dict_df_features



df_booksummaries_all = pd.read_table("E:\\Marius\\Documents\\Studium\\Programmieren\\Daten\\Text\\booksummaries\\booksummaries\\booksummaries.txt", sep='\t', names=["Wikipedia ID","Freebase ID","Book title","Book author", "Publication date" ,"Genres", "Content"])

# Content of books
content=df_booksummaries_all.Content

features_to_calc={'char_n_grams': [1,2,3,4],
        'word_n_grams': [1,2,3,4],
        #'function_words':[],
        'POS_n_grams':[1,2,3],
        'avg_sent_len':True,
        'avg_word_len':True,
        'token_per_sent':True,
        'vocabulary_richness':True,
}

token_params_1={
        'word_tokenizer':RegexpTokenizer(r'\w+'),
        'sent_tokenizer':nltk.data.load('tokenizers/punkt/english.pickle'),
        'get_tokens':'sentence',                                                    # Group tokens: 'text'|'sentence'
        'word_strain':'lemma',                                                      # 'lemma'|'stem'
        'filter_length':('long', 0),                                                # Get tokens with length >= <int> or <= <int>: ('long',<int>)|('short',<int>)
        'handle_stopwords':'',                                                      # 'get'|'remove'
        'get_sentences':False,                                                      # Sentences tokens
        'punctuation':"[,;.!—]",                                                   # Remove punctuation
        'lower_stop_words':set(stopwords.words('english')),                         # Stopwords
        'uncapitalized' : True                                                      # Original or uncapitalized stemms/lemmas
    }


calc_features(corpus_series=content, features_to_calc=features_to_calc, token_params_1=token_params_1, threshold_word_ngrams=(True,'VarianceThreshold',[0.25,0.25,0.25,0.25]), threshold_char_ngrams=(True,'VarianceThreshold',[0.25,0.25,0.25,0.25]),threshold_pos_ngrams=(True,'VarianceThreshold',[0.25,0.25,0.25,0.25]),word_perc_most_common=(True,'relative',[0.25,0.25,0.25,0.25],False), char_perc_most_common=(True,'quantile',[0.25,0.25,0.25,0.25],False),pos_perc_most_common=(False,'relative',[0.25,0.25,0.25,0.25],False))

