import re
import numpy as np
import pandas as pd
import nltk
import nltk.data
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('tagsets')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import json




def tokenize(text: str, word_tokenizer: object, sent_tokenizer: object, get_tokens: str = 'text',
             word_strain: str = 'lemma', filter_length: int = (), handle_stopwords: str = 'remove',
             lower_stop_words: list = set(stopwords.words('english')), uncapitalized=True, get_sentences=False,
             punctuation=False):
    # Get tokens grouped by sentences
    if get_sentences is True:
        return sent_tokenizer.tokenize(text)

    # Get tokens grouped by texts
    if get_tokens == 'text':
        tokens = [word_tokenizer.tokenize(text)]

    # Get Tokens grouped by sentences and texts
    elif get_tokens == 'sentence':
        # Sentence based tokens
        sent_tokens = sent_tokenizer.tokenize(text)
        tokens = [word_tokenizer.tokenize(w) for w in sent_tokens]

    # Replace chars
    if len(punctuation) > 0:
        tokens = [[re.sub(punctuation, '', w) for w in s] for s in tokens]

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
    if filter_length[0] == 'long':
        tokens = [[w for w in sent if len(w) >= filter_length[1]] for sent in tokens]


    # Extract short words
    elif filter_length[0] == 'short':
        tokens = [[w for w in sent if len(w) <= filter_length[1]] for sent in tokens]

    # uncapitalize words
    if uncapitalized:
        tokens = [[w.lower() for w in sent] for sent in tokens]

    if get_tokens == 'text':
        return tokens[0]
    if get_tokens == 'sentence':
        return tokens

###############################################################################################################################################################################################

# Normalization: Dividing by total number
def divide_by_number_of_tokens(df):
    # Number of tokens per text
    num = df.sum(axis=1)
    # Divide by total word number
    df = df.divide(num, axis='index', level=None, fill_value=None)
    # Return relative frequencies
    return df


# Get the most common tokens
def filter_by_frequencies(df_tokens, filter_value, filter_type):
    # Percentage of total number of occurences
    if filter_type == 'quantile':
        summed_values = df_tokens.sum(axis=0)
        quantile = summed_values.quantile(q=1 - filter_value, interpolation='higher')
        return summed_values.loc[summed_values > quantile].index
    # Percentage of total number of tokens
    elif filter_type == 'threshold':
        summed_values = df_tokens.sum(axis=0)
        total_sum = summed_values.sum()
        most_frequent_columns = summed_values.loc[summed_values.divide(total_sum) >= filter_value]
        return most_frequent_columns.index

    #
    elif filter_type == 'relative':
        sorted_summed_values = pd.Series(df_tokens.sum(axis=0),index=df_tokens.columns).sort_values(axis = 0, ascending = False)
        most_frequent_columns = sorted_summed_values[0:int(filter_value * df_tokens.values.shape[1])]
        return most_frequent_columns.index
    #
    elif filter_type == 'absolute':
        sorted_summed_values = pd.Series(df_tokens.sum(axis=0),index=df_tokens.columns).sort_values(axis = 0, ascending = False)
        most_frequent_columns = sorted_summed_values[0:min(filter_value,df_tokens.values.shape[1])]
        return most_frequent_columns.index



# Select features which have a high variance
def filter_by_variance(df, var_filter_type, threshold):
    df_variances = pd.DataFrame(df.var(axis=0), index=df.columns).T

    columns_with_highest_variances = filter_by_frequencies(df_tokens=df_variances, filter_value=threshold,
                                                           filter_type=var_filter_type)
    return columns_with_highest_variances




# Converts a collection of text documents to a matrix of token/ngrams occuracies. Therefore all texts are considered. Filter the tokens/ngrams to get the most relevant features.
def n_grams(ngram_range, analyzer, tokenizer,fit_transform, normalization_type, cv_min_df, vocabulary):

    # TF-IDF or TF
    if normalization_type == 'tfidf':
        c = TfidfVectorizer(tokenizer=tokenizer, ngram_range=ngram_range, lowercase=False, analyzer=analyzer,
                            vocabulary=vocabulary)
    #
    else:
        c = CountVectorizer(tokenizer=tokenizer, ngram_range=ngram_range, lowercase=False, analyzer=analyzer,
                            min_df=cv_min_df, vocabulary=vocabulary)

    # Transform ngrams to vectors
    df_frequencies = pd.DataFrame(c.fit_transform(fit_transform).toarray(), columns=c.get_feature_names())

    # Normalization by dividing the counts of tokens by the total number of tokens
    if normalization_type != 'tfidf':
        df_frequencies = divide_by_number_of_tokens(df_frequencies)


    return df_frequencies


# Get relevant n-grams und tags to use them as vocabularies for the CountVectorizer-Objects in the feature engineering step
def select_features(serie_texts, features_to_calc, token_params_1, filter_params, flag_extract_features, cv_min_df,
                    path_to_features, normalization_type):
    # Dictionary with the selected features
    dict_features = {}
    df_features = pd.DataFrame()

    # Original words in tokens -> Neither stemming nor lemmatizing
    series_original_word_tokens_grouped_by_sentences = serie_texts.apply(lambda row:tokenize(text=row, word_tokenizer=token_params_1["word_tokenizer"], sent_tokenizer=token_params_1['sent_tokenizer'], get_tokens='sentence',
             word_strain= '', filter_length= ('long',0), handle_stopwords = '',
             lower_stop_words=set(stopwords.words('english')), uncapitalized=False, get_sentences=False,
             punctuation=''))

    # Serie with stemmed/lemmatized tokens grouped by sentences and texts( text 1 -> [[<tokens in sentence 1>],[...]...])
    series_tokens_grouped_by_sentences = serie_texts.apply(lambda row: tokenize(row, **token_params_1))
    # Serie with stemmed/lemmatized tokens grouped by sentences and texts( text 1 -> [token1,token2,...])
    series_tokens_grouped_texts = series_tokens_grouped_by_sentences.apply(lambda row: sum([sent for sent in row], []))

    if flag_extract_features is True:
        # Total number of tokens per text
        series_num_of_tokens = series_tokens_grouped_by_sentences.apply(
            lambda row: np.array([len(w) for w in row]).sum())

        # Text lengths per text
        series_text_length = serie_texts.apply(lambda row: len(row))

        # Calculate (number of chars in text)/(number of sentences in text)
        if features_to_calc['avg_sent_len']:
            series_num_of_sent = series_tokens_grouped_by_sentences.apply(lambda row: len(row))
            series_avg_sent_len = series_text_length.divide(series_num_of_sent)
            df_features['avg_sent_len'] = series_avg_sent_len

        # Calculate (number of chars in text)/(number of chars in text)
        if features_to_calc['avg_word_len']:
            series_avg_token_len = series_text_length.divide(series_num_of_tokens)
            df_features['avg_word_len'] = series_avg_token_len

        # Number of tokens per sentence
        if features_to_calc['token_per_sent']:
            # Calculate number of sentences
            series_num_of_sent = series_tokens_grouped_by_sentences.apply(lambda row: len(row))
            # Number of tokens/number of sentences
            series_token_per_sent = series_num_of_tokens.divide(series_num_of_sent)
            df_features['token_per_sent'] = series_token_per_sent

        # Vocabulary_richness = (Number of different tokens)/(Total word number)
        if features_to_calc['vocabulary_richness']:
            # Define the number of unique tokens
            series_set_of_tokens = series_tokens_grouped_texts.apply(lambda row: len(set(row)))
            # Calculate the Vocabulary_richness of each author
            series_vocabulary_richness = series_set_of_tokens.divide(series_num_of_tokens)
            df_features['vocabulary_richness'] = series_vocabulary_richness

            with open(path_to_features) as f:
                dict_features = json.load(f)


    # n-grams

    # Serie to dict to pass it to CountVectorizer
    dict_tokens = series_tokens_grouped_texts.to_dict()

    # Get pos-tags grouped by sentences
    series_pos_grouped_by_sentences =  series_original_word_tokens_grouped_by_sentences.apply(lambda row: [nltk.pos_tag(sent) for sent in row])
    # Get pos-tags grouped by texts
    series_pos_grouped_by_texts=series_pos_grouped_by_sentences.apply(lambda row: sum([sent for sent in row], []))
    # Create dict to pass it to the CountVectorizer object
    dict_pos = series_pos_grouped_by_texts.apply(lambda row: [w[1] for w in row])

    # Various inputs for CountVectorizer
    n_gram_tokenizer_fit_analyzer = {
        'word_n_grams': (lambda key: dict_tokens[key], dict_tokens.keys(), 'word'),
        'char_n_grams': (None, serie_texts, 'char'),
        'pos_n_grams': (lambda key: dict_pos[key], dict_pos.keys(), 'word')
    }

    for f in n_gram_tokenizer_fit_analyzer.keys():

        for index, n in enumerate(features_to_calc[f]):
            # If flag_extract_features is True -> use already selected features as vocabulary
            # If flag_extract_features is False -> Vocabulary is None, Select new features in CountVectorizer
            if flag_extract_features is True:
                vocabulary = dict_features[f + '_' + str(n)]
            else:
                vocabulary = None

            # Get ngrams
            df_frequencies = n_grams(tokenizer=n_gram_tokenizer_fit_analyzer[f][0],
                                     fit_transform=n_gram_tokenizer_fit_analyzer[f][1],
                                     ngram_range=(n, n),
                                     analyzer=n_gram_tokenizer_fit_analyzer[f][2],
                                     normalization_type=normalization_type, cv_min_df=cv_min_df,
                                     vocabulary=vocabulary)

            # Only use features which occurs often in all texts.
            if filter_params['freq_' + f][0]:
                # Filter the ngrams regarding their frequencies
                most_common_ngrams = filter_by_frequencies(pd.DataFrame(df_frequencies,
                                                           filter_params['freq_' + f][1], filter_params['freq_' + f][1]))
                df_frequencies = df_frequencies.loc[:, most_common_ngrams]

            # Filter the ngrams regarding their variances
            if filter_params['var_' + f][0]:
                mask = filter_by_variance(df=df_frequencies, var_filter_type=filter_params['var_' + f][1],
                                          threshold=filter_params['var_' + f][2][index])
                # Return DF which includes the selected features
                df_frequencies = df_frequencies.loc[:, mask]

            # If feature selection
            if flag_extract_features is False:
                # Add to features to the feature dictionary
                dict_features[f + '_' + str(n)] = df_frequencies.columns.tolist()
            # If feature extraction
            else:
                # Add frequencies to feature dictionary
                df_features = pd.concat((df_features, df_frequencies), axis=1)

    if flag_extract_features is False:
        return dict_features

    if flag_extract_features is True:
        return df_features

