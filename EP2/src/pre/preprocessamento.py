#!/usr/bin/env python

import pandas as pd
import spacy
import string
import re

from sklearn.model_selection import train_test_split

# Loads the portuguese language on Spacy
# the command bellow install pt lang
# if you don't have it
# python -m spacy download pt
spacyPT = spacy.load('pt', disable = ["tagger", "parser", "ner"])
#stopwords = spacyPT.Defaults.stop_words
SEQUENCE_MAXLEN = 50

def filtro(data, column, keep):
    '''
    Input:
        data - A pandas dataframe
        keep - A list with values to keep
        column - A string with the name of column to filter
    Output:
        clean - Dataset with only keep values in the column
    '''
    return data[data[column].isin(keep)]


def preprocess_sentence(sentence):
    '''
    Normalize preprocess a sequence of words
    in a sentence. 
    This function removes trailing and
    leading spaces; it lowercase each word;
    removes stop words; punctuations; and symbols;
    In the final step we lemmatize each word.
    Input:
        sentence - a string
        lowercase - boolean
    Output:
        a string of preprocessed text
    '''
    sentence = sentence.lower().strip()
    sentence = sentence.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    sentence = sentence.translate(str.maketrans('', '', string.digits))
    sentence = ' '.join([re.sub(r'(.)\1+', r'\1', word) for word in sentence.split() if len(word) < 15])
    sentence = spacyPT(sentence)
    lemmatized = list()
    for word in sentence:
        #if word.text not in stopwords:
        lemma = word.lemma_
        if lemma:
            lemmatized.append(lemma)
        #lemmatized.append(word.text)
    return " ".join(lemmatized)

def sort_by_size(data):
    data['sentence_length'] = data['review_text'].apply(lambda x: len(x))
    data.sort_values(by=['sentence_length'], inplace=True, ignore_index=True)
    data.drop(['sentence_length'], inplace=True, axis=1)
    return data

def partilha(data_in, data_out, val_size, test_size, seed=29):
    '''
    Input:
        data_in - Path of input csv - filename included
        data_out - Path of output csv
        val_size - proportion of validation set to splitting
        test_size - proportion of test set to splitting
        seed - int for reproducibility
    Output:
        train, test, val -  pandas dataframes - result of splitting the data
    '''
    assert test_size >= val_size
    # reading the dataset and suffling it
    usecols = ["review_text", "overall_rating"]
    dataset = pd.read_csv(data_in, sep=';', usecols=usecols)
    dataset = dataset[dataset.review_text.map(lambda sentence : len(sentence.split())) <= SEQUENCE_MAXLEN]
    dataset = filtro(dataset, 'overall_rating', [1,2,3,4,5])
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
    dataset['review_text'] = dataset['review_text'].map(lambda sentence : preprocess_sentence(sentence))
    dataset['overall_rating'] = dataset['overall_rating'].map(lambda rate : rate - 1)
    # splitting the dataset - stratified splitting
    overall_size = (val_size + test_size)
    train, val = train_test_split(dataset, test_size=overall_size, stratify=dataset['overall_rating'], random_state=seed)
    test, val = train_test_split(val, test_size=val_size/overall_size, stratify=val['overall_rating'], random_state=seed)
    # reseting the index and saving the csv files on disk
    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    #sort_by_size(train)
    
    train.to_csv(data_out + "/" + "train.csv", sep=';', index=False)
    val.to_csv(data_out + "/" + "val.csv", sep=';', index=False)
    test.to_csv(data_out + "/" + "test.csv", sep=';', index=False)
    
    # printing the proportions of each set
    print('Train samples size: ', len(train))
    print('Val samples size: ', len(val))
    print('Test samples size: ', len(test))
    print()
    print('Train samples %: ', len(train)/(len(train)+len(val) + len(test)))
    print('Val samples %: ', len(val)/(len(train)+len(val) + len(test)))
    print('Test samples %: ', len(test)/(len(train)+len(val) + len(test)))
    print()
    print('Preprocessamento realizado com sucesso!')
    print('Arquivos csv  persistidos no diretório ' + data_out)
    
    return train, val, test


def create_subset_cbow():
    '''
    Cria um subconjunto do cbow de 50 dimensões do NILC
    200.000 palavras mais frequentes
    '''
    N = 200001
    with open("../data/cbow_s50.txt", "r") as file:
        head = [next(file) for x in range(N)]

    head[0] = str(N-1)+ " " + "50" + "\n"  # Conserta contagem de palavras
    with open("../data/word2vec_200k.txt", "w") as file:
        for line in head:
            file.write(line)

    print('Preprocessamento cbow 50 dim do Nilc realizado com sucesso')


# cria conjuntos de treino, validação e teste.
_, _, _ = partilha("../data/B2W-Reviews01.csv", "../data", .10, .25)

# Cria subconjunto do cbow de 50 dimensões do nilc
create_subset_cbow()
