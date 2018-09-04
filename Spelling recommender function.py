#For anlyzing moby dicks need to use nltk to explore the Herman Melville novel Moby Dick. Then in spelling recommender, create a spelling recommender function that uses nltk to find words similar to the misspelling.

import nltk
import pandas as pd
import numpy as np

#To work with the raw text use 'moby_raw'

with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
#To work with the novel in nltk.Text format use 'text1'

moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)

                                          #Analyzing moby dick

#How many tokens(words and punctuations symbols) are in text?

def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

example_one()

#How many unique tokens (unique words and punctuations) does text1 have?

def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

example_two()

#After lemmatizing the verb,how many unique tokens does text1 have?

from nltk.stem import WordNetLemmatizer

def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

example_three()

#What is the lexical diversity of the given text input?(i.e. ratio of unique tokens to the total number of tokens)

def answer_one():
    
    
    return example_two()/example_one()

answer_one()

#What percentage of token is 'whale' or 'Whale'?

def answer_two():
    
    
    return (text1.vocab()['whale'] + text1.vocab()['Whale']) / len(nltk.word_tokenize(moby_raw)) * 100

answer_two()

#What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
#This function should return a list of 20 tuples where each tuple is of the form (token, frequency). The list should be sorted in descending order of frequency.

def answer_three():
    import operator
    
    return sorted(text1.vocab().items(), key=operator.itemgetter(1), reverse=True)[:20] 

answer_three()

#What tokens have a length of greater than 5 and frequency of more than 150?
#This function should return a sorted list of the tokens that match the above constraints. To sort your list, use sorted()

def answer_four():
    
    
    return sorted([token for token, freq in text1.vocab().items() if len(token) > 5 and freq > 150])

answer_four()

#Find the longest word in text1 and that word's length.
#This function should return a tuple (longest_word, length).

def answer_five():
    
    import operator
    
    return sorted([(token, len(token))for token, freq in text1.vocab().items()], key=operator.itemgetter(1), reverse=True)[0]

answer_five()

#What unique words have a frequency of more than 2000? What is their frequency?
#"Hint: you may want to use isalpha() to check if the token is a word and not punctuation."
#This function should return a list of tuples of the form (frequency, word) sorted in descending order of frequency.

def answer_six():
    
    import operator
    
    return sorted([(freq, token) for token, freq in text1.vocab().items() if freq > 2000 and token.isalpha()], key=operator.itemgetter(0), reverse=True) 

answer_six()

#What is the average number of tokens per sentence?This function should return a float.

def answer_seven():
    
    
    return np.mean([len(nltk.word_tokenize(sent)) for sent in nltk.sent_tokenize(moby_raw)]) 
answer_seven()

#What are the 5 most frequent parts of speech in this text? What is their frequency?This function should return a list of tuples of the form (part_of_speech, frequency) sorted in descending order of frequency.

def answer_eight():
    
    from collections import Counter
    import operator
    
    return sorted(Counter([tag for token, tag in nltk.pos_tag(text1)]).items(), key=operator.itemgetter(1), reverse=True)[:5]
answer_eight()

                                      #Spelling recommender

'''
For this part of the project create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.

For every misspelled word, the recommender should find find the word in correct_spellings that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.

*Each of the three different recommenders will use a different distance measure (outlined below).

Each of the recommenders should provide recommendations for the three default words provided: ['cormulent', 'incendenece', 'validrate'].
'''

from nltk.corpus import words

correct_spellings = words.words()

#For this recommender, function should provide recommendations for the three default words provided above using the following distance metric:Jaccard distance on the trigrams of the two words.This function should return a list of length three: ['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation'].

def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    result = []
    import operator
    for entry in entries:
        spell_list = [spell for spell in correct_spellings if spell.startswith(entry[0]) and len(spell) > 2]
        distance_list = [(spell, nltk.jaccard_distance(set(nltk.ngrams(entry, n=3)), set(nltk.ngrams(spell, n=3)))) for spell in spell_list]

        result.append(sorted(distance_list, key=operator.itemgetter(1))[0][0])
    
    return result
answer_nine()

#For this recommender, function should provide recommendations for the three default words provided above using the following distance metric:Jaccard distance on the 4-grams of the two words.This function should return a list of length three: ['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation'].

def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    result = []
    import operator
    for entry in entries:
        spell_list = [spell for spell in correct_spellings if spell.startswith(entry[0]) and len(spell) > 2]
        distance_list = [(spell, nltk.jaccard_distance(set(nltk.ngrams(entry, n=4)), set(nltk.ngrams(spell, n=4)))) for spell in spell_list]

        result.append(sorted(distance_list, key=operator.itemgetter(1))[0][0])
    
    return result
answer_ten()

#For this recommender, function should provide recommendations for the three default words provided above using the following distance metric:Edit distance on the two words with transpositions.This function should return a list of length three: ['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation'].

def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    result = []
    import operator
    for entry in entries:
        spell_list = [spell for spell in correct_spellings if spell.startswith(entry[0]) and len(spell) > 2]
        distance_list = [(spell, nltk.edit_distance(entry, spell, transpositions=True)) for spell in spell_list]

        result.append(sorted(distance_list, key=operator.itemgetter(1))[0][0])
    
    return result
answer_eleven()




