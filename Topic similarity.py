import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd


def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
    
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted

    Returns:
        list of synsets

    Example:
        doc_to_synsets('Fish are nvqjp friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """
    

    tokens = nltk.word_tokenize(doc)
    pos = nltk.pos_tag(tokens)
    tags = [tag[1] for tag in pos]
    wntag = [convert_tag(tag) for tag in tags]
    ans = list(zip(tokens,wntag))
    sets = [wn.synsets(x,y) for x,y in ans]
    final = [val[0] for val in sets if len(val) > 0]
    
    return final

def similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
    """
    
    
    s=[]
    for i1 in s1:
        r=[]
        scores=[x for x in [i1.path_similarity(i2) for i2 in s2]if x is not None]
        if scores:
            s.append(max(scores))
    
    return sum(s)/len(s)

def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2
document_path_similarity('I like cat', 'I like dog')

#test_document_path_similarity
#Use this function to check if doc_to_synsets and similarity_score are correct.This function should return the similarity score as a float.

def test_document_path_similarity():
    doc1 = 'This is a function to test document_path_similarity.'
    doc2 = 'Use this function to see if your code in doc_to_synsets \
    and similarity_score is correct!'
    return document_path_similarity(doc1, doc2)
test_document_path_similarity()

#paraphrases is a DataFrame which contains the following columns: Quality, D1, and D2.Quality is an indicator variable which indicates if the two documents D1 and D2 are paraphrases of one another (1 for paraphrase, 0 for not paraphrase).
# Use this dataframe for questions most_similar_docs and label_accuracy

paraphrases = pd.read_csv('paraphrases.csv')
paraphrases.head()

#most_similar_docs
#Using document_path_similarity, find the pair of documents in paraphrases which has the maximum similarity score.This function should return a tuple (D1, D2, similarity_score)

def most_similar_docs():
    
    similarities = [(paraphrase['D1'], paraphrase['D2'], document_path_similarity(paraphrase['D1'], paraphrase['D2']))
                    for index, paraphrase in paraphrases.iterrows()]
    similarity = max(similarities, key=lambda item:item[2])
    
    return similarity
most_similar_docs()

#label_accuracy
#Provide labels for the twenty pairs of documents by computing the similarity for each pair using document_path_similarity. Let the classifier rule be that if the score is greater than 0.75, label is paraphrase (1), else label is not paraphrase (0). Report accuracy of the classifier using scikit-learn's accuracy_score.This function should return a float

def label_accuracy():
    from sklearn.metrics import accuracy_score

    df = paraphrases.apply(update_sim_score, axis=1)
    score = accuracy_score(df['Quality'].tolist(), df['paraphrase'].tolist())
    
    return score

def update_sim_score(row):
    row['similarity_score'] = document_path_similarity(row['D1'], row['D2'])
    row['paraphrase'] = 1 if row['similarity_score'] > 0.75 else 0
    return row


