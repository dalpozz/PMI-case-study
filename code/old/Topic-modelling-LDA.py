

# # Topic Modelling of the Enron corpus, using LDA (Latent Dirichlet Allocation)



# We will look at the "sent" directory of each of the 150 employees of Enron. We need to import the data and in turn, clean up the data. 
# Info from https://rforwork.info/2013/11/03/a-rather-nosy-topic-model-analysis-of-the-enron-email-corpus/ 
# and https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html proved to be very useful. 
# Also see http://www.colorado.edu/ics/sites/default/files/attached-files/01-11_0.pdf 

# We are going to place all the emails of each user into one large list. 
# In order to utalise the LDA algorithm we require there to me multiple documents. 
# The obvious question that arises is whether to consider each email as a seperate document, 
# or to consider the collection of each user's emails as a seperate document. For example:
# 
# Consider person $A$ has emails $A_1$, $A_2$, $A_3$ and person $B$ has emails $B_1$ and $B_2$. 
# Then we can create a list that is L = [$A_1$, $A_2$, $A_3$, $B_1$, $B_2$] or L = [$A_1A_2A_3$, $B_1B_2$]. 
# For now, all the emails are going to be treated as seperate documents. 
# 
# Once the LDA algorithm has been implemented, we want to be able to list all the documents that fall under a given catagory. 
# 
# We now set up the regular expressions to remove the 'clutter' from the emails.
# (Note, they are purposefully long to avoid successive searches through large data)
# 
# An alternate set of regular expressions are also included. These are seperated and thus take longer to iterate. 

import re, os

workdir = "/Users/Andrea/Documents/Education/ULB/Phd/Github/PMI-case-study/"
dirmail = str(workdir)  + "/data/enron/maildir/"
dircode = str(workdir)  + "/code/"

os.chdir(dircode)

from preprocessing import *
from collections import defaultdict




# We now build a list of strings - each string being an email (document). 
# Each document is filtered according to the regular expressions above. We also build a dictionary, namely, docs_num_dict that stores for each iteration of a name, the corresponding name and as well as a list of the filtered text.


docs = []
docs_num_dict = [] # Stores email sender's name and number

# For each user we extract all the emails in their inbox

names = [i for i in listdir()]
m = 0
for name in names:
    sent =  str(dirmail) + str(name) + '/sent'   
    try: 
        chdir(sent)
        d = []
        for email in listdir():          
            text = open(email,'r').read()
            # Regular 'clutter' from each email
            text = remove_unwanted_text(text)
            docs.append(text)
            d.append(text)
        docs_num_dict.append((m,[name,d]))
        m += 1
    except:
        pass
    
docs_num_dict = dict(docs_num_dict)


# We can make use of either a) Stemming or b) Lemmatizing to find word roots. 
# See http://textminingonline.com/dive-into-nltk-part-iv-stemming-and-lemmatization for a more detailed explination of the two. Right below, the lemmatizer is implemented. 
# 
# The stemmer generally cuts off prefixes of words according to some set rules. 
# Thus words like 'facilitate' and shortened to 'faci' - this can be confusing and requires that the words are 're-built' before displayed. 
# The lemmatizer also used set rules for words of a certain form, but it has the advantage of comparing words to a dictionary.
# 
# In general, the lemmatizer will have preference of use. 
# 
# While creating a new 'texts' variable that stores the filtered documents, we also edit the docs_num_dict to update the words according to the tokenize,stop word, lemmatize procedure.

# ### Using the lemmatizer (consider using this instead of the stemmer):


# To build the dictionary
from collections import defaultdict
d = defaultdict(int)

# We now employ the techniques as outline in the second link at the top - see **
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


texts = []

for i in range(0,len(docs_num_dict.items())):
    new_docs_num_dict_1 = []
    for doc in docs_num_dict[i][1]:
        
        tokens = get_token(doc)

        texts.append(tokens)
        new_docs_num_dict_1.append(tokens)

        # We now build the dictionary
        for word in tokens:
            d[word] += 1  
    docs_num_dict[i][1] = new_docs_num_dict_1


# The texts file as well as the dictinary d (this counts the total number of times a given word is used in the corpus) is saved.


import json

chdir(str(workdir) + '/LDA/')

# Save the texts file as texts_raw (will be edited again below)
with open('texts_raw.jsn','w') as f:
    json.dump(texts,f)
f.close()

# Save the dictionary d
with open('d.jsn','w') as f:
    json.dump(d,f)
f.close()




## Loading the raw texts file
#with open('texts_raw.jsn','r') as f:
#    texts = json.load(f)
#f.close()
#    
## Loading the dictionary d 
#with open('d.jsn','r') as f:
#    d = json.load(f)
#f.close()


# We now build the dictionary of dictionaries, docs_name_dict. 
# The dictinary associates to the names of each employee, a dictionary that stores all the words used by the given person, as well as the number of times they used each of these words. 


from collections import defaultdict
docs_name_dict = []

for i in range(0,len(docs_num_dict.items())):
    temp_dict = defaultdict(int)
    for j in docs_num_dict[i][1]:
        for k in j:
            temp_dict[k] += 1
    # Append the temporary dictionary to docs_name_dict
    docs_name_dict.append((docs_num_dict[i][0],temp_dict)) 
docs_name_dict = dict(docs_name_dict)


# We now want to remove the words from our documents that cause clutter. 
# We will remove all the words that appear in more than 20% of documents as well as removing all the words that occur in less than 4 of the documents. 
# We have a dictionary that counts the number of times a word in present across all the $57000$ documents. 
# 
# To further enhance the quality of the text we analyse, the loops below remove all words of length 1 or 2. 


num_docs = len(texts)
temp_texts = texts
texts= []
upper_lim = int(0.20*num_docs)

for doc in temp_texts:
    temp_doc = []
    for word in doc:
        # If the word is in the required interval, we add it to a NEW texts variable
        if 4 < d[word] < upper_lim and len(word) > 2:
            temp_doc.append(word)
        # If the word is not in the required interval, 
        # we lower the index of the word in the docs_name_dict dictinoary
        else:
            for group in docs_name_dict.items():
                person = group[0]
                if word in docs_name_dict[person]:
                    if docs_name_dict[person][word] > 1:
                        docs_name_dict[person][word] -= 1
                    else:
                        del docs_name_dict[person][word]
    texts.append(temp_doc)


# We proceed to save the refined texts file and the dictionary, docs_name_dict.



# We save the new 'refined' texts file
with open('texts.jsn','w') as f:
    json.dump(texts,f)
f.close()


import pickle

# We save the docs_name_dict global person, word-count dictionary
pickle.dump( docs_name_dict , open( "docs_name_dict.p", "wb" ) )



## Loading the texts file
#with open('texts.jsn', 'r') as f:
#    texts = json.load(f)
#f.close()
#
## Loading the docs_name_dict dicitonary
#docs_name_dict = pickle.load( open( "docs_name_dict.p", "rb" ) )


# Below, we construct the document term matrix whereafter the fairly lengthy process of constructing the model takes place. Thus far the model seems be linear. With a single pass, the model takes just upward of a minute to execute, whereas for 5 passes, the model takes roughly 5.5 minutes.
# 
# The model was run for 350 passes and took 316 minutes to execute.

# Constructing a document-term matrix

from gensim import corpora, models

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]


num_topics = 10

ldamodel = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=350)
#ldamodel = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=50)


# We save both the LDA data as well as the results. We can reanalyse later. See the folder called LDAdata.
# 
# To load the files again:
# 
# ldamodel = models.LdaModel.load('ldamodel.model') and dictionary = corpora.Dictionary.load('dictionary')
# 

chdir(str(workdir) + '/LDA/LDAdata_results')

# Saving the dictionary
dictionary.save('dictionary')

# Saving the corpus    
with open('corpus.jsn','w') as f:
    json.dump(corpus,f)    
f.close()

# Saving the ldamodel
ldamodel.save('ldamodel')




## Load dictionary
#dictionary = corpora.Dictionary.load('dictionary')
#
## Load ldamodel
#ldamodel = models.LdaModel.load('ldamodel') 
#
## Load corpus
#with open('corpus.jsn','r') as f:
#    corpus = json.load(f)
#f.close()


# We now print the words for each of the given topics. 
# It must be noted, that even though considerable emphasis has been placed on the construction of the regular expressions, 'junk-text' may be present.


num_words = 15
topic_list = ldamodel.print_topics(num_topics, num_words)


def get_word_topics(topic_num, topic_list):
    word_list = re.sub(r'(.\....\*)|(\+ .\....\*)', '', topic_list[topic_num][1])
    words = [word for word in word_list.split()]
    return(words)
    

Topic_words =[]
for i in range(0, num_topics):
    words = get_word_topics(i, topic_list)    
    Topic_words.append(words)
    print('Topic ' + str(i) + ': ' + ', '.join(words))
    #print('\n' + '-'*100 + '\n')


# The list of words created above is saved below, from longest to shortest length.
for i in range(0,len(Topic_words)):
    temp = Topic_words[i]
    sort_key = lambda s: (-len(s), s)
    temp.sort(key = sort_key)
    print(temp)
    Topic_words[i] = temp


# Saving the list of words
with open('topic_words.jsn','w') as f:
    json.dump(Topic_words,f)
f.close()




def text_likelihood_per_topic(texts, ldamodel, num_topics):

    score = np.zeros(num_topics)
    
    for text in texts:
        #ldamodel output : [(id1, score1), (id2, score2),... if id != 0]
        for topic_id, topic_lik in ldamodel[dictionary.doc2bow(text)]:
            # returns each topic and the likelihood that the query relates to that topic. 
            # Gensim defaults to only showing the top ones that meet a certain threshold (>= 0.01)
            score[topic_id] += topic_lik
   
    # score is the sum of likelihood on each topic
    # we normalize the score
    norm_score = score/np.sum(score)
    
    return norm_score 




# We also want to export the list of words in a csv file such that we can use the data in out D3 visualisation.


#with open('topic_words.jsn','r') as f:
#    Topic_words = json.load(f)
#f.close()


## We could now proceed to visualise the data above by using the [pyLDAvis](https://pyldavis.readthedocs.io/en/latest/index.html) package.
#
#import warnings
#warnings.filterwarnings('ignore')
#
#import pyLDAvis.gensim
#
#lda_visualise = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
#pyLDAvis.display(lda_visualise)
#


# We now also have a method to see which topics are prevalent for a given person.
# 
# Below, we create two functions, namely, get_person_topics and get_topic_persons.
# 
# get_person_topics takes in a specific person as a string and returns a dictionary with a ratio value (out of 1) for each of the 20 topics. 
#This indicates the prevalance of each of the topics as a percentage for a given person.

from collections import defaultdict

def get_person_topics(person):
    person_topics = defaultdict(int)
    total = 0
    for word in docs_name_dict[person]:
        try:
            term_topics = ldamodel.get_term_topics(word)
            #Returns most likely topics for a particular word in vocab.
            if term_topics:
                for topic_tuple in term_topics:
                    topic_id = topic_tuple[0]
                    topic_lik = topic_tuple[1] #likelihood
                    person_topics[topic_id] += topic_lik
                    total += topic_lik
        except:
            pass
        
    #person_topics contains the sum of likelihood on each topic   
    #we normalize the score based on the total likelihood
    
    #scale the values
    if total != 0:
        for person in person_topics:
            person_topics[person] = person_topics[person]/total
    return person_topics



# Finding top topic for a given person

person_topic = get_person_topics('allen-p')
maximum_topic = max(person_topic.keys(), key=(lambda key: person_topic[key]))
print(maximum_topic, '{0:.2%}'.format(person_topic[maximum_topic]))
words = get_word_topics(maximum_topic, topic_list)    


# get_topic_persons takes in a topic as an integer and returns a dictionary with a ratio value (out of 1) for all the employees. 
#This indicates which employees fall under a specific topic. 

def get_topic_persons(topic):
    specific_topic_persons = defaultdict(int)
    
    total = 0
    for person in docs_name_dict:
        person_topics = get_person_topics(person)
        person_value = person_topics[topic]
        specific_topic_persons[person] += person_value
        total += person_value

    #Scale the numbers in the dictionary to a percentage
    if total != 0:
        for person in docs_name_dict:
            specific_topic_persons[person] = specific_topic_persons[person]/total
        
    return specific_topic_persons
                


# We now see which person falls under a given topic the 'most' as well as which topic falls under a given person the 'most'.

# Finding top person for a given topic

topic_person = get_topic_persons(3)
maximum_person = max(topic_person.keys(), key=(lambda key: topic_person[key]))
print(maximum_person, '{0:.2%}'.format(topic_person[maximum_person]))



import pandas as pd
idx = ['Topic'+str(i+1) for i in range(num_topics)]
labels = ['Word'+str(i+1) for i in range(num_words)]
Topics_df = pd.DataFrame.from_records(Topic_words, columns=labels, index=idx)
Topics_df


