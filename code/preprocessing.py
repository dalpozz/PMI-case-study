

# functions used to preproces the data

import re

# sue regular expressions to remove unwanted text

def remove_unwanted_text(email):
    
    re0 = re.compile('>')
    text = re.sub(re0, ' ', email)
    re1 = re.compile('(Message-ID(.*?\n)*X-FileName.*?\n)|'
                     '(To:(.*?\n)*?Subject.*?\n)|'
                     '(< (Message-ID(.*?\n)*.*?X-FileName.*?\n))')
    text = re.sub(re1, ' ', text)
    re2 = re.compile('(.+)@(.+)') # Remove emails
    text = re.sub(re2, ' ', text)
    re3 = re.compile('\s(-----)(.*?)(-----)\s', re.DOTALL)
    text = re.sub(re3, ' ', text)
    re4 = re.compile('''\s(\*\*\*\*\*)(.*?)(\*\*\*\*\*)\s''', re.DOTALL)
    text = re.sub(re4, ' ', text)
    re5 = re.compile('\s(_____)(.*?)(_____)\s', re.DOTALL)
    text = re.sub(re5, ' ', text)
    re6 = re.compile('\n( )*-.*')
    text = re.sub(re6, ' ', text)
    re7 = re.compile('\n( )*\d.*')
    text = re.sub(re7, ' ', text)
    re8 = re.compile('(\n( )*[\w]+($|( )*\n))|(\n( )*(\w)+(\s)+(\w)+(( )*\n)|$)|(\n( )*(\w)+(\s)+(\w)+(\s)+(\w)+(( )*\n)|$)')
    text = re.sub(re8, ' ', text)
    re9 = re.compile('.*orwarded.*')
    text = re.sub(re9, ' ', text)
    re10 = re.compile('From.*|Sent.*|cc.*|Subject.*|Embedded.*|http.*|\w+\.\w+|.*\d\d/\d\d/\d\d\d\d.*')
    text = re.sub(re10, ' ', text)
    re11 = re.compile(' [\d:;,.]+ ')
    text = re.sub(re11, ' ', text)

    return(text)



from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


#We can make use of either a) Stemming or b) Lemmatizing to find word roots. 
#The stemmer generally cuts off prefixes of words according to some set rules. 
#Thus words like 'facilitate' and shortened to 'faci' - this can be confusing and requires that the words are 're-built' before displayed. The lemmatizer also used set rules for words of a certain form, but it has the advantage of comparing words to a dictionary.
#We will use the lemmatizer.


#Tokenization
def get_token(text):

    raw = text.lower()
    
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(raw)

    # create English stop words list
    additional_stop = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the",'like', 'think', 'know', 'want', 'sure', 'thing', 'send', 'sent', 'speech', 'print', 'time','want', 'said', 'maybe', 'today', 'tomorrow', 'thank', 'thanks']
    en_stop = stopwords.words('english') + additional_stop

    # Removing stop words
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # lemmatize token
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [wordnet_lemmatizer.lemmatize(i) for i in stopped_tokens]
    
    # Removing words having len <= 2
    final_tokens = [i for i in lemmatized_tokens if len(i)>2]

    return(final_tokens)


