"""
Foundations of Natural Language Processing
473118c2-74f0-4a1e-96f8-e01a281e9ef7 = submission confirmation
Assignment 1

Please complete functions, based on their doc_string description
and instructions of the assignment. 

To test your code run:

```
[hostname]s1234567 python3 s1234567.py
```

Before submission executed your code with ``--answers`` flag
```
[hostname]s1234567 python3 s1234567.py --answers
```
include generated answers.py file.

Best of Luck!
"""
from collections import defaultdict, Counter

import numpy as np  # for np.mean() and np.std()
import nltk, sys, inspect
import nltk.corpus.util
from nltk import MaxentClassifier
from nltk.corpus import brown, ppattach  # import corpora

# Import the Twitter corpus and LgramModel
from nltk_model import *  # See the README inside the nltk_model folder for more information

# Import the Twitter corpus and LgramModel
from twitter.twitter import *

twitter_file_ids = "20100128.txt"
assert twitter_file_ids in xtwc.fileids()


# Some helper functions

def ppEandT(eAndTs):
    '''
    Pretty print a list of entropy-tweet pairs

    :type eAndTs: list(tuple(float,list(str)))
    :param eAndTs: entropies and tweets
    :return: None
    '''

    for entropy, tweet in eAndTs:
        print("{:.3f} [{}]".format(entropy, ", ".join(tweet)))


def compute_accuracy(classifier, data):
    """
    Computes accuracy (range 0 - 1) of a classifier.
    :type classifier: NltkClassifierWrapper or NaiveBayes
    :param classifier: the classifier whose accuracy we compute.
    :type data: list(tuple(list(any), str))
    :param data: A list with tuples of the form (list with features, label)
    :rtype float
    :return accuracy (range 0 - 1).
    """
    correct = 0
    for d, gold in data:
        predicted = classifier.classify(d)
        correct += predicted == gold
    return correct/len(data)


def apply_extractor(extractor_f, data):
    """
    Helper function:
    Apply a feature extraction method to a labeled dataset.
    :type extractor_f: (str, str, str, str) -> list(any)
    :param extractor_f: the feature extractor, that takes as input V, N1, P, N2 (all strings) and returns a list of features
    :type data: list(tuple(str))
    :param data: a list with tuples of the form (id, V, N1, P, N2, label)

    :rtype list(tuple(list(any), str))
    :return a list with tuples of the form (list with features, label)
    """
    r = []
    for d in data:
        r.append((extractor_f(*d[1:-1]), d[-1]))
    return r


class NltkClassifierWrapper:
    """
    This is a little wrapper around the nltk classifiers so that we can interact with them
    in the same way as the Naive Bayes classifier.
    """
    def __init__(self, classifier_class, train_features, **kwargs):
        """

        :type classifier_class: a class object of nltk.classify.api.ClassifierI
        :param classifier_class: the kind of classifier we want to create an instance of.
        :type train_features: list(tuple(list(any), str))
        :param train_features: A list with tuples of the form (list with features, label)
        :param kwargs: additional keyword arguments for the classifier, e.g. number of training iterations.
        :return None
        """
        self.classifier_obj = classifier_class.train(
            [(NltkClassifierWrapper.list_to_freq_dict(d), c) for d, c in train_features], **kwargs)

    @staticmethod
    def list_to_freq_dict(d):
        """
        :param d: list(any)
        :param d: list of features
        :rtype dict(any, int)
        :return: dictionary with feature counts.
        """
        return Counter(d)

    def classify(self, d):
        """
        :param d: list(any)
        :param d: list of features
        :rtype str
        :return: most likely class
        """
        return self.classifier_obj.classify(NltkClassifierWrapper.list_to_freq_dict(d))

    def show_most_informative_features(self, n = 10):
        self.classifier_obj.show_most_informative_features(n)

# End helper functions

# ==============================================
# Section I: Language Identification [60 marks]
# ==============================================

# Question 1 [7 marks]
def clean(word):
    if not word.isalpha():
        word = ''
    return word.lower()

def train_LM(corpus):
    '''
    Build a bigram letter language model using LgramModel
    based on the all-alpha subset the entire corpus

    :type corpus: nltk.corpus.CorpusReader
    :param corpus: An NLTK corpus
    :rtype: LgramModel
    :return: A padded letter bigram model based on nltk.model.NgramModel
    '''
    # subset the corpus to only include all-alpha tokens
    
    
    corpus_tokens = [clean(w) for w in corpus.words() if clean(w) != '']
    
    # Return a smoothed padded bigram letter language model
    lm = LgramModel(2, corpus_tokens, pad_left = True, pad_right = True)
    
    return lm



# Question 2 [7 marks]

def tweet_ent(file_name, bigram_model):
    '''
    Using a character bigram model, compute sentence entropies
    for a subset of the tweet corpus, removing all non-alpha tokens and
    tweets with less than 5 all-alpha tokens

    :type file_name: str
    :param file_name: twitter file to process
    :rtype: list(tuple(float,list(str)))
    :return: ordered list of average entropies and tweets'''

    # Clean up the tweet corpus to remove all non-alpha
    # tokens and tweets with less than 5 (remaining) tokens
    list_of_tweets = xtwc.sents(file_name)
    
    cleaned_LOT_full = [[clean(word) for word in tweet if clean(word) != '']for tweet in list_of_tweets ]
    cleaned_list_of_tweets = [tweet for tweet in cleaned_LOT_full if len(tweet) >= 5]
    # Construct a list of tuples of the form: (entropy,tweet)
    #  for each tweet in the cleaned corpus, where entropy is the
    #  average word for the tweet, and return the list of
    #  (entropy,tweet) tuples sorted by entropy
    def entropy(tweet):
        tweet_en = [bigram_model.entropy(word, pad_left=True, pad_right=True, perItem = True) for word in tweet]
        av_en = sum(tweet_en)/len(tweet_en)
        return av_en
    ent_tweet_pairs = [(entropy(tweet), tweet) for tweet in cleaned_list_of_tweets ]
    ent_tweet_pairs.sort(key = lambda word: word[0])

    return ent_tweet_pairs


# Question 3 [8 marks]
def open_question_3():
    '''
    Question: What differentiates the beginning and end of the list
    of tweets and their entropies?

    :rtype: str
    :return: your answer [500 chars max]
    '''
    return inspect.cleandoc("""
    High Entropy Tweets consist of
    - words missing spaces between them
    - mainly non-english words 
    - informal onomatopoeia e.g. haha, arghh
    - elongated words e.g. babyyy, sleeep 
    - informal abbreviations e.g. btw, omg, lmao
    - phonetic spellings e.g. nuffin, imma
    - word shortenings e.g. tatts, yr
    - misspellings 
    - Non-ascii words e.g. not latin alphabet, accented
    Low Entropy Tweets have many fewer instances of these and consist more common of words and 
    character sequences and more formal vocabulary""")[0:500]


# Question 4
def open_question_4() -> str:
    '''
    Problem: noise in Twitter data

    :rtype: str
    :return: your answer [500 chars max]
    '''
    return inspect.cleandoc("""
    Problems
    1. letters left after splitting clitics at apostrophies are treated as separate words
    2. common abbreviations like dm, rt and gt have high bigram entropy and are uninformative
    3. letter repetition skews the word entropy to the repeated bigram e.g. babyyyy
    4. acyonyms dont reflect the true phrase entropies

    Solutions
    1. 2. remove all 1 or 2 letter words 
    3. triples don't occur in english so remove excess letter after two repetitions
    4. expand 50 most common abbreviations of length > 3
     """)[0:500]


# Question 5 [15 marks]
def tweet_filter(list_of_tweets_and_entropies):
    '''
    Compute entropy mean, standard deviation and using them,
    likely non-English tweets in the all-ascii subset of list 
    of tweets and their letter bigram entropies

    :type list_of_tweets_and_entropies: list(tuple(float,list(str)))
    :param list_of_tweets_and_entropies: tweets and their
                                    english (brown) average letter bigram entropy
    :rtype: tuple(float, float, list(tuple(float,list(str)), list(tuple(float,list(str)))
    :return: mean, standard deviation, ascii tweets and entropies,
             non-English tweets and entropies
    '''
    br_lm = train_LM(brown)
    tw_en = tweet_ent(twitter_file_ids, br_lm)
    # Find the "ascii" tweets - those in the lowest-entropy 90%
    #  of list_of_tweets_and_entropies
    list_of_ascii_tweets_and_entropies = tw_en[:int(len(tw_en)*0.9)]

    # Extract a list of just the entropy values
    list_of_entropies = [word[0] for word in list_of_ascii_tweets_and_entropies]

    # Compute the mean of entropy values for "ascii" tweets
    mean = np.mean(list_of_entropies)

    # Compute their standard deviation
    standard_deviation = np.std(list_of_entropies)

    # Get a list of "probably not English" tweets, that is
    #  "ascii" tweets with an entropy greater than (mean + std_dev))
    threshold = mean + standard_deviation
    
    list_of_not_English_tweets_and_entropies = [pair for pair in list_of_ascii_tweets_and_entropies if pair[0] > threshold]

    # Return mean, standard_deviation,
    #  list_of_ascii_tweets_and_entropies,
    #  list_of_not_English_tweets_and_entropies
    return mean, standard_deviation, list_of_ascii_tweets_and_entropies, list_of_not_English_tweets_and_entropies



# Question 6 [15 marks]

# Question 6 [15 marks]
def open_question_6():
    """
    Suppose you are asked to find out what the average per word entropy of English is.
    - Name 3 problems with this question, and make a simplifying assumption for each of them.
    - What kind of experiment would you perform to estimate the entropy after you have these simplifying assumptions?
       Justify the main design decisions you make in your experiment.
    :rtype: str
    :return: your answer [1000 chars max]
    """
    return inspect.cleandoc("""
    Problems
    1. The per word entropy depends on many features of words, including contextual information
    2. we do not have an infinite amount of time, data or processing capabilities and sentence instances are sparse
    3. the average per word entropy must be representative of all situations english is encountered
    Assumption
    1. as english has a poor morphology take per word entropy to only depend on the order of words and metadata specifying the domain
    2. the entropy will be accurate if we take a long enough sequence of words using the Shannon-McMillan-Breiman theorem and use cross instead of true entropy
    3. a weighted combination of per word entropy from samples of all major domains will be representative
    Experiment
    -webscrape sentences from written, transcribed spoken, published and informal sources and 
    use an 5-gram model with katz backoff on data of size 10 million words
    -Train logistic regression using features from the metadata, topic, audience, source type in addition to the n-grams
    """)[:1000]


#############################################
# SECTION II - RESOLVING PP ATTACHMENT AMBIGUITY
#############################################

# Question 7 [15 marks]
class NaiveBayes:
    """
    Naive Bayes model with Lidstone smoothing (parameter alpha).
    """

    def __init__(self, data, alpha):
        """
        :type data: list(tuple(list(any), str))
        :param data: A list with tuples of the form (list with features, label)
        :type alpha: float
        :param alpha: \alpha value for Lidstone smoothing
        """
        self.vocab = self.get_vocab(data)
        self.alpha = alpha
        self.prior, self.likelihood = self.train(data, alpha, self.vocab)

    @staticmethod
    def get_vocab(data):
        """
        Compute the set of all possible features from the (training) data.
        :type data: list(tuple(list(any), str))
        :param data: A list with tuples of the form (list with features, label)
        :rtype: set(any)
        :return: The set of all features used in the training data for all classes.
        """
        return set([f for tuple in data for f in tuple[0]])
        

    @staticmethod
    def train(data, alpha, vocab):
        """
        Estimates the prior and likelihood from the data with Lidstone smoothing.

        :type data: list(tuple(list(any), str))
        :param data: A list of tuples ([f1, f2, ... ], c) with the first element
                     being a list of features and the second element being its class.

        :type alpha: float
        :param alpha: \alpha value for Lidstone smoothing

        :type vocab: set(any)
        :param vocab: The set of all features used in the training data for all classes.


        :rtype: tuple(dict(str, float), dict(str, dict(any, float)))
        :return: Two dictionaries: the prior and the likelihood (in that order).
        We expect the returned values to relate as follows to the probabilities:
            prior[c] = P(c)
            likelihood[c][f] = P(f|c)
        """
        assert alpha >= 0.0
        
        # Compute raw frequency distributions
        # Compute prior (MLE). Compute likelihood with smoothing.
        
        prior = {}
        likelihood = {}
        class_count = {}
        N = len(data)
        V = len(vocab)
        
        for points in data:
            if points[1] in prior.keys():
                prior[points[1]] += 1
                class_count[points[1]] += len(points[0]) 
            else:
                prior[points[1]] = 1
                class_count[points[1]] = len(points[0])      

        for classes in class_count.keys():
            class_count[classes] = 1/(class_count[classes] + alpha*V)

        for classes in class_count.keys():
            likelihood[classes] = {}
            for features in vocab:
                likelihood[classes][features] = alpha

        for points in data:
        
            for features in points[0]:
                    likelihood[points[1]][features] += 1

        for classes in likelihood.keys():
            for features in likelihood[classes].keys():
                likelihood[classes][features] *= class_count[classes]

        for classes in prior.keys():
            prior[classes] /= N
            
        return prior, likelihood


    
    def prob_classify(self, d):
        """
        Compute the probability P(c|d) for all classes.
        :type d: list(any)
        :param d: A list of features.
        :rtype: dict(str, float)
        :return: The probability p(c|d) for all classes as a dictionary.
        """
        
        prob = {}
        for classes in self.prior.keys():
            prob[classes] = self.prior[classes] 

            for feature in d:
                if feature in self.likelihood[classes].keys():
                    prob[classes] *= self.likelihood[classes][feature] 
        
        denom = 0
        for classes in self.prior.keys():
            mult = self.prior[classes]
            for features in d:
                if features in self.vocab:
                    mult *= self.likelihood[classes][features]
            denom += mult
        sum1 = 0
        for classes in prob.keys():
            prob[classes] /= denom
            sum1 += prob[classes]
        
        return prob

    def classify(self, d):
        """
        Compute the most likely class of the given "document" with ties broken arbitrarily.
        :type d: list(any)
        :param d: A list of features.
        :rtype: str
        :return: The most likely class.
        """
        valmax = 0
        prob = self.prob_classify(d)
        for classes in prob.keys():
            if prob[classes] > valmax:
                valmax = prob[classes]
                
                argmax = classes
                
        return argmax



# Question 8 [10 marks]

def open_question_8() -> str:
    """
    How do you interpret the differences in accuracy between the different ways to extract features?
    :rtype: str
    :return: Your answer of 500 characters maximum.
    """
    return inspect.cleandoc("""
    -English has a scant morphology, so n1, v, p, and n2 are less accurate than the underlying structure of the combination
    -p must fit the grammar and semantics of both n2 and v or n1 so p encodes the most information about its attachment
    -n2 is least informative, nouns can follow most prepositions and are open class so make more sparse features
    -NBM is generative so performs worse on larger data sets than the discriminative LRM
    -weighting important features is more indicative than relative freq.""")[:500]


# Feature extractors used in the table:
# see your_feature_extractor for documentation on arguments and types.
def feature_extractor_1(v, n1, p, n2):
    return [v]


def feature_extractor_2(v, n1, p, n2):
    return [n1]


def feature_extractor_3(v, n1, p, n2):
    return [p]


def feature_extractor_4(v, n1, p, n2):
    return [n2]


def feature_extractor_5(v, n1, p, n2):
    return [("v", v), ("n1", n1), ("p", p), ("n2", n2)]


# Question 9.1 [5 marks]
def your_feature_extractor(v, n1, p, n2):
    """
    Takes the head words and produces a list of features. The features may
    be of any type as long as they are hashable.
    :type v: str
    :param v: The verb.
    :type n1: str
    :param n1: Head of the object NP.
    :type p: str
    :param p: The preposition.
    :type n2: str
    :param n2: Head of the NP embedded in the PP.
    :rtype: list(any)
    :return: A list of features produced by you.
    """
    v, n1, p, n2 = v.lower(), n1.lower(), p.lower(), n2.lower()
    features = [("v", v), ("n1", n1), ("p", p), ("n1p", n1 +  p ), ("vp", v +p), ("pn2", p +  n2), 
    ("vpn2", v + p +n2), ('n1_type', n1.isalpha()), ('n2_type', n2.isalpha()), ('tag', nltk.pos_tag(v))]
    return features
    


# Question 9.2 [10 marks]
def open_question_9():
    """
    Briefly describe your feature templates and your reasoning for them.
    Pick 3 examples of informative features and discuss why they make sense or why they do not make sense
    and why you think the model relies on them.
    :rtype: str
    :return: Your answer of 1000 characters maximum.
    """
    
    return inspect.cleandoc("""-n2 is most significant in its co-occurences with p, v or n1 
    as the semantic agreement is its most useful discernable attribute
    -the prepositional phrase p+n2 as a whole is more likely to have the same attachment than either word individually 
    but is more sparse so we keep n2 and p
    -p can either attach to n1 or v so matching instances of v+p or n1+p indicate common semantic and correct grammatical attachments
    -in the WSJ, dates and figures are common but matching numbers exactly is rare so the pp attachment may depend on noun type, mainly alpha vs numeric
    Feature examples
    'attributed to' is a very common verb phrase so much less likely to indicate noun attachment
    'followingofreport' increases the probability of verb phrase attachment but 'following of' is not grammatically correct unless following is a noun
    perhaps the POS-tagger made a mistake with multiple instances of similar phrases
    'until' is strongly correlated with processes so is less likely to indicate noun over verb phrases""")[:1000]


"""
Format the output of your submission for both development and automarking. 
!!!!! DO NOT MODIFY THIS PART !!!!!
"""

def answers():
    # Global variables for answers that will be used by automarker
    global ents, lm
    global best10_ents, worst10_ents, mean, std, best10_ascci_ents, worst10_ascci_ents
    global best10_non_eng_ents, worst10_non_eng_ents
    global answer_open_question_4, answer_open_question_3, answer_open_question_6,\
        answer_open_question_8, answer_open_question_9
    global ascci_ents, non_eng_ents

    global naive_bayes
    global acc_extractor_1, naive_bayes_acc, lr_acc, logistic_regression_model, dev_features

    print("*** Part I***\n")

    print("*** Question 1 ***")
    print('Building brown bigram letter model ... ')
    lm = train_LM(brown)
    print('Letter model built')

    print("*** Question 2 ***")
    ents = tweet_ent(twitter_file_ids, lm)
    print("Best 10 english entropies:")
    best10_ents = ents[:10]
    ppEandT(best10_ents)
    print("Worst 10 english entropies:")
    worst10_ents = ents[-10:]
    ppEandT(worst10_ents)

    print("*** Question 3 ***")
    answer_open_question_3 = open_question_3()
    print(answer_open_question_3)

    print("*** Question 4 ***")
    answer_open_question_4 = open_question_4()
    print(answer_open_question_4)

    print("*** Question 5 ***")
    mean, std, ascci_ents, non_eng_ents = tweet_filter(ents)
    print('Mean: {}'.format(mean))
    print('Standard Deviation: {}'.format(std))
    print('ASCII tweets ')
    print("Best 10 English entropies:")
    best10_ascci_ents = ascci_ents[:10]
    ppEandT(best10_ascci_ents)
    print("Worst 10 English entropies:")
    worst10_ascci_ents = ascci_ents[-10:]
    ppEandT(worst10_ascci_ents)
    print('--------')
    print('Tweets considered non-English')
    print("Best 10 English entropies:")
    best10_non_eng_ents = non_eng_ents[:10]
    ppEandT(best10_non_eng_ents)
    print("Worst 10 English entropies:")
    worst10_non_eng_ents = non_eng_ents[-10:]
    ppEandT(worst10_non_eng_ents)

    print("*** Question 6 ***")
    answer_open_question_6 = open_question_6()
    print(answer_open_question_6)


    print("*** Part II***\n")

    print("*** Question 7 ***")
    naive_bayes = NaiveBayes(apply_extractor(feature_extractor_5, ppattach.tuples("training")), 0.1)
    naive_bayes_acc = compute_accuracy(naive_bayes, apply_extractor(feature_extractor_5, ppattach.tuples("devset")))
    print(f"Accuracy on the devset: {naive_bayes_acc * 100}%")

    print("*** Question 8 ***")
    answer_open_question_8 = open_question_8()
    print(answer_open_question_8)

    # This is the code that generated the results in the table of the CW:

    # A single iteration of suffices for logistic regression for the simple feature extractors.
    
    extractors_and_iterations = [feature_extractor_1, feature_extractor_2, feature_extractor_3, feature_extractor_4, feature_extractor_5]
    
    print("Extractor    |  Accuracy")
    print("------------------------")
    
    for i, ex_f in enumerate(extractors_and_iterations, start=1):
        training_features = apply_extractor(ex_f, ppattach.tuples("training"))
        dev_features = apply_extractor(ex_f, ppattach.tuples("devset"))
    
        a_logistic_regression_model = NltkClassifierWrapper(MaxentClassifier, training_features, max_iter=6, trace=0)
        lr_acc = compute_accuracy(a_logistic_regression_model, dev_features)
        print(f"Extractor {i}  |  {lr_acc*100}")


    print("*** Question 9 ***")
    training_features = apply_extractor(your_feature_extractor, ppattach.tuples("training"))
    dev_features = apply_extractor(your_feature_extractor, ppattach.tuples("devset"))
    logistic_regression_model = NltkClassifierWrapper(MaxentClassifier, training_features, max_iter=10)
    lr_acc = compute_accuracy(logistic_regression_model, dev_features)

    print("30 features with highest absolute weights")
    logistic_regression_model.show_most_informative_features(30)

    print(f"Accuracy on the devset: {lr_acc*100}")

    answer_open_question_9 = open_question_9()
    print("Answer to open question:")
    print(answer_open_question_9)




if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--answers':
        from autodrive_embed import run, carefulBind
        import adrive1

        with open("userErrs.txt", "w") as errlog:
            run(globals(), answers, adrive1.extract_answers, errlog)
    else:
        answers()
