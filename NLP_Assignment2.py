#  d1768eaf-9ce2-401d-85e8-b114a8b739c9. confimation number

from re import T
import nltk, inspect, sys, hashlib

from nltk.corpus import brown

# module for computing a Conditional Frequency Distribution
from nltk.probability import ConditionalFreqDist

# module for computing a Conditional Probability Distribution
from nltk.probability import ConditionalProbDist, LidstoneProbDist

from nltk.tag import map_tag

from adrive2 import trim_and_warn

assert map_tag('brown', 'universal', 'NR-TL') == 'NOUN', '''
Brown-to-Universal POS tag map is out of date.'''
import numpy as np
import math

def lidstone_estimator(fd):
        return LidstoneProbDist(fd, gamma = 0.001, bins = fd.B() + 1)
    
def lidstone2_estimator(fd):
    return LidstoneProbDist(fd, gamma = 0.001, bins = fd.B())

class HMM:
    def __init__(self, train_data):
        """
        Initialise a new instance of the HMM.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        """
        self.train_data = train_data

        # Emission and transition probability distributions
        self.emission_PD = None
        self.transition_PD = None
        self.states = []
        self.viterbi = []
        self.backpointer = []

    # Q1

    # Compute emission model using ConditionalProbDist with a LidstoneProbDist estimator.
    #   To achieve the latter, pass a function
    #    as the probdist_factory argument to ConditionalProbDist.
    #   This function should take 3 arguments
    #    and return a LidstoneProbDist initialised with +0.001 as gamma and an extra bin.
    #   See the documentation/help for ConditionalProbDist to see what arguments the
    #    probdist_factory function is called with.
    
    
    
    def emission_model(self, train_data):
        """Compute an emission model based on labelled training data.
        Don't forget to lowercase the observation otherwise it mismatches the test data.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The emission probability distribution and a list of the states
        :rtype: Tuple[ConditionalProbDist, list(str)]
        """
        # TODO prepare data
        data = [ (tag, word.lower()) for data in train_data for (word, tag) in data]
        
        # Don't forget to lowercase the observation otherwise it mismatches the test data
        # Do NOT add <s> or </s> to the input sentences
        # TODO compute the emission model
        
        emission_FD = ConditionalFreqDist(data) #frequency distribution
        
        self.emission_PD = ConditionalProbDist(emission_FD, lidstone_estimator ) 
        #smoothed probability distribution
        for s in data:
            if not s[0] in self.states:
                self.states.append(s[0])
        return self.emission_PD, self.states

    # Q1

    # Access function for testing the emission model
    # For example model.elprob('VERB','is') might be -1.4
    def elprob(self, state, word):
        """
        The log of the estimated probability of emitting a word from a state

        :param state: the state name
        :type state: str
        :param word: the word
        :type word: str
        :return: log base 2 of the estimated emission probability
        :rtype: float
        """
        p = self.emission_PD[state].prob(word) #Word given tag probability
        log_prob = np.log2(p) # Log base 2 probability
        
        return float(log_prob)
    
    # Q2
    # Compute transition model using ConditionalProbDist with the same
    #  estimator as above (but without the extra bin)
    # See comments for emission_model above for details on the estimator.
    def transition_model(self, train_data):
        """
        Compute a transition model using a ConditionalProbDist based on
          labelled data.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The transition probability distribution
        :rtype: ConditionalProbDist
        """
        
        # TODO: prepare the data
        
        predata = [ [tag for (word, tag) in data] for data in train_data]
        # The data object should be an array of tuples of conditions and observations,
        # in our case the tuples will be of the form (tag_(i),tag_(i+1)).
        # DON'T FORGET TO ADD THE START SYMBOL </s> and the END SYMBOL </s>
        data = []
        for s in predata: #s is a sentences
            data.append((s[-1], '</s>')) #end of sentence
            data.append(('<s>', s[0])) # beginning of sentence
            for i in range(len(s)-1): # all other tags in sentence
                data.append((s[i], s[i+1]))
        # TODO compute the transition model

        transition_FD = ConditionalFreqDist(data)
        self.transition_PD = ConditionalProbDist(transition_FD, lidstone2_estimator)
    
        return self.transition_PD

    # Q2
    # Access function for testing the transition model
    # For example model.tlprob('VERB','VERB') might be -2.4
    def tlprob(self, state1, state2):
        """
        The log of the estimated probability of a transition from one state to another

        :param state1: the first state name
        :type state1: str
        :param state2: the second state name
        :type state2: str
        :return: log base 2 of the estimated transition probability
        :rtype: float
        """
        log_prob = np.log2(self.transition_PD[state1].prob(state2))
        
        return float(log_prob)

    # Train the HMM
    def train(self):
        """
        Trains the HMM from the training data
        """
        self.emission_model(self.train_data)
        self.transition_model(self.train_data)

         # Part B: Implementing the Viterbi algorithm.

    # Q3
    # Initialise data structures for tagging a new sentence.
    # Describe the data structures with comments.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD
    # Input: first word in the sentence to tag and the total number of observations.
    def initialise(self, observation, number_of_observations):
        """
        Initialise data structures self.viterbi and self.backpointer for tagging a new sentence.

        :param observation: the first word in the sentence to tag
        :type observation: str
        :param number_of_observations: the number of observations
        :type number_of_observations: int
        """
        # Initialise step 0 of viterbi, including
        #  transition from <s> to observation
        num_of_tags = len(self.states) # number of rows of viterbi matrix
        self.viterbi = np.zeros((num_of_tags, number_of_observations)) # extra column for the transition from the last tag to </s>.
        # every row corresponds to a tag in self.states and every column to a transition between states or <s> to the first state.
        self.backpointer = np.zeros((num_of_tags, number_of_observations)) # Initialise step 0 of backpointer
        # each row and column of the backpointer matrix refers to the index of the row in the previous column that generated the minimum cost
        
        for i in range(num_of_tags):
            state = self.states[i]
            emiss = self.elprob(state, observation) # emission probabilities of the first word given the state corresponding to the row
            trans = self.tlprob('<s>', state) # transition probabilities, start of sentence boundary tag to first the state corresponding to the row
            # Alternatively can be seen as probability that you start with the given state
            self.viterbi[i, 0] = -trans-emiss #negative log probabilities (costs) 
            #initialise the first column of the backpointer to point to nothing
            self.backpointer[i, 0] = None
        # Q3
    # Access function for testing the viterbi data structure
    # For example model.get_viterbi_value('VERB',2) might be 6.42
    def get_viterbi_value(self, state, step):
        """
        Return the current value from self.viterbi for
        the state (tag) at a given step

        :param state: A tag name
        :type state: str
        :param step: The (0-origin) number of a step:  if negative,
          counting backwards from the end, i.e. -1 means the last step
        :type step: int
        :return: The value (a cost) for state as of step
        :rtype: float
        """
        ind = self.states.index(state) # find viterbi row entry
        
        return float(self.viterbi[ind, step])

    # Q3
    # Access function for testing the backpointer data structure
    # For example model.get_backpointer_value('VERB',2) might be 'NOUN'
    def get_backpointer_value(self, state, step):
        """
        Return the current backpointer from self.backpointer for
        the state (tag) at a given step

        :param state: A tag name
        :type state: str
        :param step: The (0-origin) number of a step:  if negative,
          counting backwards from the end, i.e. -1 means the last step
        :type step: int
        :return: The state name to go back to at step-1
        :rtype: str
        """
        ind = int(self.states.index(state)) # find row from state
        bp = int(self.backpointer[ind, step]) # find row from backpointer
        
        return self.states[bp] #find state from row index

   # Q4a
    # Tag a new sentence using the trained model and already initialised data structures.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD.
    # Update the self.viterbi and self.backpointer data structures.
    # Describe your implementation with comments.
    def tag(self, observations):
        """Tag a new sentence using the trained model and already initialised data structures.
        :param observations: List of words (a sentence) to be tagged
        :type observations: list(str)
        :return: List of tags corresponding to each word of the input """
        sent_length = len(observations)
        num_of_tags = len(self.states)
        for j in range(1, sent_length): # iterate over each column after having computed entries for all rows in previous column
            for i in range(num_of_tags): # iterate over each row, each corresponding to a state
                transition_probs = [self.tlprob(self.states[h], self.states[i]) for h in range(num_of_tags)]
                # List of transition log probabilities from all states to the state in row i
                emission_prob = self.elprob(self.states[i], observations[j])
                # emission log probabilities from state in row i+1 to the j+1th word in the sentence
                prev_probs = list(self.viterbi[:,j -1]) # the cost of the previous j words 
                # having ended in the state corresponding to its position in the list 
                trans_prev = [-transition_probs[h]+prev_probs[h] for h in range(num_of_tags)] 
                # the costs of the previous j words and the transition from the final state h of those previous words to state 
                m = min(trans_prev)
                self.viterbi[i, j] = m - emission_prob #discard all but the most probable sequence up to that state and observation
                self.backpointer[i, j] = trans_prev.index(m) # point to where that minimum value came from in the previous column
        
        # Termination step
        last_col = sent_length
    
        final_trans_costs = [self.viterbi[i, last_col - 1] - self.tlprob(self.states[i], '</s>') for i in range(num_of_tags)]
        # Take the cost of the sentence up until the final word and add the cost of transitioning to the end of the sentence
    
        # We don't choose a minimum cost value for every row in this column so it's generated by the same row in the previous column
        bestpathpointer = final_trans_costs.index(min(final_trans_costs))
        # find the minimum of the final costs and let its index be the first backpointer from which to construct the hidden state sequence
        tag_index = [bestpathpointer]
        tags = []
        for i in range(1, sent_length):
            tag_index.append(int(self.backpointer[tag_index[i-1], -i]))
        for i in range(sent_length):
            tags.append(self.states[tag_index[-i-1]])
        return tags

    def tag_sentence(self, sentence):
        """
        Initialise the HMM, lower case and tag a sentence. Returns a list of tags.
        :param sentence: the sentence
        :type sentence: list(str)
        :rtype: list(str)
        """
        lower_sent = [word.lower() for word in sentence] #lower case input sentence to tag
        self.initialise(lower_sent[0], len(lower_sent))   #initialise viterbi with first word and length of sentence
         
        return self.tag(lower_sent)


def answer_question4b():
    """
    Report a hand-chosen tagged sequence that is incorrect, correct it
    and discuss
    :rtype: list(tuple(str,str)), list(tuple(str,str)), str
    :return: incorrectly tagged sequence, correctly tagged sequence and your answer [max 280 chars]
    """
    # One sentence, i.e. a list of word/tag pairs, in two versions
    #  1) As tagged by your HMM
    #  2) With wrong tags corrected by hand
    tagged_sequence = [('Tooling', 'X'), ('through', 'ADP'), ('Sydney', 'NOUN'), ('on', 'ADP'), ('his', 'DET'), ('way', 'NOUN'), ('to', 'ADP'), ('race', 'NOUN'), ('in', 'ADP'), ('the', 'DET'), ('New', 'ADJ'), ('Zealand', 'X'), ('Grand', 'X'), ('Prix', 'X'), (',', '.'), ("Britain's", 'X'), ('balding', 'X'), ('Ace', 'X'), ('Driver', 'X'), ('Stirling', 'X'), ('Moss', 'X'), (',', '.'), ('31', 'NUM'), (',', '.'), ('all', 'PRT'), ('but', 'CONJ'), ('smothered', 'ADV'), ('himself', 'PRON'), ('in', 'ADP'), ('his', 'DET'), ('own', 'ADJ'), ('exhaust', 'NOUN'), ('of', 'ADP'), ('self-crimination', 'NUM'), ('.', '.')]
    correct_sequence = [('Tooling', 'VERB'), ('through', 'ADP'), ('Sydney', 'NOUN'), ('on', 'ADP'), ('his', 'DET'), ('way', 'NOUN'), ('to', 'PRT'), ('race', 'VERB'), ('in', 'ADP'), ('the', 'DET'), ('New', 'NOUN'), ('Zealand', 'NOUN'), ('Grand', 'NOUN'), ('Prix', 'NOUN'), (',', '.'), ("Britain's", 'NOUN'), ('balding', 'ADJ'), ('Ace', 'NOUN'), ('Driver', 'NOUN'), ('Stirling', 'NOUN'), ('Moss', 'NOUN'), (',', '.'), ('31', 'NUM'), (',', '.'), ('all', 'PRT'), ('but', 'ADP'), ('smothered', 'VERB'), ('himself', 'PRON'), ('in', 'ADP'), ('his', 'DET'), ('own', 'ADJ'), ('exhaust', 'NOUN'), ('of', 'ADP'), ('self-crimination', 'NOUN'), ('.', '.')]
    
    # Why do you think the tagger tagged this example incorrectly?
    answer = inspect.cleandoc("""'on his way to'+ "location" is simpler and so more common than + VERB and 'way' implies destination. tlprob(ADP, NOUN) is higher than tlprob(PRT, NOUN) with respect to the transistions to VERBs. Mislabelling 'to' caused the POS tag ambiguity of 'race' caused it to be mislabelled too.""")

    return tagged_sequence, correct_sequence, trim_and_warn("Q4a", 280, answer)


# Q5a
def hard_em(labeled_data, unlabeled_data, k):
    """
    Run k iterations of hard EM on the labeled and unlabeled data.
    Follow the pseudo-code in the coursework instructions.

    :param labeled_data:
    :param unlabeled_data:
    :param k: number of iterations
    :type k: int
    :return: HMM model trained with hard EM.
    :rtype: HMM
    """
    model = HMM(labeled_data)
    model.train()
    
    for i in range(k):
        new_labels = [list(zip(s, model.tag_sentence(s))) for s in unlabeled_data]
        new_data =  new_labels + labeled_data
        # print(labeled_data[0], new_labels[0])
        model = HMM(new_data)
        model.train()
        
    return model

def answer_question5b():
    """
    Sentence:  In    fact  he    seemed   delighted  to  get   rid  of  them   .
    Gold POS:  ADP   NOUN  PRON  VERB     VERB      PRT  VERB  ADJ  ADP  PRON  .
    T_0     :  PRON  VERB  NUM    ADP     ADJ       PRT  VERB  NUM  ADP  PRON  .
    T_k     :  PRON  VERB  PRON  VERB     ADJ       PRT  VERB  NUM  ADP  NOUN  .

    1) T_0 erroneously tagged "he" as "NUM" and T_k correctly identifies it as "PRON".
        Speculate why additional unlabeled data might have helped in that case.
        Refer to the training data (inspect the 20 sentences!).
    2) Where does T_k mislabel a word but T_0 is correct? Why do you think did hard EM hurt in that case?

    :rtype: str
    :return: your answer [max 500 chars]
    """
    return trim_and_warn("Q5b", 500, inspect.cleandoc("""VERB PRON has count 4, VERB NUM and
    NUM ADP count 1 and PRON ADP doesn't occur (count = 0.001), with additional data, 'he' is likely to appear (it doesn't for T0) so the 
    difference in counts of PRON 'he' and NUM 'he' would be less, so probabilites of (ADP, PRON) and (PRON, 'he') increase, scale with the counts.
    ADP NOUN occurs 17 times more than ADP PRON so hard EM amplifies the larger bias in the tag probabilities, 
    'them' has 2 counts in T0, even 1 mislabelling makes 'NOUN' more likely."""))


def answer_question6():
    """
    Suppose you have a hand-crafted grammar that has 100% coverage on
        constructions but less than 100% lexical coverage.
        How could you use a POS tagger to ensure that the grammar
        produces a parse for any well-formed sentence,
        even when it doesn't recognise the words within that sentence?

    :rtype: str
    :return: your answer [max 500 chars]
    """
    return trim_and_warn("Q6", 500, inspect.cleandoc("""
    You could use CKY algorithm for bottom up parsing, for words not covered by the PCFG you use the POS tagger 
    to generate a new preterminal grammar rules. Then compute the probabilities of all rules including counts of the POS labelled words chosen in the parse trees, 
    then iterate these steps. Incorrectly labelled words might make the parser perform worse but a high 
    accuracy POS tagger making rules with multiple tag options with probabilities (soft labels) will converge to a local optimal value."""))

def answer_question7():
    """
    Why else, besides the speedup already mentioned above, do you think we
    converted the original Brown Corpus tagset to the Universal tagset?
    What do you predict would happen if we hadn't done that?  Why?

    :rtype: str
    :return: your answer [max 500 chars]
    """

    return trim_and_warn("Q7", 500, inspect.cleandoc("""\
    At 4500 sentences for training, this is still a small training set, so 
    using a larger tagset will introduce a more sparsity causing a lot of similar probabilities from smoothing
    for unseen words or tags, so smaller tolerances for correct labelling, so lower accuracy.
    For N words, each will have more possible POS tags, up to m, and so generate many combinations of tags, N^m. 
    Fine grain POS tags rely more on wider context e.g. previous 2 tags instead but higher order HMMs also increase sparsity."""))



def compute_acc(hmm, test_data, print_mistakes):
    """
    Computes accuracy (0.0 - 1.0) of model on some data.
    :param hmm: the HMM
    :type hmm: HMM
    :param test_data: the data to compute accuracy on.
    :type test_data: list(list(tuple(str, str)))
    :param print_mistakes: whether to print the first 10 model mistakes
    :type print_mistakes: bool
    :return: float
    """
    # TODO: modify this to print the first 10 sentences with at least one mistake if print_mistakes = True
    correct = 0
    incorrect = 0
    c = 0
    for sentence in test_data:
        s = [word for (word, tag) in sentence]
        tags = hmm.tag_sentence(s)
        inc = 0
        c +=1
        goldt = []
        for ((word, gold), tag) in zip(sentence, tags):
            if tag == gold:
                correct += 1
            else:
                incorrect += 1
                inc += 1
        if print_mistakes == True and inc != 0 and c <= 10:
            print(s)
                
    return float(correct) / (correct + incorrect)



# Useful for testing
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    # http://stackoverflow.com/a/33024979
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def answers():
    global tagged_sentences_universal, test_data_universal, \
        train_data_universal, model, test_size, train_size, ttags, \
        correct, incorrect, accuracy, \
        good_tags, bad_tags, answer4b, answer5, answer6, answer7, answer5b, \
        t0_acc, tk_acc

    # Load the Brown corpus with the Universal tag set.
    tagged_sentences_universal = brown.tagged_sents(categories='news', tagset='universal')

    # Divide corpus into train and test data.
    test_size = 500
    train_size = len(tagged_sentences_universal) - test_size

    # tail test set
    test_data_universal = tagged_sentences_universal[-test_size:]  # [:test_size]
    train_data_universal = tagged_sentences_universal[:train_size]  # [test_size:]
#     if hashlib.md5(''.join(map(lambda x: x[0],
#                                train_data_universal[0] + train_data_universal[-1] + test_data_universal[0] +
#                                test_data_universal[-1])).encode(
#             'utf-8')).hexdigest() != '164179b8e679e96b2d7ff7d360b75735':
#         print('!!!test/train split (%s/%s) incorrect -- this should not happen, please contact a TA !!!' % (
#         len(train_data_universal), len(test_data_universal)), file=sys.stderr)

    # Create instance of HMM class and initialise the training set.
    model = HMM(train_data_universal)

    # Train the HMM.
    model.train()


    # Some preliminary sanity checks
    # Use these as a model for other checks
    e_sample = model.elprob('VERB', 'is')
    if not (type(e_sample) == float and e_sample <= 0.0):
        print('elprob value (%s) must be a log probability' % e_sample, file=sys.stderr)

    t_sample = model.tlprob('VERB', 'VERB')
    if not (type(t_sample) == float and t_sample <= 0.0):
        print('tlprob value (%s) must be a log probability' % t_sample, file=sys.stderr)

    if not (type(model.states) == list and \
            len(model.states) > 0 and \
            type(model.states[0]) == str):
        print('model.states value (%s) must be a non-empty list of strings' % model.states, file=sys.stderr)

    print('states: %s\n' % model.states)

    ######
    # Try the model, and test its accuracy [won't do anything useful
    #  until you've filled in the tag method
    ######
    s = 'the cat in the hat came back'.split()
    ttags = model.tag_sentence(s)
    print("Tagged a trial sentence:\n  %s" % list(zip(s, ttags)))

    v_sample = model.get_viterbi_value('VERB', 5)
    if not (type(v_sample) == float and 0.0 <= v_sample):
        print('viterbi value (%s) must be a cost' % v_sample, file=sys.stderr)

    b_sample = model.get_backpointer_value('VERB', 5)
    if not (type(b_sample) == str and b_sample in model.states):
        print('backpointer value (%s) must be a state name' % b_sample, file=sys.stderr)

    # check the model's accuracy (% correct) using the test set
    accuracy = compute_acc(model, test_data_universal, print_mistakes=True)
    print('\nTagging accuracy for test set of %s sentences: %.4f' % (test_size, accuracy))

    #Tag the sentence again to put the results in memory for automarker.
    model.tag_sentence(s)

    # Question 5a
    # Set aside the first 20 sentences of the training set
    num_sentences = 20
    semi_supervised_labeled = train_data_universal[:num_sentences]  # type list(list(tuple(str, str)))
    semi_supervised_unlabeled = [[word for (word, tag) in sent] for sent in train_data_universal[num_sentences:]]  # type list(list(str))
    print("Running hard EM for Q5a. This may take a while...")
    t0 = hard_em(semi_supervised_labeled, semi_supervised_unlabeled, 0) # 0 iterations
    tk = hard_em(semi_supervised_labeled, semi_supervised_unlabeled, 3)
    print("done.")

    t0_acc = compute_acc(t0, test_data_universal, print_mistakes=False)
    tk_acc = compute_acc(tk, test_data_universal, print_mistakes=False)
    print('\nTagging accuracy of T_0: %.4f' % (t0_acc))
    print('\nTagging accuracy of T_k: %.4f' % (tk_acc))
    ########

    # Print answers for 4b, 5b, 6 and 7.
    bad_tags, good_tags, answer4b = answer_question4b()
    print('\nA tagged-by-your-model version of a sentence:')
    print(bad_tags)
    print('The tagged version of this sentence from the corpus:')
    print(good_tags)
    print('\nDiscussion of the difference:')
    print(answer4b)
    answer5b = answer_question5b()
    print("\nFor Q5b:")
    print(answer5b)
    answer6 = answer_question6()
    print('\nFor Q6:')
    print(answer6)
    answer7 = answer_question7()
    print('\nFor Q7:')
    print(answer7)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--answers':
        import adrive2
        from autodrive_embed import run, carefulBind

        with open("userErrs.txt", "w") as errlog:
            run(globals(), answers, adrive2.a2answers, errlog)
    else:
        answers()
