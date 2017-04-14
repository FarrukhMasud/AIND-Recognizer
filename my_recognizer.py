import warnings
from asl_data import SinglesData
from hmmlearn.hmm import GaussianHMM


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for i in range(test_set.num_items):
        max_prob = -100000000
        guess = ''
        d = dict()
        x = test_set.get_item_Xlengths(i)
        for w in models:
            m: GaussianHMM = models[w]
            try:
                s = m.score(x[0], x[1])
                d[w] = s
                if s > max_prob:
                    max_prob = s
                    guess = w
            except:
                d[w] = -100000000
                pass
        probabilities.append(d)
        guesses.append(guess)
    return probabilities, guesses
