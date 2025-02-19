# # Natural Language Toolkit: Machine Translation
# #
# # Copyright (C) 2001-2024 NLTK Project
# # Author: Uday Krishna <udaykrishna5@gmail.com>
# # Contributor: Tom Aarsen
# # URL: <https://www.nltk.org/>
# # For license information, see LICENSE.TXT


# from itertools import chain, product
# from typing import Callable, Iterable, List, Tuple

# from nltk.corpus import WordNetCorpusReader, wordnet
# from nltk.stem.api import StemmerI
# from nltk.stem.porter import PorterStemmer
# from .tokenizer import default_tokenize_func

# def _generate_enums(
#     predictions: Iterable[str],
#     reference: Iterable[str],
#     preprocess: Callable[[str], str] = str.lower,
# ) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
#     """
#     Takes in pre-tokenized inputs for predictions and reference and returns
#     enumerated word lists for each of them

#     :param predictions: pre-tokenized predictions
#     :param reference: pre-tokenized reference
#     :preprocess: preprocessing method (default str.lower)
#     :return: enumerated words list
#     """
#     if isinstance(predictions, str):
#         raise TypeError(
#             f'"predictions" expects pre-tokenized predictions (Iterable[str]): {predictions}'
#         )

#     if isinstance(reference, str):
#         raise TypeError(
#             f'"reference" expects pre-tokenized reference (Iterable[str]): {reference}'
#         )

#     enum_predictions_list = list(enumerate(map(preprocess, predictions)))
#     enum_reference_list = list(enumerate(map(preprocess, reference)))
#     return enum_predictions_list, enum_reference_list


# def exact_match(
#     predictions: Iterable[str], reference: Iterable[str]
# ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
#     """
#     matches exact words in predictions and reference
#     and returns a word mapping based on the enumerated
#     word id between predictions and reference

#     :param predictions: pre-tokenized predictions
#     :param reference: pre-tokenized reference
#     :return: enumerated matched tuples, enumerated unmatched predictions tuples,
#              enumerated unmatched reference tuples
#     """
#     enum_predictions_list, enum_reference_list = _generate_enums(predictions, reference)
#     return _match_enums(enum_predictions_list, enum_reference_list)




# def _match_enums(
#     enum_predictions_list: List[Tuple[int, str]],
#     enum_reference_list: List[Tuple[int, str]],
# ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
#     """
#     matches exact words in predictions and reference and returns
#     a word mapping between enum_predictions_list and enum_reference_list
#     based on the enumerated word id.

#     :param enum_predictions_list: enumerated predictions list
#     :param enum_reference_list: enumerated reference list
#     :return: enumerated matched tuples, enumerated unmatched predictions tuples,
#              enumerated unmatched reference tuples
#     """
#     word_match = []
#     for i in range(len(enum_predictions_list))[::-1]:
#         for j in range(len(enum_reference_list))[::-1]:
#             if enum_predictions_list[i][1] == enum_reference_list[j][1]:
#                 word_match.append(
#                     (enum_predictions_list[i][0], enum_reference_list[j][0])
#                 )
#                 enum_predictions_list.pop(i)
#                 enum_reference_list.pop(j)
#                 break
#     return word_match, enum_predictions_list, enum_reference_list


# def _enum_stem_match(
#     enum_predictions_list: List[Tuple[int, str]],
#     enum_reference_list: List[Tuple[int, str]],
#     stemmer: StemmerI = PorterStemmer(),
# ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
#     """
#     Stems each word and matches them in predictions and reference
#     and returns a word mapping between enum_predictions_list and
#     enum_reference_list based on the enumerated word id. The function also
#     returns a enumerated list of unmatched words for predictions and reference.

#     :param enum_predictions_list: enumerated predictions list
#     :param enum_reference_list: enumerated reference list
#     :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
#     :return: enumerated matched tuples, enumerated unmatched predictions tuples,
#              enumerated unmatched reference tuples
#     """
#     stemmed_enum_predictions_list = [
#         (word_pair[0], stemmer.stem(word_pair[1])) for word_pair in enum_predictions_list
#     ]

#     stemmed_enum_reference_list = [
#         (word_pair[0], stemmer.stem(word_pair[1])) for word_pair in enum_reference_list
#     ]

#     return _match_enums(stemmed_enum_predictions_list, stemmed_enum_reference_list)


# def stem_match(
#     predictions: Iterable[str],
#     reference: Iterable[str],
#     stemmer: StemmerI = PorterStemmer(),
# ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
#     """
#     Stems each word and matches them in predictions and reference
#     and returns a word mapping between predictions and reference

#     :param predictions: pre-tokenized predictions
#     :param reference: pre-tokenized reference
#     :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
#     :return: enumerated matched tuples, enumerated unmatched predictions tuples,
#              enumerated unmatched reference tuples
#     """
#     enum_predictions_list, enum_reference_list = _generate_enums(predictions, reference)
#     return _enum_stem_match(enum_predictions_list, enum_reference_list, stemmer=stemmer)




# def _enum_wordnetsyn_match(
#     enum_predictions_list: List[Tuple[int, str]],
#     enum_reference_list: List[Tuple[int, str]],
#     wordnet: WordNetCorpusReader = wordnet,
# ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
#     """
#     Matches each word in reference to a word in predictions
#     if any synonym of a predictions word is the exact match
#     to the reference word.

#     :param enum_predictions_list: enumerated predictions list
#     :param enum_reference_list: enumerated reference list
#     :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
#     """
#     word_match = []
#     for i in range(len(enum_predictions_list))[::-1]:
#         predictions_syns = set(
#             chain.from_iterable(
#                 (
#                     lemma.name()
#                     for lemma in synset.lemmas()
#                     if lemma.name().find("_") < 0
#                 )
#                 for synset in wordnet.synsets(enum_predictions_list[i][1])
#             )
#         ).union({enum_predictions_list[i][1]})
#         for j in range(len(enum_reference_list))[::-1]:
#             if enum_reference_list[j][1] in predictions_syns:
#                 word_match.append(
#                     (enum_predictions_list[i][0], enum_reference_list[j][0])
#                 )
#                 enum_predictions_list.pop(i)
#                 enum_reference_list.pop(j)
#                 break
#     return word_match, enum_predictions_list, enum_reference_list

# def wordnetsyn_match(
#     predictions: Iterable[str],
#     reference: Iterable[str],
#     wordnet: WordNetCorpusReader = wordnet,
# ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
#     """
#     Matches each word in reference to a word in predictions if any synonym
#     of a predictions word is the exact match to the reference word.

#     :param predictions: pre-tokenized predictions
#     :param reference: pre-tokenized reference
#     :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
#     :return: list of mapped tuples
#     """
#     enum_predictions_list, enum_reference_list = _generate_enums(predictions, reference)
#     return _enum_wordnetsyn_match(
#         enum_predictions_list, enum_reference_list, wordnet=wordnet
#     )




# def _enum_align_words(
#     enum_predictions_list: List[Tuple[int, str]],
#     enum_reference_list: List[Tuple[int, str]],
#     stemmer: StemmerI = PorterStemmer(),
#     wordnet: WordNetCorpusReader = wordnet,
# ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
#     """
#     Aligns/matches words in the predictions to reference by sequentially
#     applying exact match, stemmed match and wordnet based synonym match.
#     in case there are multiple matches the match which has the least number
#     of crossing is chosen. Takes enumerated list as input instead of
#     string input

#     :param enum_predictions_list: enumerated predictions list
#     :param enum_reference_list: enumerated reference list
#     :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
#     :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
#     :return: sorted list of matched tuples, unmatched predictions list,
#              unmatched reference list
#     """
#     exact_matches, enum_predictions_list, enum_reference_list = _match_enums(
#         enum_predictions_list, enum_reference_list
#     )

#     stem_matches, enum_predictions_list, enum_reference_list = _enum_stem_match(
#         enum_predictions_list, enum_reference_list, stemmer=stemmer
#     )

#     wns_matches, enum_predictions_list, enum_reference_list = _enum_wordnetsyn_match(
#         enum_predictions_list, enum_reference_list, wordnet=wordnet
#     )

#     return (
#         sorted(
#             exact_matches + stem_matches + wns_matches, key=lambda wordpair: wordpair[0]
#         ),
#         enum_predictions_list,
#         enum_reference_list,
#     )


# def align_words(
#     predictions: Iterable[str],
#     reference: Iterable[str],
#     stemmer: StemmerI = PorterStemmer(),
#     wordnet: WordNetCorpusReader = wordnet,
# ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
#     """
#     Aligns/matches words in the predictions to reference by sequentially
#     applying exact match, stemmed match and wordnet based synonym match.
#     In case there are multiple matches the match which has the least number
#     of crossing is chosen.

#     :param predictions: pre-tokenized predictions
#     :param reference: pre-tokenized reference
#     :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
#     :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
#     :return: sorted list of matched tuples, unmatched predictions list, unmatched reference list
#     """
#     enum_predictions_list, enum_reference_list = _generate_enums(predictions, reference)
#     return _enum_align_words(
#         enum_predictions_list, enum_reference_list, stemmer=stemmer, wordnet=wordnet
#     )




# def _count_chunks(matches: List[Tuple[int, int]]) -> int:
#     """
#     Counts the fewest possible number of chunks such that matched unigrams
#     of each chunk are adjacent to each other. This is used to calculate the
#     fragmentation part of the metric.

#     :param matches: list containing a mapping of matched words (output of align_words)
#     :return: Number of chunks a sentence is divided into post alignment
#     """
#     i = 0
#     chunks = 1
#     while i < len(matches) - 1:
#         if (matches[i + 1][0] == matches[i][0] + 1) and (
#             matches[i + 1][1] == matches[i][1] + 1
#         ):
#             i += 1
#             continue
#         i += 1
#         chunks += 1
#     return chunks


# def single_meteor_score(
#     reference: Iterable[str],
#     predictions: Iterable[str],
#     preprocess: Callable[[str], str] = str.lower,
#     stemmer: StemmerI = PorterStemmer(),
#     wordnet: WordNetCorpusReader = wordnet,
#     alpha: float = 0.9,
#     beta: float = 3.0,
#     gamma: float = 0.5,
# ) -> float:
#     """
#     Calculates METEOR score for single predictions and reference as per
#     "Meteor: An Automatic Metric for MT Evaluation with HighLevels of
#     Correlation with Human Judgments" by Alon Lavie and Abhaya Agarwal,
#     in Proceedings of ACL.
#     https://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf


#     >>> predictions1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 'the', 'party']

#     >>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']


#     >>> round(single_meteor_score(reference1, predictions1),4)
#     0.6944

#         If there is no words match during the alignment the method returns the
#         score as 0. We can safely  return a zero instead of raising a
#         division by zero error as no match usually implies a bad translation.

#     >>> round(single_meteor_score(['this', 'is', 'a', 'cat'], ['non', 'matching', 'predictions']),4)
#     0.0

#     :param reference: pre-tokenized reference
#     :param predictions: pre-tokenized predictions
#     :param preprocess: preprocessing function (default str.lower)
#     :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
#     :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
#     :param alpha: parameter for controlling relative weights of precision and recall.
#     :param beta: parameter for controlling shape of penalty as a
#                  function of as a function of fragmentation.
#     :param gamma: relative weight assigned to fragmentation penalty.
#     :return: The sentence-level METEOR score.
#     """
#     enum_predictions, enum_reference = _generate_enums(
#         predictions, reference, preprocess=preprocess
#     )
#     translation_length = len(enum_predictions)
#     reference_length = len(enum_reference)
#     matches, _, _ = _enum_align_words(
#         enum_predictions, enum_reference, stemmer=stemmer, wordnet=wordnet
#     )
#     matches_count = len(matches)
#     try:
#         precision = float(matches_count) / translation_length
#         recall = float(matches_count) / reference_length
#         fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
#         chunk_count = float(_count_chunks(matches))
#         frag_frac = chunk_count / matches_count
#     except ZeroDivisionError:
#         return 0.0
#     penalty = gamma * frag_frac**beta
#     return {
#         "meteor":(1 - penalty) * fmean,
#         "penalty":penalty,
#         "precision":precision,
#         "recall":recall,
#         "alpha":alpha,
#         "beta":beta,
#         "gamma":gamma,
#     }


# def meteor_score(
#     references: Iterable[Iterable[str]],
#     predictions: Iterable[str],
#     preprocess: Callable[[str], str] = str.lower,
#     stemmer: StemmerI = PorterStemmer(),
#     wordnet: WordNetCorpusReader = wordnet,
#     alpha: float = 0.9,
#     beta: float = 3.0,
#     gamma: float = 0.5,
# ) -> float:
#     """
#     Calculates METEOR score for predictions with multiple references as
#     described in "Meteor: An Automatic Metric for MT Evaluation with
#     HighLevels of Correlation with Human Judgments" by Alon Lavie and
#     Abhaya Agarwal, in Proceedings of ACL.
#     https://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf


#     In case of multiple references the best score is chosen. This method
#     iterates over single_meteor_score and picks the best pair among all
#     the references for a given predictions

#     >>> predictions1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 'the', 'party']
#     >>> predictions2 = ['It', 'is', 'to', 'insure', 'the', 'troops', 'forever', 'hearing', 'the', 'activity', 'guidebook', 'that', 'party', 'direct']

#     >>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']
#     >>> reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which', 'guarantees', 'the', 'military', 'forces', 'always', 'being', 'under', 'the', 'command', 'of', 'the', 'Party']
#     >>> reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the', 'army', 'always', 'to', 'heed', 'the', 'directions', 'of', 'the', 'party']

#     >>> round(meteor_score([reference1, reference2, reference3], predictions1),4)
#     0.6944

#         If there is no words match during the alignment the method returns the
#         score as 0. We can safely  return a zero instead of raising a
#         division by zero error as no match usually implies a bad translation.

#     >>> round(meteor_score([['this', 'is', 'a', 'cat']], ['non', 'matching', 'predictions']),4)
#     0.0

#     :param references: pre-tokenized reference sentences
#     :param predictions: a pre-tokenized predictions sentence
#     :param preprocess: preprocessing function (default str.lower)
#     :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
#     :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
#     :param alpha: parameter for controlling relative weights of precision and recall.
#     :param beta: parameter for controlling shape of penalty as a function
#                  of as a function of fragmentation.
#     :param gamma: relative weight assigned to fragmentation penalty.
#     :return: The sentence-level METEOR score.
#     """
#     return max(
#         single_meteor_score(
#             reference,
#             predictions,
#             preprocess=preprocess,
#             stemmer=stemmer,
#             wordnet=wordnet,
#             alpha=alpha,
#             beta=beta,
#             gamma=gamma,
#         )
#         for reference in references
#     )


# def compute_meteor(reference_corpus, translation_corpus, max_order=4, smooth=False, tokenizer=default_tokenize_func('en')):
#     results = {
#         "meteor":0,
#         "penalty":0,
#         "precision":0,
#         "recall":0,
#         "alpha":0,
#         "beta":0,
#         "gamma":0,
#     }
#     count = 0
#     for (references, translation) in zip(reference_corpus, translation_corpus):
#         if not isinstance(references,list):
#            new_references = [references]
#         else:
#            new_references = references
#         if isinstance(translation,list):
#            new_translation = translation[0]
#         else :
#            new_translation = translation
#         result = meteor_score([tokenizer(r) for r in new_references],tokenizer(new_translation))
#         for key in result:
#             results[key]  = results[key] + result[key]
#         count = count + 1
#     for key in result:
#         results[key]  = results[key] / count
#     fmean = (result["precision"] * result["recall"]) / (result["alpha"] * result["precision"] + (1 - result["alpha"]) * result["recall"])
#     penalty = (result["penalty"] / count)
#     meteor = (1 - penalty) * fmean
#     return {'meteor_macro': results["meteor"], "meteor_micro":meteor}


# Natural Language Toolkit: Machine Translation
#
# Copyright (C) 2001-2024 NLTK Project
# Author: Uday Krishna <udaykrishna5@gmail.com>
# Contributor: Tom Aarsen
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT


from itertools import chain, product
from typing import Callable, Iterable, List, Tuple

from nltk.corpus import WordNetCorpusReader, wordnet
from nltk.stem.api import StemmerI
from nltk.stem.porter import PorterStemmer
from .tokenizer import default_tokenize_func

def _generate_enums(
    predictions: Iterable[str],
    reference: Iterable[str],
    preprocess: Callable[[str], str] = str.lower,
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Takes in pre-tokenized inputs for predictions and reference and returns
    enumerated word lists for each of them

    :param predictions: pre-tokenized predictions
    :param reference: pre-tokenized reference
    :preprocess: preprocessing method (default str.lower)
    :return: enumerated words list
    """
    if isinstance(predictions, str):
        raise TypeError(
            f'"predictions" expects pre-tokenized predictions (Iterable[str]): {predictions}'
        )

    if isinstance(reference, str):
        raise TypeError(
            f'"reference" expects pre-tokenized reference (Iterable[str]): {reference}'
        )

    enum_predictions_list = list(enumerate(map(preprocess, predictions)))
    enum_reference_list = list(enumerate(map(preprocess, reference)))
    return enum_predictions_list, enum_reference_list


def exact_match(
    predictions: Iterable[str], reference: Iterable[str]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    matches exact words in predictions and reference
    and returns a word mapping based on the enumerated
    word id between predictions and reference

    :param predictions: pre-tokenized predictions
    :param reference: pre-tokenized reference
    :return: enumerated matched tuples, enumerated unmatched predictions tuples,
             enumerated unmatched reference tuples
    """
    enum_predictions_list, enum_reference_list = _generate_enums(predictions, reference)
    return _match_enums(enum_predictions_list, enum_reference_list)




def _match_enums(
    enum_predictions_list: List[Tuple[int, str]],
    enum_reference_list: List[Tuple[int, str]],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    matches exact words in predictions and reference and returns
    a word mapping between enum_predictions_list and enum_reference_list
    based on the enumerated word id.

    :param enum_predictions_list: enumerated predictions list
    :param enum_reference_list: enumerated reference list
    :return: enumerated matched tuples, enumerated unmatched predictions tuples,
             enumerated unmatched reference tuples
    """
    word_match = []
    for i in range(len(enum_predictions_list))[::-1]:
        for j in range(len(enum_reference_list))[::-1]:
            if enum_predictions_list[i][1] == enum_reference_list[j][1]:
                word_match.append(
                    (enum_predictions_list[i][0], enum_reference_list[j][0])
                )
                enum_predictions_list.pop(i)
                enum_reference_list.pop(j)
                break
    return word_match, enum_predictions_list, enum_reference_list


def _enum_stem_match(
    enum_predictions_list: List[Tuple[int, str]],
    enum_reference_list: List[Tuple[int, str]],
    stemmer: StemmerI = PorterStemmer(),
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Stems each word and matches them in predictions and reference
    and returns a word mapping between enum_predictions_list and
    enum_reference_list based on the enumerated word id. The function also
    returns a enumerated list of unmatched words for predictions and reference.

    :param enum_predictions_list: enumerated predictions list
    :param enum_reference_list: enumerated reference list
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :return: enumerated matched tuples, enumerated unmatched predictions tuples,
             enumerated unmatched reference tuples
    """
    stemmed_enum_predictions_list = [
        (word_pair[0], stemmer.stem(word_pair[1])) for word_pair in enum_predictions_list
    ]

    stemmed_enum_reference_list = [
        (word_pair[0], stemmer.stem(word_pair[1])) for word_pair in enum_reference_list
    ]

    return _match_enums(stemmed_enum_predictions_list, stemmed_enum_reference_list)


def stem_match(
    predictions: Iterable[str],
    reference: Iterable[str],
    stemmer: StemmerI = PorterStemmer(),
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Stems each word and matches them in predictions and reference
    and returns a word mapping between predictions and reference

    :param predictions: pre-tokenized predictions
    :param reference: pre-tokenized reference
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :return: enumerated matched tuples, enumerated unmatched predictions tuples,
             enumerated unmatched reference tuples
    """
    enum_predictions_list, enum_reference_list = _generate_enums(predictions, reference)
    return _enum_stem_match(enum_predictions_list, enum_reference_list, stemmer=stemmer)




def _enum_wordnetsyn_match(
    enum_predictions_list: List[Tuple[int, str]],
    enum_reference_list: List[Tuple[int, str]],
    wordnet: WordNetCorpusReader = wordnet,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Matches each word in reference to a word in predictions
    if any synonym of a predictions word is the exact match
    to the reference word.

    :param enum_predictions_list: enumerated predictions list
    :param enum_reference_list: enumerated reference list
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    """
    word_match = []
    for i in range(len(enum_predictions_list))[::-1]:
        predictions_syns = set(
            chain.from_iterable(
                (
                    lemma.name()
                    for lemma in synset.lemmas()
                    if lemma.name().find("_") < 0
                )
                for synset in wordnet.synsets(enum_predictions_list[i][1])
            )
        ).union({enum_predictions_list[i][1]})
        for j in range(len(enum_reference_list))[::-1]:
            if enum_reference_list[j][1] in predictions_syns:
                word_match.append(
                    (enum_predictions_list[i][0], enum_reference_list[j][0])
                )
                enum_predictions_list.pop(i)
                enum_reference_list.pop(j)
                break
    return word_match, enum_predictions_list, enum_reference_list

def wordnetsyn_match(
    predictions: Iterable[str],
    reference: Iterable[str],
    wordnet: WordNetCorpusReader = wordnet,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Matches each word in reference to a word in predictions if any synonym
    of a predictions word is the exact match to the reference word.

    :param predictions: pre-tokenized predictions
    :param reference: pre-tokenized reference
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :return: list of mapped tuples
    """
    enum_predictions_list, enum_reference_list = _generate_enums(predictions, reference)
    return _enum_wordnetsyn_match(
        enum_predictions_list, enum_reference_list, wordnet=wordnet
    )




def _enum_align_words(
    enum_predictions_list: List[Tuple[int, str]],
    enum_reference_list: List[Tuple[int, str]],
    stemmer: StemmerI = PorterStemmer(),
    wordnet: WordNetCorpusReader = wordnet,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Aligns/matches words in the predictions to reference by sequentially
    applying exact match, stemmed match and wordnet based synonym match.
    in case there are multiple matches the match which has the least number
    of crossing is chosen. Takes enumerated list as input instead of
    string input

    :param enum_predictions_list: enumerated predictions list
    :param enum_reference_list: enumerated reference list
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :return: sorted list of matched tuples, unmatched predictions list,
             unmatched reference list
    """
    exact_matches, enum_predictions_list, enum_reference_list = _match_enums(
        enum_predictions_list, enum_reference_list
    )

    stem_matches, enum_predictions_list, enum_reference_list = _enum_stem_match(
        enum_predictions_list, enum_reference_list, stemmer=stemmer
    )

    wns_matches, enum_predictions_list, enum_reference_list = _enum_wordnetsyn_match(
        enum_predictions_list, enum_reference_list, wordnet=wordnet
    )

    return (
        sorted(
            exact_matches + stem_matches + wns_matches, key=lambda wordpair: wordpair[0]
        ),
        enum_predictions_list,
        enum_reference_list,
    )


def align_words(
    predictions: Iterable[str],
    reference: Iterable[str],
    stemmer: StemmerI = PorterStemmer(),
    wordnet: WordNetCorpusReader = wordnet,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Aligns/matches words in the predictions to reference by sequentially
    applying exact match, stemmed match and wordnet based synonym match.
    In case there are multiple matches the match which has the least number
    of crossing is chosen.

    :param predictions: pre-tokenized predictions
    :param reference: pre-tokenized reference
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :return: sorted list of matched tuples, unmatched predictions list, unmatched reference list
    """
    enum_predictions_list, enum_reference_list = _generate_enums(predictions, reference)
    return _enum_align_words(
        enum_predictions_list, enum_reference_list, stemmer=stemmer, wordnet=wordnet
    )




def _count_chunks(matches: List[Tuple[int, int]]) -> int:
    """
    Counts the fewest possible number of chunks such that matched unigrams
    of each chunk are adjacent to each other. This is used to calculate the
    fragmentation part of the metric.

    :param matches: list containing a mapping of matched words (output of align_words)
    :return: Number of chunks a sentence is divided into post alignment
    """
    i = 0
    chunks = 1
    while i < len(matches) - 1:
        if (matches[i + 1][0] == matches[i][0] + 1) and (
            matches[i + 1][1] == matches[i][1] + 1
        ):
            i += 1
            continue
        i += 1
        chunks += 1
    return chunks


def single_meteor_score(
    reference: Iterable[str],
    predictions: Iterable[str],
    preprocess: Callable[[str], str] = str.lower,
    stemmer: StemmerI = PorterStemmer(),
    wordnet: WordNetCorpusReader = wordnet,
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
) -> float:
    """
    Calculates METEOR score for single predictions and reference as per
    "Meteor: An Automatic Metric for MT Evaluation with HighLevels of
    Correlation with Human Judgments" by Alon Lavie and Abhaya Agarwal,
    in Proceedings of ACL.
    https://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf


    >>> predictions1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 'the', 'party']

    >>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']


    >>> round(single_meteor_score(reference1, predictions1),4)
    0.6944

        If there is no words match during the alignment the method returns the
        score as 0. We can safely  return a zero instead of raising a
        division by zero error as no match usually implies a bad translation.

    >>> round(single_meteor_score(['this', 'is', 'a', 'cat'], ['non', 'matching', 'predictions']),4)
    0.0

    :param reference: pre-tokenized reference
    :param predictions: pre-tokenized predictions
    :param preprocess: preprocessing function (default str.lower)
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :param alpha: parameter for controlling relative weights of precision and recall.
    :param beta: parameter for controlling shape of penalty as a
                 function of as a function of fragmentation.
    :param gamma: relative weight assigned to fragmentation penalty.
    :return: The sentence-level METEOR score.
    """
    enum_predictions, enum_reference = _generate_enums(
        predictions, reference, preprocess=preprocess
    )
    translation_length = len(enum_predictions)
    reference_length = len(enum_reference)
    matches, _, _ = _enum_align_words(
        enum_predictions, enum_reference, stemmer=stemmer, wordnet=wordnet
    )
    matches_count = len(matches)
    try:
        precision = float(matches_count) / translation_length
        recall = float(matches_count) / reference_length
        fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
        chunk_count = float(_count_chunks(matches))
        frag_frac = chunk_count / matches_count
    except ZeroDivisionError:
        return 0.0
    penalty = gamma * frag_frac**beta
    return (1 - penalty) * fmean


def meteor_score(
    references: Iterable[Iterable[str]],
    predictions: Iterable[str],
    preprocess: Callable[[str], str] = str.lower,
    stemmer: StemmerI = PorterStemmer(),
    wordnet: WordNetCorpusReader = wordnet,
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
) -> float:
    """
    Calculates METEOR score for predictions with multiple references as
    described in "Meteor: An Automatic Metric for MT Evaluation with
    HighLevels of Correlation with Human Judgments" by Alon Lavie and
    Abhaya Agarwal, in Proceedings of ACL.
    https://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf


    In case of multiple references the best score is chosen. This method
    iterates over single_meteor_score and picks the best pair among all
    the references for a given predictions

    >>> predictions1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 'the', 'party']
    >>> predictions2 = ['It', 'is', 'to', 'insure', 'the', 'troops', 'forever', 'hearing', 'the', 'activity', 'guidebook', 'that', 'party', 'direct']

    >>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']
    >>> reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which', 'guarantees', 'the', 'military', 'forces', 'always', 'being', 'under', 'the', 'command', 'of', 'the', 'Party']
    >>> reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the', 'army', 'always', 'to', 'heed', 'the', 'directions', 'of', 'the', 'party']

    >>> round(meteor_score([reference1, reference2, reference3], predictions1),4)
    0.6944

        If there is no words match during the alignment the method returns the
        score as 0. We can safely  return a zero instead of raising a
        division by zero error as no match usually implies a bad translation.

    >>> round(meteor_score([['this', 'is', 'a', 'cat']], ['non', 'matching', 'predictions']),4)
    0.0

    :param references: pre-tokenized reference sentences
    :param predictions: a pre-tokenized predictions sentence
    :param preprocess: preprocessing function (default str.lower)
    :param stemmer: nltk.stem.api.StemmerI object (default PorterStemmer())
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :param alpha: parameter for controlling relative weights of precision and recall.
    :param beta: parameter for controlling shape of penalty as a function
                 of as a function of fragmentation.
    :param gamma: relative weight assigned to fragmentation penalty.
    :return: The sentence-level METEOR score.
    """
    return max(
        single_meteor_score(
            reference,
            predictions,
            preprocess=preprocess,
            stemmer=stemmer,
            wordnet=wordnet,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        for reference in references
    )


def compute_meteor(reference_corpus, translation_corpus, max_order=4, smooth=False, tokenizer=default_tokenize_func('en')):
    scores = 0
    for (references, translation) in zip(reference_corpus, translation_corpus):
        if not isinstance(references,list):
           new_references = [references]
        else:
           new_references = references
        if isinstance(translation,list):
           new_translation = translation[0]
        else :
           new_translation = translation
        score = meteor_score([tokenizer(r) for r in new_references],tokenizer(new_translation))
        scores += score
    return {'meteor': scores/len(translation_corpus)}