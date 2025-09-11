# @Author       : Duhongkai
# @Time         : 2024/1/23 11:48
# @Description  : 用来计算distinct

import nltk


def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(nltk.ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


# 数据输入样式[[1,2,3,4,5],[6,7,8,9,10]]
def distinct_n_corpus_level(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)
