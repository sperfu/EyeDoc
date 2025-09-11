# @Author       : Duhongkai
# @Time         : 2024/1/23 11:49
# @Description  : 用来计算bleu
import nltk


# 数据输入样式[[1,2,3,4,5],[6,7,8,9]]
def bleu(references, candidates, n):
    data = list()
    for reference, candidate in zip(references, candidates):
        single_data = modified_precision([reference], candidate, n)
        data.append(single_data)
    return sum(data) / len(data)


def modified_precision(references, candidate, n):
    """
    In the modified n-gram precision, a reference word will be considered
    exhausted after a matching candidate word is identified, e.g.

        >>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
        ...               'ensures', 'that', 'the', 'military', 'will',
        ...               'forever', 'heed', 'Party', 'commands']
        >>> reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
        ...               'guarantees', 'the', 'military', 'forces', 'always',
        ...               'being', 'under', 'the', 'command', 'of', 'the',
        ...               'Party']
        >>> reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
        ...               'army', 'always', 'to', 'heed', 'the', 'directions',
        ...               'of', 'the', 'party']
        >>> candidate= 'of the'.split()
        >>> references = [reference1, reference2, reference3]
        >>> float(modified_precision(references, candidate, n=1))
        1.0
        >>> float(modified_precision(references, candidate, n=2))
        1.0

    :param references: A list of reference translations.
    :type references: list(list(str))
    :param hypothesis: A hypothesis translation.
    :type hypothesis: list(str)
    :param n: The ngram order.
    :type n: int
    :return: BLEU's modified precision for the nth order ngram.
    :rtype: Fraction
    """
    # Extracts all ngrams in hypothesis
    # Set an empty Counter if hypothesis is empty.
    counts = nltk.Counter(nltk.ngrams(candidate, n)) if len(candidate) >= n else nltk.Counter()
    # Extract a union of references' counts.
    # max_counts = reduce(or_, [Counter(ngrams(ref, n)) for ref in references])
    max_counts = {}
    for reference in references:
        reference_counts = (
            nltk.Counter(nltk.ngrams(reference, n)) if len(reference) >= n else nltk.Counter()
        )
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])

    # Assigns the intersection between hypothesis and references' counts.
    clipped_counts = {
        ngram: min(count, max_counts[ngram]) for ngram, count in counts.items()
    }

    numerator = sum(clipped_counts.values())
    # Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
    # Usually this happens when the ngram order is > len(reference).
    denominator = max(1, sum(counts.values()))

    return numerator / denominator