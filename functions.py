import csv
from collections import defaultdict, Counter
import numpy as np
from flask import Flask, request, jsonify

import pandas as pd
import math
from pathlib import Path


def cosine_similarity(query_to_search, index):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """
    # YOUR CODE HERE
    sim_dict = defaultdict(float)
    query_dict = Counter(query_to_search)
    N = len(index.DL)
    for w, pls in index.posting_lists_iter_by_query(query_to_search):
        wdf = index.df[w]
        wqtf = query_dict[w]
        for doc_id, freq in pls:
            sim_dict[doc_id] += (freq / index.DL[doc_id]) * math.log(N / wdf, 10) * wqtf * index.nf[doc_id]
    QL = len(query_to_search)
    for doc_id in sim_dict.keys():
        sim_dict[doc_id] = sim_dict[doc_id] / QL

    return sim_dict


def boolean_similarity(query_to_search, index):
    sim_dict = defaultdict(float)
    for w, pls in index.posting_lists_iter_by_query(query_to_search):
        for doc_id, freq in pls:
            sim_dict[doc_id] += 1
    return sim_dict


def merge_results(title_scores, body_scores, title_weight=0.5, text_weight=0.5):
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body).

    Parameters:
    -----------
    title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)

    body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    dictionary of querires and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id,score).
    """
    # YOUR CODE HERE
    merge_dict = defaultdict(float)
    for d, weight in [(title_scores, title_weight), (body_scores, text_weight)]:
        for doc_id, score in d.items():
            merge_dict[doc_id] += score * weight
    return merge_dict


def BM25(query_to_search, index, b=0.75, k1=1.5, k3=1.5, base_log=10):
    sim_dict = defaultdict(float)
    query_dict = Counter(query_to_search)
    N = len(index.DL)
    for w, pls in index.posting_lists_iter_by_query(query_to_search):
        wdf = index.df[w]
        wqtf = query_dict[w]
        for doc_id, freq in pls:
            numerator = (k1 + 1) * freq * math.log((N + 1) / wdf, base_log) * (k3 + 1) * wqtf
            denominator = freq + k1 * (1 - b + b * index.DL[doc_id] /index.AVGDL) * (k3 + wqtf)
            sim_dict[doc_id] += (numerator / denominator)
    return sim_dict


def get_top_n(sim_dict, N=100):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """

    return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[
           :N]


def result_doc_to_title(arr, titles_dict):
    result = []
    for doc_id, score in arr:
        result.append((doc_id, titles_dict[doc_id]))
    return result


def get_page_views(pages, page_view_dict):
    return [(doc_id, page_view_dict[doc_id]) for doc_id in pages]


def get_page_rank(pages, data):
    return [(page, data[page]) for page in pages if page in data]
