import csv
from collections import defaultdict, Counter
import numpy as np
from flask import Flask, request, jsonify
import threading
import pandas as pd
import math
from pathlib import Path
import multiprocessing


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
        idf = math.log(N / wdf, 10)
        for doc_id, freq in pls:
            sim_dict[doc_id] += freq * idf * wqtf
    Qnorm = math.sqrt(sum([tf ** 2 for tf in query_dict.values()]))
    for doc_id in sim_dict.keys():
        sim_dict[doc_id] = sim_dict[doc_id] * (1 / Qnorm) * index.nf[doc_id]

    return sim_dict


def boolean_similarity(query_to_search, index):
    sim_dict = defaultdict(float)
    for w, pls in index.posting_lists_iter_by_query(query_to_search):
        for doc_id, freq in pls:
            sim_dict[doc_id] += 1
    return sim_dict


def merge_results(dict_scores_weight):
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
    merge_dict = {}
    for d, weight in dict_scores_weight:
        for doc_id, score in d.items():
            merge_dict[doc_id] = merge_dict.get(doc_id, 0.0) + score * weight
    return merge_dict


def BM25(query_to_search, index, b=0.75, k1=1.5, k3=1.5, base_log=10):
    sim_dict = {}
    query_dict = Counter(query_to_search)
    N = len(index.DL)
    for w, pls in index.posting_lists_iter_by_query(query_to_search):
        wdf = index.df[w]
        wqtf = query_dict[w]
        for doc_id, freq in pls:
            numerator = (k1 + 1) * freq * math.log((N + 1) / wdf, int(base_log)) * (k3 + 1) * wqtf
            denominator = freq + k1 * (1 - b + b * index.DL[doc_id] / index.AVGDL) * (k3 + wqtf)
            sim_dict[doc_id] = sim_dict.get(doc_id, 0.0) + (numerator / denominator)
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

    return [doc_id for doc_id, score in
            sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[
            :N]]


def result_doc_to_title(arr, titles_dict):
    result = []
    for doc_id in arr:
        result.append((doc_id, titles_dict.get(doc_id, "Not found doc title")))
    return result


def get_page_views(pages, page_view_dict):
    return [(doc_id, page_view_dict.get(doc_id, 0.0)) for doc_id in pages]


def get_page_rank(pages, data):
    return [(page, data.get(page, 0.0)) for page in pages]


def normalize_dict(scores_dict):
    max_elem = max(scores_dict.values())
    for key, value in scores_dict.items():
        scores_dict[key] = value / max_elem
    return scores_dict


def merge_results_with_PW(dict_scores_weight, page_view_dict):
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
    merge_dict = {}
    for d, weight in dict_scores_weight:
        for doc_id, score in d.items():
            merge_dict[doc_id] = merge_dict.get(doc_id, 0.0) + score * weight * math.log(
                page_view_dict.get(doc_id, 0) + 2, 2)
    return merge_dict


# def BM25_w_docs(query_to_search, index_body, index_body_docs, docs2search, b=0.75, k1=1.5, k3=1.5, base_log=10):
#     sim_dict = {}
#     query_dict = Counter(query_to_search)
#     N = len(index_body.DL)
#     AVG = index_body.AVGDL
#     for doc_id, pls in index_body_docs.posting_lists_iter_by_doc(docs2search):
#         DL = index_body.DL[doc_id]
#         for w, freq in pls:
#             if w in query_to_search:
#                 wdf = index_body.df[w]
#                 wqtf = query_dict[w]
#                 numerator = (k1 + 1) * freq * math.log((N + 1) / wdf, int(base_log)) * (k3 + 1) * wqtf
#                 denominator = freq + k1 * (1 - b + b * DL / AVG) * (k3 + wqtf)
#                 sim_dict[doc_id] = sim_dict.get(doc_id, 0.0) + (numerator / denominator)
#     return sim_dict

def boolean_n_BM25(query_to_search, index_body, index_title, page_view_dict, page_rank_dict, expanded_query=None,
                   top_n2merge=300, b=0.75, k1=1.5, k3=1.5, base_log=10, body_weight=0.5, title_weight=0.5):
    sim_dict = {}
    sim_dict_body, pl_body = boolean_similarity_n_pl(query_to_search, index_body)
    sim_dict_title, pl_title = boolean_similarity_n_pl(query_to_search, index_title)
    top_n_docs_by_boolean = get_top_n(
        merge_results_with_PW_PR([(sim_dict_body, body_weight), (sim_dict_title, title_weight)], page_view_dict,
                                 page_rank_dict), top_n2merge)

    query_dict = Counter(query_to_search)
    N = len(index_body.DL)
    for w, pls in pl_body:
        wdf = index_body.df[w]
        wqtf = query_dict[w]
        for doc_id, freq in pls:
            if doc_id in top_n_docs_by_boolean:
                numerator = (k1 + 1) * freq * math.log((N + 1) / wdf, int(base_log)) * (k3 + 1) * wqtf
                denominator = freq + k1 * (
                        1 - b + b * index_body.DL.get(doc_id, index_body.AVGDL) / index_body.AVGDL) * (k3 + wqtf)
                sim_dict[doc_id] = sim_dict.get(doc_id, 0.0) + (numerator / denominator)
    for w, pls in pl_title:
        wdf = index_title.df[w]
        wqtf = query_dict[w]
        for doc_id, freq in pls:
            if doc_id in top_n_docs_by_boolean:
                numerator = (k1 + 1) * freq * math.log((N + 1) / wdf, int(base_log)) * (k3 + 1) * wqtf
                denominator = freq + k1 * (
                        1 - b + b * index_title.DL.get(doc_id, index_title.AVGDL) / index_title.AVGDL) * (k3 + wqtf)
                sim_dict[doc_id] = sim_dict.get(doc_id, 0.0) + (numerator / denominator)
    return sim_dict


def boolean_n_cosineSimilarity(query_to_search, index_body, index_title, page_view_dict, page_rank_dict,
                               top_n2merge=300, body_weight=0.5, title_weight=0.5):
    sim_dict = defaultdict(float)
    sim_dict_body, pl_body = boolean_similarity_n_pl(query_to_search, index_body)
    sim_dict_title, pl_title = boolean_similarity_n_pl(query_to_search, index_title)
    top_n_docs_by_boolean = get_top_n(
        merge_results_with_PW_PR([(sim_dict_body, body_weight), (sim_dict_title, title_weight)], page_view_dict,
                                 page_rank_dict), top_n2merge)
    query_dict = Counter(query_to_search)
    N = len(index_body.DL)
    for w, pls in pl_body:
        wdf = index_body.df[w]
        wqtf = query_dict[w]
        idf = math.log(N / wdf, 10)
        for doc_id, freq in pls:
            if doc_id in top_n_docs_by_boolean:
                sim_dict[doc_id] += freq * idf * wqtf
    Qnorm = math.sqrt(sum([tf ** 2 for tf in query_dict.values()]))
    for doc_id in sim_dict.keys():
        sim_dict[doc_id] = sim_dict[doc_id] * (1 / Qnorm) * index_body.nf[doc_id]

    return sim_dict


def boolean_similarity_n_pl(query_to_search, index):
    sim_dict = defaultdict(float)
    pl = []
    for w, pls in index.posting_lists_iter_by_query(query_to_search):
        for doc_id, freq in pls:
            sim_dict[doc_id] += 1
        pl.append((w, pls))
    return sim_dict, pl


def merge_results_with_PW_PR(dict_scores_weight, page_view_dict, page_rank_dict):
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
    merge_dict = {}
    for d, weight in dict_scores_weight:
        for doc_id, score in d.items():
            merge_dict[doc_id] = merge_dict.get(doc_id, 0.0) + score * weight * math.log(
                page_view_dict.get(doc_id, 0) + 2, 2) * math.log(page_rank_dict.get(doc_id, 0) + 2, 2)
    return merge_dict
