import pickle
import nltk
import numpy as np
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from inverted_index_gcp import InvertedIndex
import pandas as pd
import math
from functions import *
from datetime import datetime
import re
from itertools import chain
import gensim.downloader as api

nltk.download('stopwords')


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


def init():
    global body_index
    global title_index
    global anchor_index
    global doc_index
    global docid_title_dict
    global pageRank_dict
    global page_view_dict
    global all_stopwords
    global RE_WORD

    body_index = InvertedIndex.read_index("./body_index", "body_index")
    title_index = InvertedIndex.read_index("./title_index", "title_index")
    anchor_index = InvertedIndex.read_index("./anchor_index", "anchor_index")
    # doc_index = InvertedIndex.read_index("./body_docs_index", "body_docs_index")

    with open('docid_title_dict.pkl', 'rb') as handle:
        docid_title_dict = pickle.load(handle)
    with open('pageRank_dict.pickle', 'rb') as handle:
        pageRank_dict = pickle.load(handle)
    with open('pageviews_dict.pkl', 'rb') as handle:
        page_view_dict = pickle.load(handle)

    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]

    english_stopwords = frozenset(stopwords.words('english'))
    all_stopwords = english_stopwords.union(corpus_stopwords)
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')

    p_body_cosine_similarity_in = p_body_cosine_similarity
    p_body_BM25_in =p_body_BM25
    p_body_boolean_in = p_body_boolean
    # if '?' in query:
    #     p_body_cosine_similarity_in*=3
    #     p_body_BM25_in *= 3
    #     p_body_boolean_in *= 3
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    all_models = []

    query = [token.group() for token in RE_WORD.finditer(query.lower()) if token.group() not in all_stopwords]
    res = get_top_n(
        boolean_n_BM25(query, body_index, title_index, page_view_dict,pageRank_dict, top_n2merge=top_n2merge, b=b_body_BM25,
                       k1=k1_body_BM25, k3=k3_body_BM25, base_log=base_log_body_BM25,body_weight=p_body_BM25_in,title_weight=p_title_boolean))
    res = result_doc_to_title(res, docid_title_dict)

    # if p_title_cosine_similarity > 0:
    #     all_models.append((normalize_dict(cosine_similarity(query, title_index)), p_title_cosine_similarity))
    # if p_title_BM25 > 0:
    #     all_models.append((normalize_dict(
    #         BM25(query, title_index, b=b_title_BM25, k1=k1_title_BM25, k3=k3_title_BM25, base_log=base_log_title_BM25)),
    #                        p_title_BM25))
    # if p_title_boolean > 0:
    #     all_models.append((normalize_dict(boolean_similarity(query, title_index)), p_title_boolean))
    # if p_body_cosine_similarity_in > 0:
    #     all_models.append((normalize_dict(cosine_similarity(query, body_index)), p_body_cosine_similarity_in))
    # if p_body_BM25_in > 0:
    #     all_models.append(
    #         (normalize_dict(
    #             BM25(query, body_index, b=b_body_BM25, k1=k1_body_BM25, k3=k3_body_BM25, base_log=base_log_body_BM25)),
    #          p_body_BM25_in))
    # if p_body_boolean_in > 0:
    #     all_models.append((normalize_dict(boolean_similarity(query, body_index)), p_body_boolean_in))
    # res = get_top_n(merge_results(all_models))

    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
            SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
            staff-provided tokenizer from Assignment 3 (GCP part) to do the
            tokenization and remove stopwords.

            To issue a query navigate to a URL like:
             http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
            where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
            if you're using ngrok on Colab or your external IP on GCP.
        Returns:
        --------
            list of up to 100 search results, ordered from best to worst where each
            element is a tuple (wiki_id, title).
        '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = [token.group() for token in RE_WORD.finditer(query.lower()) if token.group() not in all_stopwords]
    cosine_similarity_dict = cosine_similarity(query, body_index)
    res = get_top_n(cosine_similarity_dict)
    res = result_doc_to_title(res, docid_title_dict)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = [token.group() for token in RE_WORD.finditer(query.lower()) if token.group() not in all_stopwords]
    boolean_similarity_dict = boolean_similarity(query, title_index)
    res = get_top_n(boolean_similarity_dict, len(boolean_similarity_dict))
    res = result_doc_to_title(res, docid_title_dict)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
            IN THE ANCHOR TEXT of articles, ordered in descending order of the
            NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
            DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
            3 (GCP part) to do the tokenization and remove stopwords. For example,
            a document with a anchor text that matches two distinct query words will
            be ranked before a document with anchor text that matches only one
            distinct query word, regardless of the number of times the term appeared
            in the anchor text (or query).

            Test this by navigating to the a URL like:
             http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
            where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
            if you're using ngrok on Colab or your external IP on GCP.
        Returns:
        --------
            list of ALL (not just top 100) search results, ordered from best to
            worst where each element is a tuple (wiki_id, title).
        '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = [token.group() for token in RE_WORD.finditer(query.lower()) if token.group() not in all_stopwords]
    boolean_similarity_dict = boolean_similarity(query, anchor_index)
    res = get_top_n(boolean_similarity_dict, len(boolean_similarity_dict))
    res = result_doc_to_title(res, docid_title_dict)
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

            Test this by issuing a POST request to a URL like:
              http://YOUR_SERVER_DOMAIN/get_pagerank
            with a json payload of the list of article ids. In python do:
              import requests
              requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
            As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
            if you're using ngrok on Colab or your external IP on GCP.
        Returns:
        --------
            list of floats:
              list of PageRank scores that correrspond to the provided article IDs.
        '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    wiki = []
    for id in wiki_ids:
        try:
            wiki.append(int(id))
        except:
            continue
    res = get_page_rank(wiki, pageRank_dict)
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
            had in August 2021.

            Test this by issuing a POST request to a URL like:
              http://YOUR_SERVER_DOMAIN/get_pageview
            with a json payload of the list of article ids. In python do:
              import requests
              requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
            As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
            if you're using ngrok on Colab or your external IP on GCP.
        Returns:
        --------
            list of ints:
              list of page view numbers from August 2021 that correrspond to the
              provided list article IDs.
        '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    wiki = []
    for id in wiki_ids:
        try:
            wiki.append(int(id))
        except:
            continue
    res = get_page_views(wiki, page_view_dict)
    # END SOLUTION
    return jsonify(res)


@app.route("/set_parameters", methods=['POST'])
def set_parameters():
    '''
        set the parameters of the search in the this order:
     p_title_cosine_similarity
     p_title_boolean
     p_title_BM25
     b_title_BM25
     k1_title_BM25
     k3_title_BM25
     base_log_title_BM25

     p_body_cosine_similarity
     p_body_boolean
     p_body_BM25
     b_body_BM25
     k1_body_BM25
     k3_body_BM25
     base_log_body_BM25
        '''
    all_param = request.get_json()
    if len(all_param) != 15:
        return jsonify([False])
    # BEGIN SOLUTION
    parameters = []
    for param in all_param:
        try:
            if (float(param) < 0):
                return jsonify([False])
            parameters.append(float(param))

        except:
            return jsonify([False])
    global p_title_cosine_similarity
    global p_title_boolean
    global p_title_BM25
    global b_title_BM25
    global k1_title_BM25
    global k3_title_BM25
    global base_log_title_BM25

    global p_body_cosine_similarity
    global p_body_boolean
    global p_body_BM25
    global b_body_BM25
    global k1_body_BM25
    global k3_body_BM25
    global base_log_body_BM25

    global top_n2merge

    p_title_cosine_similarity = parameters[0]
    p_title_boolean = parameters[1]
    p_title_BM25 = parameters[2]
    b_title_BM25 = parameters[3]
    k1_title_BM25 = parameters[4]
    k3_title_BM25 = parameters[5]
    base_log_title_BM25 = parameters[6]

    p_body_cosine_similarity = parameters[7]
    p_body_boolean = parameters[8]
    p_body_BM25 = parameters[9]
    b_body_BM25 = parameters[10]
    k1_body_BM25 = parameters[11]
    k3_body_BM25 = parameters[12]
    base_log_body_BM25 = parameters[13]

    top_n2merge = int(parameters[14])

    # END SOLUTION
    return jsonify([True])


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    init()
    app.run(host='0.0.0.0', port=8080, debug=True)
