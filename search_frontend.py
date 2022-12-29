import pickle
from collections import defaultdict, Counter
import numpy as np
from flask import Flask, request, jsonify
from inverted_index_gcp import InvertedIndex
import pandas as pd
import math
from functions import *
from datetime import datetime


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


def init():
    global body_index
    body_index = InvertedIndex.read_index("./body_index", "body_index")
    global title_index
    title_index = InvertedIndex.read_index("./title_index", "title_index")
    global anchor_index
    anchor_index = InvertedIndex.read_index("./anchor_index", "anchor_index")
    global docid_title_dict
    with open('docid_title_dict.pkl', 'rb') as handle:
        docid_title_dict = pickle.load(handle)
    global pageRank_dict
    with open('pageRank_dict.pickle', 'rb') as handle:
        pageRank_dict = pickle.load(handle)
    global page_view_dict
    with open('pageviews_dict.pkl', 'rb') as handle:
        page_view_dict = pickle.load(handle)



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
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = query.lower()
    query = query.split()
    cosine_similarity_dict = cosine_similarity(query, body_index)
    res = get_top_n(cosine_similarity_dict)
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
    query = query.lower()
    query = query.split()
    cosine_similarity_dict = cosine_similarity(query, body_index)
    res = get_top_n(cosine_similarity_dict)

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
    query = query.lower()
    query = query.split()
    boolean_similarity_dict = boolean_similarity(query, title_index)
    res = get_top_n(boolean_similarity_dict,len(boolean_similarity_dict))
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
    query = query.lower()
    query = query.split()
    boolean_similarity_dict = boolean_similarity(query, anchor_index)
    res = get_top_n(boolean_similarity_dict, len(boolean_similarity_dict))
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
    res = get_page_rank(wiki_ids, pageRank_dict)
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
    res = get_page_views(wiki_ids,page_view_dict)
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    init()
    app.run(host='0.0.0.0', port=8080, debug=True)
