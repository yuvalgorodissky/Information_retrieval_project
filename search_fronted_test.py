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
    global word2vec_dict
    global body_index_stemming
    global title_index_stemming

    word2vec_dict = api.load('glove-wiki-gigaword-50')
    # original query

    body_index = InvertedIndex.read_index("./body_index", "body_index")
    title_index = InvertedIndex.read_index("./title_index", "title_index")
    anchor_index = InvertedIndex.read_index("./anchor_index", "anchor_index")
    # doc_index = InvertedIndex.read_index("./body_docs_index", "body_docs_index")
    body_index_stemming = InvertedIndex.read_index("./body_index_stemming", "body_index_stemming")
    title_index_stemming = InvertedIndex.read_index("./title_index_stemming", "title_index_stemming")

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

    # p_body_BM25_in = p_body_BM25

    # if '?' in query:
    #     p_body_BM25_in *= 3

    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    all_models = []

    query = [token.group() for token in RE_WORD.finditer(query.lower()) if token.group() not in all_stopwords]
    # find synonyms of the query
    # expanded_query = word2vec_dict.most_similar(positive=query, topn=5)
    # query = query + [w for w,score in expanded_query]

    # add synonyms to the query

    if p_title_cosine_similarity > 0:
        all_models.append((normalize_dict(cosine_similarity(query, title_index)), p_title_cosine_similarity))
    if p_title_BM25 > 0:
        all_models.append((normalize_dict(
            BM25(query, title_index, b=b_title_BM25, k1=k1_title_BM25, k3=k3_title_BM25, base_log=base_log_title_BM25)),
                           p_title_BM25))
    if p_title_boolean > 0:
        all_models.append((normalize_dict(boolean_similarity(query, title_index)), p_title_boolean))
    if p_body_cosine_similarity > 0:
        all_models.append((normalize_dict(cosine_similarity(query, body_index)), p_body_cosine_similarity))
    if p_body_BM25 > 0:
        all_models.append(
            (normalize_dict(
                BM25(query, body_index, b=b_body_BM25, k1=k1_body_BM25, k3=k3_body_BM25, base_log=base_log_body_BM25)),
             p_body_BM25))
    if p_body_boolean > 0:
        all_models.append((normalize_dict(boolean_similarity(query, body_index)), p_body_boolean))
    res = get_top_n(merge_results(all_models))
    res = result_doc_to_title(res, docid_title_dict)

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
            if float(param) < 0:
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
