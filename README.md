# Information Retrieval Project
This is a repository for final assignment of the Information Retrieval course at the [Ben-Gurion University](https://in.bgu.ac.il/), Israel.

## Assignment Description
In this project we built a retrieval engine to retrieve information from the English Wikipedia corpus [Wikipedia](https://www.wikipedia.org/).

## How we retrieve information :

## Indexing:
Building the index of the corpus, indexing the body to the title and anchoring the text of a document.For the purpose of construction, we used Google's cloud [Google Cloud](https://cloud.google.com/) and the Spark library [Spark](https://spark.apache.org/docs/latest/rdd-programming-guide.html) to calculate the index. You can see it in the following document [Inverted Index Maker](inverted_indexes_maker/inveted_index_makers.ipynb).

## Similarity Methods:
Similarity methods we implemented are:
1. The Boolean model [Boolean model](https://en.wikipedia.org/wiki/Boolean_model_of_information_retrieval#:~:text=The%20BIR%20is%20based%20on,documents%20contain%20the%20query%20terms.)
2.Cosine Similarity [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
3. BM25 [BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
4. TF-IDF [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
Our implementation can see [here](functions.py)

## Our best search
At this project we were asked to provide the best retrieval method from the options we learned in the course.
The method that maximized our results was to use the *Boolean model* on the body and the title to retrieve a number of about 200 documents and then to retrieve from them with *BM25* while giving priority to documents with high *pageview* and *pagerank*


## Mendatory search methods
You have 5 search options:
1. Search in the body of the document by cosine simularity
2. Search in the title of the document by the Boolean module
3. Search in anchor text by the Boolean module
4. Search pageview of a document
5. Search pagerank of a document


## How to start search
To activate the engine, you have to run the [search_frontend.py](search_frontend.py) file that initializes the application. Initially, when the application comes up, it loads the Indexes and initialize initialize variable then the server opens.
The server receives queries in http requests in the following protocols:
* http://server_domain/search?query=hello+world
* http://server_domain/search_body?query=hello+world
* http://server_domain/search_title?query=hello+world
* http://server_domain/search_anchor?query=hello+world
* http://server_domain/get_pagerank', json=[1,5,8]
* http://server_domain/get_pageview', json=[1,5,8]




Written by Yuval Gorodissky and Noam Azulay, students in the course.

