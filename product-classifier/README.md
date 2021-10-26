# Product Classifier

A word representation as a vector was used to create a category classifer using the direction of the vector, corresponding to the collection of similar topic\terms\concepts, and cosine similarity.

The average vector across the word in a category can be used to create a representation of a topic. Each vector, in each category, can be compared, pairwise, to the average vector representation of an incoming block of text. The averaging method and comparison between vectors can be adjusted, i.e., L2-Norm and cosine similarity.

This method requires a taxonomy of categories and subcategories. It is also possible to add additional vocabulary (mispelled or now words) by averaging the vectors of related terms.