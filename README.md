# Sentiment analysis based on Polish COVID-19 tweets.

The aim of the project was to create a sentiment analyzer for polarization of society in Poland on the basis of COVID-19 tweets.

The project was created in several steps. Each of them was implemented in a separate script (the numbers refer to the order of execution):
* 0_tweets_download
* 1_labeling_tweets
* 2_data_preprocesing
* 3_tweets_ML
* 4_tweets_CNN
* 5_tweets_BERT

## Data
The first step was to obtain the data. About 500k polish tweets about COVID-19 have been collected. Tweets were collected using the `tweepy` API based on their ID. Tweet IDs were downloaded from the website https://zenodo.org/record/5090588#.YZDZdmDMKUn.

## Labeling
The next main step was the labeling of tweets due to their sentiment sentiment of the 3 classes *positive/neutral/negative* After collecting over 450k, the data preprocessing stage was started. Duplicates and stopwords have been removed. Performed lemmatization for unambiguous tokens. Clean text from URLs, twitter usernames, digits, punctuation marks, emoticons and other special characters. Collocation detection was made. After that, Word2Vec model was trained.<br>
The K-means algorithm was used to mark the sentiment of individual words.<br>
**NOTE: There is no guarantee that K-means will group tokens with the same sentiment.**<br>
However, we assume that after using word embeddings, tokens with a similar sentiment will not lie too far apart, therefore, due to the lack of a better method, we will use K-means. Please note that the assigned sentiment will not always be correct.<br>
The sentiment for tweets was determined on the basis of the average value of the sentiments of individual tokens in the tweet.

## Preprocessing
The next step was to prepare the data after appropriately tagging it in the previous step. The methods used are similar to those in notebook *1_labeling_tweets*, but with some differences (e.g. no lemmatization was used). The result of this step was the prepared data for modeling.

## Classic ML models
After cleaning text, some duplicates appeared. We removed them. Then, with RegexpTokenizer from nltk module, we made tokenization. In the next step, we converted text to vectors using CountVectorizer from sklearn.<br>
Then built 3 models:
* logistic regression
* naive bayes
* kNN
The last step was evaluation with ROC curves and confusion matrixes.

## CNN model
The CNN model was also built as an alternative to the classical methods. We used keras and tensorflow.
In the next step we created tokenizer with max_words = 5000 (max_words is the maximum words in vocabulary) and used pad_sequence for padding to length=200.<br>
After train-test split (the same seed like in previous script with classic ML) we created CNN model for sentiment classification. Then evaluated with ROC and confusion matrix.

## BERT model
We used also `ktrain` module for sentiment classification with BERT model.<br>
The process of preparing data for modeling using BERT was slightly different.<br>
Since training for all 455k tweets using BERT would take more than 12 hours on the GPU, and the limitations of Google Colab do not allow such a long process, we decided to trim the dataset so that the training lasted shorter and so that the model could be presented. The evaluation was performed using the ROC and confusion matrix.
