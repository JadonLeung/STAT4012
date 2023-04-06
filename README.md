# STAT4012

To do list:
1. Web scrapping from crypto slate (Completed)
2. Gather BTC data from exchange, maybe Binance
3. Cleaning data to ret and decision
4. Model training



STAT4012 Group 2 Project Outline

Fong Yuen Tung 1155158139, Ng Weng Sam 1155133815, Leung Ho Kwan 1155144270, 
Cheng Chu Fung 1155143199, So Fu Shing 1155159902
Objective:
The aim of this project is to develop a model that accurately predicts the direction of Bitcoin price movements through news sentiment analysis using natural language processing (NLP) techniques. Specifically, we will generate signals for Bitcoin trading, indicating whether to buy, sell, or wait, based on analyzing news headlines from Coindesk. In this analysis, the timescope of open positions will be five minutes.

Dataset:
The dataset for this project will consist of two parts. The first dataset is news headlines from Coindesk. We choose Coindesk because it is a popular and well-established media outlet that covers the cryptocurrency and blockchain space extensively. We will gather ~2200 headlines with their publishing time from coindesk in a 1-year period (2022/03/01 - 2023/02/28). The second dataset is the corresponding Bitcoin price movements in the 5 minutes following the release of each headline via Binance API as the same period with headlines. 

(CoinDesk webshttps://www.coindesk.com/tag/bitcoin/

Method:
The project will involve several steps. First, we will collect news headlines from coindesk and preprocess them to ensure consistency in formatting and remove any irrelevant information. Also, after gathering data from Binance, we will classify them into three classes decided by a classification threshold which will be decided by the statistics of data gathered.

The dataset will be split into training and testing sets, with the training set used to train the model and the testing set used to evaluate its performance. Next, we will use NLP techniques to analyze the sentiment of each headline.

We will use the sentiment analysis with Multi-layer Perceptron and Recurrent Neural Network results to generate buy, sell, or hold signals for Bitcoin trading, based on the expected price movements in the 5 minutes following each news release. We will evaluate the performance of our models using metrics such as accuracy, precision, and recall and compare them with other machine learning models like KNN and logistic regression.
