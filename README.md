Genre identification deals with the Natural Language Processing problem to identify the thema of the content. This allows the user to eliminate the cumbersome task of reading the summary/abstract of the content and identify the Genre. 
The content provided for the task is a subset of English Fiction books belonging to Gutenberg Corpus. 
The corpus was a set of 996 books with each book belonging to one of the following Genre:-
* Literary - 794 instances
* Detective and Mystery - 111 instances
* Sea and Adventure - 36 instances
* Love and Romance - 18 instances
* Western Stories - 18 instances
* Ghost and Horror - 6 instances
* Humorous and Wit and Satire - 6 instances
* Christmaas Stories - 5 instances
* Allegories - 2 instances

The Dataset was highly imbalanced  as 79% of the data belonged to class- Literary.

Moreover, the number of words in an instance were different throughout the dataset.

The above mentioned issues posed a real challenge to our learning and several ideas were proposed and adopted to handle the problem.

Also, For Feature selection, our aim was to extract both Symantic and Syntactic information of an instance instead of just extracting lexical information.

For this purpose, we decided to handcraft 22 features based on our reading from various resources. The features extrracted are as follows:-

* Paragraph count
* Female Pronoun Count
* Male Pronoun Count
* Personal Pronoun Count
* Possesive Pronoun
* Preposition Count
* Coordinate Conjunction
* Comma Count
* period Count
* Colon Count
* Semi Colon Count
* Hyphen Count
* Interjection Count
* Punctuation and Subordinate Conjunction
* Sentence Length
* Ouotes Count
* Negative Sentiment
* Possitive Sentiment
* Neutral Sentiment
* Flesch Reading Score
* Number of Characters
* Type Token Ratio

These features, helped us to capture sentiment, thema and the writing style of the writer.

Also, to handle the problem of class imbalance and create more number of instances for the model to learn, each instance was subdivided into chunks and also Up-Sampling was performed.For evaluation of our performance, 20% of the data was kept seperately as a test set.

Several models were tried with algorithms like Naive-Bayes, Multi Layer Perceptron and Support Vector Machines.

The complete pipeline and results are documented in the Report.

