# Learner-Query-Classification

Problem Statement
The objective of this project is to categorize monthly learner queries into predefined categories. The categories are as follows:
1. General
2. Content
3. Assessment
4. Platform Support
5. Certificates

This classification helps in better understanding the nature of queries learners have and allows for more efficient handling of these queries by the support team.

Approach
To solve the problem, I implemented several machine learning models to classify the learner queries. These models were trained using a dataset that contained learner queries, their corresponding categories, and responses provided by a chatbot. The following algorithms were utilized:

1. Support Vector Machine (SVM)
2. Gradient Descent
3. Random Forest Classifier
4. BERT (Bidirectional Encoder Representations from Transformers)
5. Naive Bayes
6. SpaCy-based Text Classification

Data
The data used for training the models includes:
- Learner Queries: Textual data containing questions or queries raised by learners.
- Question Category: The category each query belongs to, which serves as the label for training.
- Chatbot Responses: The responses provided by a chatbot for each query.

Models and Evaluation
Each model was evaluated based on its accuracy in categorizing the learner queries. The performance of the models was compared to identify the most effective algorithm for this task.

Conclusion
Through the evaluation of various machine learning models, the project aimed to determine the best approach for categorizing learner queries into predefined categories. Each model was analyzed based on its accuracy to understand which one provides the most reliable results.

Prerequisites
- Python 3.x
- Libraries: scikit-learn, TensorFlow/PyTorch (for BERT), SpaCy, pandas, numpy
  
Consideration:
Dataset provided by the organisation was not accurate and clear to optimise the model more but when tested on correct dummy data the model response was good.
