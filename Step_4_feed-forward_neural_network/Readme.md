# Step 4:-

Write a python script using keras to train a fully-connected feed-forward neural network classifier to classify documents in the Amazon corpus into positive and negative classes. Your network must consist of:
1. Input layer of the word2vec embeddings you prepared in Step 3.
2. One hidden layer. For the hidden layer, try the following activation functions: ReLU, sigmoid and tanh.
3. Final layer with softmax activation function.
4. Use cross-entropy as the loss function.
5. Add L2-norm regularization.
6. Add dropout. Try a few different dropout rates.

# ACCURACIES:-

| Activation functions | L2 and drop out        | Accuracy(Test Set) |
| ------------------   |:----------------------:| ------------------:|
| Relu                 | No                     | 81.60%             |
| Relu                 | yes(L2=0.2)			| 76.02%             |
| Relu                 | yes(dropout=0.2)		| 77.01%             |
| Relu                 | yes(L2=0.2,dropout=0.2)| 77.09%             |
| Relu                 | yes(L2=0.4,dropout=0.4)| 79.78%             |
| Tanh                 | No                     | 81.99%             |
| Tanh                 | yes(L2=0.2)			| 80.62%             |
| Tanh                 | yes(dropout=0.2)		| 80.73%             |
| Tanh                 | yes(L2=0.2,dropout=0.2)| 81.54%             |
| Tanh                 | yes(L2=0.4,dropout=0.4)| 80.87%             |
| Sigmoid              | No                     | 80.53%             |
| Sigmoid              | yes(L2=0.2)			| 79.23%             |
| Sigmoid              | yes(dropout=0.2)		| 78.92%             |
| Sigmoid              | yes(L2=0.2,dropout=0.2)| 79.89%             |
| Sigmoid              | yes(L2=0.4,dropout=0.4)| 79.41%             |


# BRIEF REPORT:-

From the accuarcies obtained with different activation function, it is visible that using Tanh as an activation function provided better result when compared to using sigmoid and Relu. ReLu is the most used activation function. The range of ReLu is from (0 to infinity). But, the issue is negative values become zero immediately which decreases the ability to map the negative values appropriately. Tanh is similar to sigmoid but better than sigmoid and the range is from (-1 to 1) where negative inputs will be mapped strongly negative and the zero inputs will be mapped near zero. Hence for two class classification, Tanh performs slightly better than relu even though this is true for a very deep architecture tanh suffers from vanishing gradient and hence not popularly used. 

It is also evident from the table that for all activation function without l2 and dropout performed slightly better. L2 regularization and dropouts makes the model more robust and avoids overfitting in a deep neural network architecture and hence using it will provide better performance on test dataset. However, in our architecture since we have used only one single hidden layer and the LSTM units being 128, without dropouts and L2 regularization performed better. Checked with only dropout and only L2 still the performance was poor indicating that for this shallow architecture and less hidden units dropouts and L2 regularization not suitable

# EXECUTION DETAILS:-

I have only placed the .py files in GitHub. The MSCI_Assignement_4.py file contains all the execution and the main.py is the file to be executed which has the driver of MSCI_Assignement_4.py.
`inference.py` script classifies a given sentence into a positive/negative class. It should accept the two command-line arguments described below.
i. arg1: Path to a .txt file, which contains some sentences compiled for evaluation. There will be one sentence per line.
ii. arg2: Type of classifier to use. Its value will be one of the following – relu, sigmoid, tanh. Example: relu indicates that the neural network with ReLU
activation should be selected for classifying sentences in the aforementioned .txt file.


