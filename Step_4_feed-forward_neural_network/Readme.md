

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

I have only placed the .py files in GitHub. The MSCI_Assignement_4.py file contains all the execution and the main.py is the file to be executed which has the driver of MSCI_Assignement_4.py. Kindly execute main.py. The main.py expects one argument to be passed from command line which is the path to the folder which has the split made in assignmnet 1. For eg:- "a1/data" and I have used os.path .join to combine the path with the file name. For w2v model I have hard coded the path like "a3/data/w2v.model". 
inference. py accepts two argument path to the text file and model


# STUDENT DETAILS:-
Name :- Amith Nandakumar
Student id:- 20859891
email:- a4nandak@uwaterloo.ca

