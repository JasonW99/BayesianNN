# Bayesian Neural Network
solve a neural network classification problem using Bayesian approach
## The Data Set
2 dimensional data comes from 3 groups. using it as training data, we propose models to classify points into appropriate groups. not good.
![alt text](/pic/data_set.png)
## Linear Model
we firstly propose a linear model (GLM in multinomial case). clearly, the result is not good.
![alt text](/pic/linear_result.png)
## NN Model 
we then propose non-linear model using neural network. and we solve the system in both Frequentist and Bayesian approach.
### Frequentist Approach
only gives the result of the decision
![alt text](/pic/NN_frequentist_result.png)
### Bayesian Approach
not only gives the result of the decision, but also the distribution of the decision. from the variability of the distribution, we can see the uncertainty of the decision.
![alt text](/pic/NN_bayesian_result2_regionpred.png)
![alt text](/pic/NN_bayesian_result3_uncertainty.png)

## Authors
* **Jason W**