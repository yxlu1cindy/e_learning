<h2> Course 1 </h2>
1. Why deep-learning?
  - perform better on large dataset
  - perform better when large NN
  - first layer(detecting edges)->following layers: (combine edges)

2. Activataion Function: 
  - Why non-linear:
    - if linear, then only consider the linear relationship between all features: if f and g re linear, then f(g(x)) is also linear
  -Sigmoid vs Relu(->Leaky Relu) vs tanh:
    - range : Sigmoid(0-1), tanh(-1,1)
    - Relu is computational effcient when calculating gradient descent; both gd of sigmoid and tanh when z is large is very small
    - Generallt, tanh  is better than sigmoid : whhen centering the data, the mean tend to close to 0, which mean(tanh) == 0 ); but when binary classification, sigmoid may be better
  
3. Loss Function(logistic):
  - \frac{1}{2} (\hat y - y)^2 --> not convex --> may be trapped in local minimum
  - use -(ylog(\hat y) +(1-y)log(1-\hat y))  --> convex

4. Loss Function vs Cost Function:
  - The loss function computes the error for a single training example; the cost function is the average of the loss functions of the entire training set.
  
5. Initialization:
  - w for hidden layer -- cannot just set np.zeros() --> after iteration, the w[i][1] & w[i][2] will always be symetric (*need to be proov*)
  - so need to initialize randomly, start with a small number (0.01) -> Why small value? w big -> z big -> large value in sigmoid/tanh function -> small grdient --> very slow when iterating
  - b can be set 0 when initialization
  - *He initialization* ( He et al., 2015) -> He initialization works well for networks with ReLU activations.
  - Xaivers -> Xaivers initialization works well for networks with tanh activations.

6. Construction:
  - a "small" L-layer deep NN <-> a shallower network withwxponentially more hidden units
  - ? Can this be another reason why deep neutral network works better? A simple regression/classification can be seen a one-layer NN, which requires more para to get the same performance than a deep NN (Personal Understanding)

7. hyperparameter vs parameter:
  - parameter : w, b
  - hyperparameter: learning rates & # of iteration in gradint descent, L, number of hidden units, choice of activation function
  - hyperparameter determins the value of parameters
*DMH*

<h2> Course 2 </h2>
1. Data split:   
  - traditionally: 70%/30%(train/test),60%/40%/40%(train,dev(validation),test)
  - big data: if you have 1,000,000 data, you may only need 10000 to dev set and 10000 to test set. so 98%/1%/1% is also acceptable
  - size of test set :big enough to give you high confidence in the overall performance of the system
  - data for dev and test must from same data sourse, which means same distribution. But training set can use another dataset, because we choose the algorithms based on the performance of the dev set
  - 
  - Function: training set - train model; dev set - compare different model and choose the best one; test set - make a unbias estimation on the performance
  
 
2. Bias and Variance (trade-off)   
  - Assumptions 1: 
    1. Based on humen error/bayes error = 0 ( which means human can do the classification with 100% correct, sometimes if the quality of the dataset is poor, human may even have the probability to make a mistake, then at this time bayes error != 0)
    2. training set and dev set have same distribution/from same dataset
    3. Anaylsis:
      - high variance -- overfitting
      - high bias -- underfitting
      - high variance:
        - a large difference between training set error & dev set error
        - training set error is small
      - high bias:
        - training set error is large
        - training set error & dev set error is similar
      - high bias and high variance:
        - training set error is large
        - dev set error is much greater than training set error
        

3. Basic recipe for ML projects:
  - high bias? (continue training until get a small training set error)
      - try a bigger network(more hidden units, more hidden layer)  -> always helps
      - run trains longer -> not always help, but never hurts
      - try some more advanced optimization algorithms
      - find a new NN architecture
  - high variance? (continue doing until get a small training set error and a small dev srror)
      - get more data
      - regularization
      - find a new NN architecture
      
4. Regularization:  
  - L2 Regularization
    - add a penalty in the ocst function J(w1,b1,w2,b2,...,wL,bL): J =\frac{1}{m}\sum(L(\hat{y},y)) + (\lambda \sum(||wi||^2_2)）/2m  -> L2 Regularization where ||wk||^2_2 = \sum{i}\sum{j}(w_ij^k)^2 (*_Frobenius norm_*)
      - Why only add ||w||? w is high dimension than b, so it may weights much variance than b
      - \lambda -> regularization parameter
      - L2 regularization is more often used
    - Whyl2  regularization reduces variance/prevent overfitting?
      - avoid the weight matrix W to become too large
      - if choose a large \lambda, then some of w will close to 0 in order to get minJ, then the NN can be seem as have a smaller hidden units in each layer, which increase bias. In the process, we can find a lambda that meet the "just right" condition
      - for sigmoid/tanh, when w is small, then WX is small, then it is nearly linear when z close to 0, which prevents overfitting
      - the J will monotoneic decreasing if adding penalty item, otherwise J may not continue decreasing after each iteration
  - Dropout Regularization:
    - How it works:
       - For each sample,Go through each of the layers and set some probability of elimination a node in NN (设定一些概率来对每层中的每个神经元进行保留/舍弃的操作 at random,每个sample需要舍弃的神经元是不同的！每次都要重新来） -> get a smaller NN
    - When predict:
      - KEEP ALL hidden units！！
    - Types:
      - Inverted Dropout:
    - Why it works:
      - can't rely on any one feature because each units have a probability ot be removed -> shrinking the weights -> similar effect as L2 Regularization
    - For a layer have mant units, we prefer choose a low keep rates to avoid overfitting
    - often used in compute vision area
    - cons: the cost function is not quite obvious
 - Other Regularization:
    - Data Augmentation:
      - eg: For a picture, can flip the picture and add it to dataset --> add infomation  (对图片进行一些操作：翻转，扭曲，放大缩小...)
    - Early Stopping:
      - stop the training in half way (at the point the J for the test set reaches its minimum)
      
5. Ways to speed up training:
  - normalize the input data on the whole dataset (don't first split to train/dev!) -> helps gradient descent (learning rate)
  
6. Vanishing/exploding gradient (梯度消失/梯度爆炸）
  -  if have a deep NN, 0.9^200 very small, 1.1^200 very large

7. Minibatch:
  - Why Minibatch?
    - Train model more quickly
  - Mini Batch Gradient Descent
    - for each small batch, run Gradient Descent and get W[t] & b[t], use this parameter as the initial value in next small batch
  - cost function will not monotonic decreasing (iteration) <- training on different data set
  - How to choose MiniBatch Size?
    - if training set is small, then don't use mini batch
    - Typically, use a minibatch size between 64 - 512
  - How to tune the MiniBatch Size?
    - Goal:find a size that reduce the cost function more quickly

8. Optimization Algorithms:
  - Gradient Descent
  - Momentum -> generally ,\beta=0.9
  - RMSprop - increase learning on w direction, slow down oscilliation on b direction -> dW small, db larger,generally ,\beta=0.999,\eybsilon = 10 ^(-8)
  - Adam optimization <- combined momentun & RMSprop
  
9 Learning rate decay:
  - slowly reduce learning rate \alpha after iteration
Exponentially weighted averages

10. Hyperparameters tuning: (Don't use grid)
  - use RandomizedSearch
  - coarse the fine

11. Approriate scale of htperparameters:
  - search on a log scale: eg(0.0001 - 1,do not sample uniformely, use log to separate 0.0001,0.001,0.01,0.1,1) <- random sample r in [0,4], set para = 10 ^ r

12. batch normalization:
  - given some intermediae variables in the hedden layer(z[1],z[2],...), z_new = \gamma * z_norm + \beta
  - if normalized, then there is no need for b_i
<h2> Course 3 </h2>
1. ML Strategy: fit training set will -> fit dev set will -> fit testing set will -> performs well

2. How to choose a better model
  - Use a single number evaluation matric(try to set only one criterion to select better model):   
    - if use a confusion matrix:     
     | Tables    | Precision  | Recall|     
     | ----------|:----------:| -----:|    
     | Model A   | 95%        |  90%  |    
     | Model B   | 98%        |  85%  |    
    
    - Problem: Note sure wich is better
    - so we can use F1 score ( a combination of precision and recall)
  - satisficing and optimizing metrics:     
     | Tables    | Accuracy   | running time|    
     | --------- |:----------:| -----------:|    
     | Model A   | 95%        |  90ms       |    
     | Model B   | 98%        |  85ms       |    
     
     - sometimes, we may find it is hard to combine two criterions to one index, then we may use satisficing and optimizing metrics
     - find min(Accuracy) condition on the running time smaller than t
      
3. Comparing to human-level performance:
  - algorithms should be better than human performance , but alway poorer than Bayes Optimal Error
  - sometimes, we will seem human level error as an estimation of Bayes Optimal Error
  - human do well on unstructures data(language, picture,...)
  
4. Avoidable Bias: training error - Bayes Optimal Error

5. Cleaning incorrect labeling data (消除人为label的问题)
  - the wrong label is in training set, it's okay. ML is robust to random errors in training set
  - when evaluating dev/test set: add a column to count incorrected label

6. Bias and Variance with mismatch datadistributions:
  - if training set and dev/test set use different source of data, then it is hard to decide whenther the diff of error rate is caused by high-variance or by different distribution
    - How to solve: create a training-dev set and check the error rate on training & training-dev set &dev set:
      - if the diff between training & training-dev set is big, then cause by high variance
      - if the diff between dev & training-dev set is big, then cause by data mismatch
      
 7 How to solve data mismatch problem?
    - Find what cause the mismatch pronlem
    - add similar noise
 
 8. Transfer Learning:
  - just reset the weight in last layer, use other pree-trained weight as new intial value and retrain
  - assumptions: 
      - transfer from a large dataset to a smaller dataset
      - have same input
      
9. multi-task learning
  - change the dimension of y, then we can train multi task simultaneously
  
10 end-to-end dl
  - 
<h2> Course 4 </h2> - Convolutional Neural Networks
1. Computer Vision: if a picture is 1000 * 1000, then the input layer will be very large, and the weight matrix is also very large. In order to solve such problem, comes up the idea of Convolutional Neural Networks.

2. Edge Detection
  - Vertical edges
  - horizontal edges
  - Different types of filter: (Vertical)
    - Sobel Filter : [[1,0,-1],[2,0,-2],[1,0,-1]]
    - Scharr Filter: [[3,0,-3],[10,0,-10],[3,0,-3]]
    - using backfoward to get the filter

3. Padding
  - Why Padding: 
    - When doing the edge detection, the element in the corners only be use few times, while the elements in the center will be uses for couple times -> lose the information in the corner
    - If you have a deep NN, then through each detection, you will get a small image -> get a very small image after the whole proess
  - What is Padding:
    - Add n new columns/rows with elements 0 : if n = 1, then a 6 * 6 matrix -> after padding -> 8 * 8 matrix -> after convolution (3 * 3) -> a 6 * 6 image
    
    
4. Valid Convolution VS Same Concolution:
  - Valid Convolution: no padding
  - Same Convolution: input size = output size
  - Strided Convolution

5. Pooling:
  - Why pooling?
    -
  - max pooling (prefer) - generally not use padding
    - have hyperparameter , but no parameter to learn
  - average pooling - generally not use padding
    - have hyperparameter , but no parameter to learn
    
6. CNN Model   
  - number of filter will increase through each layer

7. Classic CNN Model:
  - LeNet-5
  - AlexNet
  - VGG
ResNet
Inception

<h2> Course 5 </h2> - Sequence Models
1. RNN:
  - gradient vanishing:(hard to solve and detect)
    - hard to capture long-term denpendence
    - Solution:
      - Gated Recurrent Unit(GRU)
      - LSTM
  - gradient explosion:(easy to detect)
    - solution: gradient clipping :当梯度大于某个值的时候，对其进行缩小
    
    
