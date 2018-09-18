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
  - data for dev and test must from same data sourse, which means same distribution. But training set can use another dataset, because we choose the algorithms based on the performance of the dev set
 
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

<h2> Course 3 </h2>
1. ML Strategy: fit training set will -> fit dev set will -> fit testing set will -> performs well

2. Use a single number evaluation matric:
    | Tables    | Precision  | Recall|
    | --------- |:----------:| -----:|
    | Model A   | 95%        |  90%  |
    | Model B   | 98%        |  85%  |
