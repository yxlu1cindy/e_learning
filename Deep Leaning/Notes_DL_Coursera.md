1. Why deep-learning?
  - perform better on large dataset
  - perform better when large NN

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
