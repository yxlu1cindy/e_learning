# DeepLeaning

Keras & Tensorflow:   
  - Both for Deep Learning
  - Keras can be seen as an API build on the Tensorflow
  - Keras is easier to use; Tensorflow more flexible
  
Keras:
  -  Core: __model__-- a way to organize layers:
      - Sequential -- a linear stack of layers 
      - Model class API
  -  Core Layers:
      - Dense: -- fully connected layer
          - Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
      - Activation:
          - Applies an activation function to an output.
      - Flatten:
      - Input:
      - Conv1D
      - Activations:
        - *softmax*
        - https://keras.io/activations/
        
  - Compile
    -  Optimizer -- one of the two arguments required for compiling a Keras model:   
        - SGD: Stochastic gradient descent optimizer.   
            - sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)    
              - lr: learning rate   
              - decay: float >= 0. Learning rate decay over each update   
              - nesterov: boolean. Whether to apply *Nesterov momentum*.   
        - RMSprop: *RMSProp optimizer*
            - Recommended to leave the parameters of this optimizer at their default values (except the learning rate).
            - This optimizer is usually a good choice for recurrent neural networks.
        - https://keras.io/optimizers/
    -   loss functions:
        - l1,l2,...
        - *hinge*
        - categorical data
    -   *metrics:*
  - fit:
      - batch_size: Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32
      - validation_split: Float between 0 and 1
      - epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided
      

              
