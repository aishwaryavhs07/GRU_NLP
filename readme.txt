
                                                                                       README- ADVANCED MODEL


My advanced model consists of a convolution layer, a max pooling and a dense layer.

1)   In the advanced model, I created a convolution layer using Keras.Conv2D Class and passed kernel size, activation function and 
       number of filters as parameters
2)  An input word is passed to the convolution layer
3)  Max pooling is used to reduce the spatial dimensions of the output obtained from the Conv2D layer.
4)  This output is flattened using the flatten() method to obtain the extra dimension
5)  The output obtained the previous step is passed to dense layer.
6) This is the final output
