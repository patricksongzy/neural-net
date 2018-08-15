# neural-net
A neural network implementation in Java. This implementation passes gradient checks, but does not guaranty accuracy. This project has no commit history, because it was copied off of one of my private repositories.
## Project
This project includes:
* Convolutional, Feed-Forward, Down-Sampling/Pooling, Dropout and GRU layers.
* ReLU, Identity/Linear, Sigmoid, Softmax and TanH Activations.
* Cross-Entropy and Mean-Square-Error costs.
* He Gaussian initialization.
* Adam optimizer.

## Problems
* The implementation uses Aparapi, which does not train as fast as custom-made OpenCL kernels, allowing it to remain purely Java.
* The Plot.java file is a mess, but is not essential to the program. It was merely created to show a general idea of training progress.
* Variable naming in GRU is not completely correct (as they should be named to represent before-activation derivatives ie. if `y_t = x_t * w_t;` and `x_{t + 1} = activation(y_t);` then the "before-activation" derivative would be `dx_{t + 1}/dy_t`). I was unable to come up with proper variable names. (dhRaw is used [Here](https://gist.github.com/karpathy/d4dee566867f8291f086))
* This project does not use a matrix multiplication library for clarity.

## Requirements
This project requires [Aparapi](https://github.com/Syncleus/aparapi), created by AMD, forked by freemo.
