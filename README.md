# neural-net
A neural network implementation in Java. This implementation passes the current tests that have been implemented.
Please note: There is currently no versioning system for this project at the moment. Although the pom.xml file displays a version for the project, it is not updated.
## Project
This project includes:
* Convolutional, Feed-Forward, Down-Sampling/Pooling, Dropout and GRU layers.
* ReLU, Identity/Linear, Sigmoid, Softmax and TanH Activations.
* Cross-Entropy and Mean-Square-Error costs.
* BLAS vectorization for Feed-Forward and GRU layers.
* [He](https://arxiv.org/abs/1502.01852) Gaussian initialization.
* [Adam](https://arxiv.org/abs/1412.6980) optimizer.

## Problems
* The Plot.java file is a mess, but is not essential to the program. It was merely created to show a general idea of training progress.
* Variable naming in GRU is not completely correct (as they should be named to represent before-activation derivatives ie. if `y_t = x_t * w_t;` and `x_{t + 1} = activation(y_t);` then the "before-activation" derivative would be `dx_{t + 1}/dy_t`). I was unable to come up with proper variable names. (dhRaw is used [Here](https://gist.github.com/karpathy/d4dee566867f8291f086))
* Convolutional layers and pooling layers are not accelerated with a BLAS yet. Fast fourrier transforms (FFT), custom kernels or xgemm can all be used. This may be implemented in the future.