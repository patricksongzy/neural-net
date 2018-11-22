# neural-net
A neural network implementation in Java, for ICS3U and ICS4U. This implementation passes the current tests that have been implemented.

## Project
This project includes:
* Convolutional, Fully-connected, Down-Sampling/Pooling, Dropout, GRU, Inception, Resnet, Batch Norm layers (and more).
* ReLU, Identity/Linear, Sigmoid, Softmax and TanH Activations.
* Cross-Entropy and Mean-Square-Error costs.
* BLAS vectorization for some layers using JOCL and JOCLBlast.

## Example Usage
```java
Model model = new Model.Builder().add(
    new Convolutional.Builder().filterSize(7).filterAmount(64).pad(3).activationType(ActivationType.RELU)
    .initializer(new Constant(0)).stride(2).updaterType(UpdaterType.ADAM).build()
).add(
    new Dense.Builder().outputSize(10).activation(OutputActivationType.SOFTMAX)
    .initializer(new Constant(0)).updaterType(UpdaterType.ADAM).build()
).cost(CostType.SPARSE_CROSS_ENTROPY).inputDimensions(224, 224, 3).build(); 
```

## Problems
* Variable naming in GRU is not completely correct (as they should be named to represent before-activation derivatives ie. if `y_t = x_t * w_t;` and `x_{t + 1} = activation(y_t);` then the "before-activation" derivative would be `dx_{t + 1}/dy_t`). I was unable to come up with proper variable names. (dhRaw is used [Here](https://gist.github.com/karpathy/d4dee566867f8291f086))
* Models are all sequential. Non-sequential models would have to create new layers. (TODO)
* R-CNN, GAN and Inception-Resnet are still TODO.
* PSP net is not working correctly (don't know why).

## No Computational Graphs
* Makes it a bit clearer on how everything is implemented.

## License
* MIT license.
