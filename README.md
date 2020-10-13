# neural-net
A neural network implementation in Java, for ICS3U and ICS4U. This implementation passes the current tests that have been implemented.

## Example Project: Bird Recognizer
![A picture of a Bald Eagle being recognized by a bird recognizer](https://patricksongzy.github.io/assets/images/ae.png)

**Model**
```
@inproceedings{Simon15:NAC,
    author = {Marcel Simon and Erik Rodner},
    booktitle = {International Conference on Computer Vision (ICCV)},
    title = {Neural Activation Constellations: Unsupervised Part Model Discovery with Convolutional Networks},
    year = {2015},
}
```

## Project
This project includes:
* Convolutional, Fully-connected, Down-Sampling/Pooling, Dropout, GRU, Inception, ResNet, Batch Norm layers, (and more).
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
* Models are created sequentially. To fix this, Layers should be created as nodes in a graph. Workaround: implementing non-sequential Models as Layers.
* R-CNN, GAN and Inception-Resnet are not implemented yet.
* PSP net is not working correctly (don't know why).

## No Computational Graphs
* An implementation of computational graphs is in-progress.
* Shows how derivatives are computed, though does make implementation less clean.

## License
* MIT license.
