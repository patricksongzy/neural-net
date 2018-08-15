package main.neuralnet.initializers;

/**
 * Initializers initialize weights to optimal values, given a distribution.
 */
public interface Initializer {
	/**
	 * Initializes the weights.
	 *
	 * @param inputSize the input size
	 * @return the weight
	 */
	double initialize(int inputSize);
}