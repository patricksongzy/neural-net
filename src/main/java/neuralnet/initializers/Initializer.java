package neuralnet.initializers;

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
	float initialize(int inputSize);
}
