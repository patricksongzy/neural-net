package neuralnet.activations;

/**
 * Activations simulate the firing of neurons in a neural network. They must be differentiable, in order to back-propagate.
 */
public interface Activation {
	/**
	 * This method returns the activation type for use when exporting neural networks.
	 *
	 * @return the activation type
	 */
	ActivationType getType();

	/**
	 * This method simulates the activation of neurons. It directly modifies the input.
	 *
	 * @param x the pre-activated input
	 */
	void activation(float[] x);

	/**
	 * This method simulates the activation of neurons. It directly modifies the input.
	 *
	 * @param x the pre-activated input
	 */
	void activation(float[][] x);

	/**
	 * This method calculates the derivative of the activation.
	 *
	 * @param x the activated output
	 */
	float[][] derivative(float[][] x);
}