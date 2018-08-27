package neuralnet.costs;

/**
 * Cost functions evaluate the performance of a neural network. They must be differentiable, in order to back-propagate.
 */
public interface Cost {
	/**
	 * This method returns the cost type for use when exporting neural networks.
	 *
	 * @return the cost type
	 */
	CostType getType();

	/**
	 * This method calculates the cost.
	 *
	 * @param out the neural network output
	 * @param target the target
	 * @return the cost
	 */
	float cost(float[] out, float[] target);

	/**
	 * This method calculates the derivative of the cost.
	 *
	 * @param output the neural network output
	 * @param target the target
	 * @return the derivative of cost
	 */
	float[] derivative(float[] output, float[] target, int batchSize);

	float[] derviativeSoftmax(float[] output, float[] target, int batchSize);
}