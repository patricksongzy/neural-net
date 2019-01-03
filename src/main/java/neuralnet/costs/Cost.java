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
	 * @param targets the targets
	 * @return the cost
	 */
	float cost(float[] out, float[] targets);

	/**
	 * This method calculates the derivative of the cost.
	 *
	 * @param output the neural network output
	 * @param targets the targets
	 * @return the derivative of cost
	 */
	float[] derivative(float[] output, float[] targets, int batchSize);

	/**
	 * This method calculates the derivative of cost with the softmax function.
	 *
	 * @param output    the neural network output
	 * @param targets   the targets
	 * @param batchSize the batch size
	 * @return the derivative of cost with respect to the input to the softmax function
	 */
	float[] derviativeSoftmax(float[] output, float[] targets, int batchSize);
}