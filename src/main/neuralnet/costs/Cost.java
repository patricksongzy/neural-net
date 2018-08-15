package main.neuralnet.costs;

import main.neuralnet.activations.Activation;

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
	double cost(double[] out, double[] target);

	/**
	 * This method calculates the derivative of the cost.
	 *
	 * @param output the neural network output
	 * @param target the target
	 * @param activation the activation function
	 * @return the derivative of cost
	 */
	double[][] derivative(double[][] output, double[][] target, Activation activation);
}