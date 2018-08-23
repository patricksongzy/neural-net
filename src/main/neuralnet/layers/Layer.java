package main.neuralnet.layers;

import main.neuralnet.costs.Cost;

import java.io.DataOutputStream;
import java.io.IOException;

public interface Layer {
	/**
	 * Sets the mode of the layer. Affects whether gradients get stored and whether Dropout layers drop connections.
	 *
	 * @param mode the mode
	 */
	void setMode(Mode mode);

	/**
	 * Gets the LayerType. This can be used when exporting.
	 * @return the layer type
	 */
	LayerType getType();

	/**
	 * Sets the dimensions, given a previous layers dimensions.
	 *
	 * @param dimensions the dimensions of the previous layer
	 */
	void setDimensions(int... dimensions);

	/**
	 * Retrieves the parameters and gradients for gradient checking.

	 * @return the parameters and gradients
	 */
	float[][][] getParameters();

	/**
	 * Forward propagation of a layer.
	 *
	 * @param x the input
	 * @return the output
	 */
	float[][] forward(float[][] x);

	/**
	 * Back propagation of an output layer.
	 *
	 * @param cost the cost function
	 * @param target the target output
	 * @return the delta
	 */
	float[][] backward(Cost cost, float[][] target);

	/**
	 * Back propagation of a hidden layer, given the layer that was back propagated before.
	 *
	 * @param previousDelta the previous delta
	 * @return the delta
	 */
	float[][] backward(float[][] previousDelta);

	/**
	 * Gets the output dimensions, for initializing following layers.
	 *
	 * @return the output dimensions
	 */
	int[] getOutputDimensions();

	/**
	 * Exports the layer to an output stream.
	 *
	 * @param dos the output stream
	 */
	void export(DataOutputStream dos) throws IOException;

	/**
	 * The modes of a layer. Affect whether gradients get stored and whether Dropout layers drop connections.
	 */
	enum Mode {
		EVAL, TRAIN, GRADIENT_CHECK
	}
}