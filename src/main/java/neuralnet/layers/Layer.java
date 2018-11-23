package neuralnet.layers;

import neuralnet.costs.Cost;
import neuralnet.optimizers.UpdaterType;

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
	 * @param updaterType the updater type
	 */
	void setDimensions(int[] dimensions, UpdaterType updaterType);

	/**
	 * Retrieves the parameters and gradients for gradient checking.

	 * @return the parameters and gradients
	 */
	float[][][] getParameters();

	/**
	 * Forward propagation of a layer.
	 *
	 * @param input the input
	 * @param batchSize the batch size
	 * @return the output
	 */
	float[] forward(float[] input, int batchSize);

	/**
	 * Back propagation of an output layer.
	 *
	 * @param cost the cost function
	 * @param targets the target outputs
	 * @param calculateDelta whether to calculate delta
	 * @return the delta
	 */
	float[] backward(Cost cost, float[] targets, boolean calculateDelta);

	/**
	 * Back propagation of a hidden layer, given the layer that was back propagated before.
	 *
	 * @param previousDelta the previous delta
	 * @param calculateDelta whether to calculate delta
	 * @return the delta
	 */
	float[] backward(float[] previousDelta, boolean calculateDelta);

	/**
	 * Updates a layer's parameters.
	 *
	 * @param scale the scale
	 */
	void update(int scale);

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