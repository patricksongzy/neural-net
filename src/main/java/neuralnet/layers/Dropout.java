package neuralnet.layers;

import neuralnet.costs.Cost;
import neuralnet.optimizers.UpdaterType;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

/**
 * The Dropout layer drops certain connections to reduce over-fitting during training. During evaluation, dropout layers do not take effect.
 */
public class Dropout implements Layer {
	private Mode mode = Mode.TRAIN;

	private int batchSize;
	private int inputSize;
	private int[] dimensions;
	private float dropout;
	private float[] output;

	private Dropout(float dropout) {
		if (dropout < 0 || dropout >= 1)
			throw new IllegalArgumentException("Dropout must be > 0 and < 1.");

		this.dropout = dropout;
	}

	Dropout(DataInputStream dis) throws IOException {
		inputSize = dis.readInt();
		dropout = dis.readFloat();
	}

	public void setDimensions(int[] dimensions, UpdaterType updaterType) {
		this.dimensions = dimensions;

		inputSize = dimensions[0];
		for (int i = 1; i < dimensions.length; i++)
			inputSize *= dimensions[i];
	}

	public void setMode(Mode mode) {
		this.mode = mode;
	}

	public float[] backward(Cost cost, float[] target, boolean calculateDelta) {
		if (calculateDelta) {
			return cost.derivative(output, target, batchSize);
		}

		return null;
	}

	public LayerType getType() {
		return LayerType.DROPOUT;
	}

	public float[] backward(float[] previousDelta, boolean calculateDelta) {
		return previousDelta;
	}

	public int[] getOutputDimensions() {
		return dimensions;
	}

	public float[][][] getParameters() {
		return new float[0][][];
	}

	public void update(int length) {
	}

	public float[] forward(float[] input, int batchSize) {
		this.batchSize = batchSize;

		if (mode == Mode.TRAIN) {
			output = new float[batchSize * inputSize];

			IntStream.range(0, batchSize).parallel().forEach(b -> {
				for (int i = 0; i < inputSize; i++) {
					// if a random float is past the dropout threshold, then drop the connection by setting the output to zero
					if (ThreadLocalRandom.current().nextFloat() < dropout)
						output[i + inputSize * b] = 0;
					else
						output[i + inputSize * b] = input[i + inputSize * b] / (1 - dropout);
				}
			});

			return output;
		}

		// during evaluation, dropout does not take effect
		return input;
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(inputSize);
		dos.writeFloat(dropout);
	}

	/**
	 * Builder for Dropout layers.
	 */
	@SuppressWarnings({"unused", "WeakerAccess"})
	public static class Builder {
		private float dropout = 0.5f;

		/**
		 * The dropout is the chance that a connection will be dropped, during training.
		 *
		 * @param dropout the dropout
		 * @return the builder
		 */
		public Builder dropout(float dropout) {
			this.dropout = dropout;
			return this;
		}

		/**
		 * Builds the layer.
		 *
		 * @return the builder
		 */
		public Dropout build() {
			return new Dropout(dropout);
		}
	}
}
