package neuralnet.layers;

import neuralnet.activations.Identity;
import neuralnet.costs.Cost;

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
	private float dropout;
	private float[][] output;

	private Dropout(float dropout) {
		this.dropout = dropout;
	}

	Dropout(DataInputStream dis) throws IOException {
		inputSize = dis.readInt();
		dropout = dis.readFloat();
	}

	public void setDimensions(int... dimensions) {
		inputSize = dimensions[0];
	}

	public void setMode(Mode mode) {
		this.mode = mode;
	}

	public float[][] backward(Cost cost, float[][] target) {
		float[][] delta = new float[target.length][];

		for (int t = 0; t < output.length; t++) {
			delta[t] = cost.derivative(output[t], target[t], new Identity(), batchSize);
		}

		return delta;
	}

	public LayerType getType() {
		return LayerType.DROPOUT;
	}

	public float[][] backward(float[][] previousDelta) {
		return previousDelta;
	}

	public int[] getOutputDimensions() {
		return new int[]{inputSize};
	}

	public float[][][] getParameters() {
		return new float[0][][];
	}

	public float[][] forward(float[][] input, int batchSize) {
		this.batchSize = batchSize;

		if (mode == Mode.TRAIN) {
			output = new float[input.length][batchSize * inputSize];

			int size = input.length / batchSize;
			for (int t = 0; t < input.length; t++) {
				int time = t;

				IntStream.range(0, batchSize).parallel().forEach(b -> {
					for (int i = 0; i < size; i++) {
						// if a random float is past the dropout threshold, then drop the connection by setting the output to zero
						if (ThreadLocalRandom.current().nextFloat() < dropout)
							output[time][i + size * b] = 0;
						else
							output[time][i + size * b] = input[time][i + size * b] / dropout;
					}
				});
			}

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
		 */
		public Builder dropout(float dropout) {
			this.dropout = dropout;
			return this;
		}

		public Dropout build() {
			if (dropout > 0)
				return new Dropout(dropout);

			throw new IllegalArgumentException();
		}
	}
}