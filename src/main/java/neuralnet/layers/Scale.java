package neuralnet.layers;

import neuralnet.activations.Activation;
import neuralnet.activations.ActivationType;
import neuralnet.costs.Cost;
import neuralnet.initializers.Initializer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class Scale implements Layer {
	private int batchSize;
	private int height, width, depth;
	private float[] weights, biases;
	private float[] output;

	private Activation activation;
	private Initializer initializer;

	private Scale(Initializer initializer, ActivationType activationType) {
		this.initializer = initializer;
		this.activation = activationType;
	}

	Scale(DataInputStream dis) throws IOException {
		height = dis.readInt();
		width = dis.readInt();
		depth = dis.readInt();

		activation = Activation.fromString(dis);

		weights = new float[depth];
		biases = new float[depth];
		for (int i = 0; i < weights.length; i++) {
			weights[i] = dis.readFloat();
			biases[i] = dis.readFloat();
		}
	}

	public void setDimensions(int... dimensions) {
		if (dimensions.length < 3)
			throw new IllegalArgumentException();

		height = dimensions[0];
		width = dimensions[1];
		depth = dimensions[2];

		weights = new float[depth];
		biases = new float[depth];

		int inputSize = height * width * depth;
		for (int i = 0; i < depth; i++) {
			weights[i] = initializer.initialize(inputSize);
			biases[i] = initializer.initialize(inputSize);
		}
	}

	public void setMode(Mode mode) {
	}

	public float[] backward(Cost cost, float[] target, boolean calculateDelta) {
		return new float[height * width * depth * batchSize];
	}

	public LayerType getType() {
		return LayerType.SCALE;
	}

	public float[] backward(float[] previousDelta, boolean calculateDelta) {
		return new float[height * width * depth * batchSize];
	}

	public int[] getOutputDimensions() {
		return new int[]{height, width, depth};
	}

	public float[][][] getParameters() {
		return new float[][][]{{weights, new float[weights.length]}, {biases, new float[biases.length]}};
	}

	public void update(int size) {
	}

	public float[] forward(float[] input, int batchSize) {
		this.batchSize = batchSize;

		output = new float[batchSize * height * width * depth];

		for (int b = 0; b < batchSize; b++) {
			for (int i = 0; i < depth; i++) {
				for (int j = 0; j < height * width; j++) {
					int index = j + (height * width) * (i + depth * b);
					output[index] = input[index] * weights[i] + biases[i];
				}
			}
		}

		activation.activation(output, batchSize);

		return output;
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(height);
		dos.writeInt(width);
		dos.writeInt(depth);

		activation.export(dos);

		for (int i = 0; i < depth; i++) {
			dos.writeFloat(weights[i]);
			dos.writeFloat(biases[i]);
		}
	}

	/**
	 * Builder for Dropout layers.
	 */
	@SuppressWarnings({"unused", "WeakerAccess"})
	public static class Builder {
		private Initializer initializer;
		private ActivationType activationType;

		public Builder() {
			activationType = ActivationType.IDENTITY;
		}

		public Builder initializer(Initializer initializer) {
			this.initializer = initializer;
			return this;
		}

		public Builder activationType(ActivationType activationType) {
			this.activationType = activationType;
			return this;
		}

		public Scale build() {
			return new Scale(initializer, activationType);
		}
	}
}