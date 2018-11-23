package neuralnet.layers;

import neuralnet.costs.Cost;
import neuralnet.optimizers.UpdaterType;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

@SuppressWarnings("FieldCanBeLocal")
public class L2 implements Layer {
	private int inputSize;
	private int batchSize;
	private float epsilon;
	private float[] output;

	private L2(float epsilon) {
		this.epsilon = epsilon;
	}

	L2(DataInputStream dis) throws IOException {
		inputSize = dis.readInt();
		epsilon = dis.readFloat();
	}

	public void setDimensions(int[] dimensions, UpdaterType updaterType) {
		inputSize = dimensions[0];
		for (int i = 1; i < dimensions.length; i++)
			inputSize *= dimensions[i];
	}

	public void setMode(Layer.Mode mode) {
	}

	public LayerType getType() {
		return LayerType.L2;
	}

	public float[] forward(float[] input, int batchSize) {
		this.batchSize = batchSize;

		output = new float[input.length];
		for (int b = 0; b < batchSize; b++) {
			float sum = 0;
			for (int i = 0; i < inputSize; i++) {
				float value = input[i + inputSize * b];
				sum += value * value;
			}

			sum = (float) Math.sqrt(Math.max(sum, epsilon));

			for (int i = 0; i < inputSize; i++) {
				int index = i + inputSize * b;
				output[index] = input[index] / sum;
			}
		}

		return output;
	}

	// TODO: implement
	public float[] backward(Cost cost, float[] target, boolean calculateDelta) {
		return new float[batchSize * inputSize];
	}

	// TODO: implement
	public float[] backward(float[] previousDelta, boolean calculateDelta) {
		return new float[batchSize * inputSize];
	}

	public int[] getOutputDimensions() {
		return new int[]{inputSize};
	}

	public float[][][] getParameters() {
		return new float[0][][];
	}

	public void update(int size) {
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(inputSize);
		dos.writeFloat(epsilon);
	}

	/**
	 * Builder for L2 layers.
	 */
	@SuppressWarnings({"unused", "WeakerAccess"})
	public static class Builder {
		private float epsilon = 1e-12f;

		public Builder epsilon(float epsilon) {
			this.epsilon = epsilon;
			return this;
		}

		public L2 build() {
			return new L2(epsilon);
		}
	}
}