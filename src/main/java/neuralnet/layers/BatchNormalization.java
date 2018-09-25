package neuralnet.layers;

import neuralnet.costs.Cost;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class BatchNormalization implements Layer {
	private Mode mode = Mode.TRAIN;

	private int batchSize;
	private int height, width, depth;

	private float epsilon;
	private float[] output;
	private float[] mean, variance;

	private BatchNormalization(float epsilon) {
		this.epsilon = epsilon;
	}

	BatchNormalization(DataInputStream dis) throws IOException {
		height = dis.readInt();
		width = dis.readInt();
		depth = dis.readInt();
		epsilon = dis.readFloat();

		mean = new float[depth];
		variance = new float[depth];
		for (int i = 0; i < mean.length; i++) {
			mean[i] = dis.readFloat();
			variance[i] = dis.readFloat();
		}
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(height);
		dos.writeInt(width);
		dos.writeInt(depth);
		dos.writeFloat(epsilon);

		for (int i = 0; i < depth; i++) {
			dos.writeFloat(mean[i]);
			dos.writeFloat(variance[i]);
		}
	}

	public void setDimensions(int... dimensions) {
		if (dimensions.length != 3)
			throw new IllegalArgumentException("Invalid input dimensions.");

		this.height = dimensions[0];
		this.width = dimensions[1];
		this.depth = dimensions[2];

		mean = new float[depth];
		variance = new float[depth];
	}

	public void setMode(Mode mode) {
		this.mode = mode;
	}

	public LayerType getType() {
		return LayerType.BATCH_NORMALIZATION;
	}

	public float[] forward(float[] input, int batchSize) {
		this.batchSize = batchSize;

		if (mode == Mode.TRAIN) {
			float[] mean = new float[depth];

			for (int b = 0; b < batchSize; b++) {
				for (int i = 0; i < depth; i++) {
					for (int j = 0; j < height * width; j++) {
						mean[i] += input[j + (height * width) * (i + depth * b)];
					}
				}
			}

			float[] variance = new float[depth];
			for (int b = 0; b < batchSize; b++) {
				for (int i = 0; i < depth; i++) {
					for (int j = 0; j < height * width; j++) {
						mean[i] /= batchSize;
						this.mean[i] += mean[i];

						variance[i] += Math.pow(input[j + (height * width) * (i + depth * b)] - mean[i], 2);
					}
				}
			}

			output = new float[batchSize * input.length];
			for (int b = 0; b < batchSize; b++) {
				for (int i = 0; i < depth; i++) {
					for (int j = 0; j < height * width; j++) {
						int index = j + (height * width) * (i + depth * b);

						variance[i] /= batchSize;
						this.variance[i] += variance[i];

						output[index] += (input[index] - mean[i]) / Math.sqrt(variance[i] + epsilon);
					}
				}
			}

			return output;
		} else {
			output = new float[batchSize * input.length];

			for (int b = 0; b < batchSize; b++) {
				for (int i = 0; i < depth; i++) {
					for (int j = 0; j < height * width; j++) {
						int index = j + (height * width) * (i + depth * b);

						output[index] += (input[index] - mean[i]) / Math.sqrt(variance[i] + epsilon);
					}
				}
			}

			return output;
		}
	}

	public float[] backward(Cost cost, float[] target, boolean calculateDelta) {
		return new float[height * width * depth * batchSize];
	}

	public float[] backward(float[] previousDelta, boolean calculateDelta) {
		return new float[height * width * depth * batchSize];
	}

	public int[] getOutputDimensions() {
		return new int[]{height, width, depth};
	}

	public float[][][] getParameters() {
		return new float[][][]{{mean, new float[mean.length]}, {variance, new float[variance.length]}};
	}

	public void update(int size) {
	}

	/**
	 * Builder for BatchNormalization branch2.
	 */
	@SuppressWarnings({"unused", "WeakerAccess"})
	public static class Builder {
		private float epsilon;

		public Builder() {
			epsilon = 1e-5f;
		}

		public Builder epsilon(float epsilon) {
			this.epsilon = epsilon;
			return this;
		}

		public BatchNormalization build() {
			return new BatchNormalization(epsilon);
		}
	}
}