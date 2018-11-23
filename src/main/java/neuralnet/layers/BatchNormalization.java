package neuralnet.layers;

import neuralnet.activations.Activation;
import neuralnet.activations.ActivationType;
import neuralnet.costs.Cost;
import neuralnet.initializers.Initializer;
import neuralnet.optimizers.UpdaterType;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

@SuppressWarnings("FieldCanBeLocal")
public class BatchNormalization implements Layer {
	private Mode mode;

	private int batchSize;
	private int height, width, depth;

	private float epsilon;
	private float[] mean, variance;
	private float[] weights, biases;
	private float[] output;

	private Activation activation;
	private Initializer initializer;

	private BatchNormalization(float epsilon, Initializer initializer, ActivationType activationType) {
		this.epsilon = epsilon;
		this.initializer = initializer;
		this.activation = activationType;
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

		activation = Activation.fromString(dis);

		weights = new float[depth];
		biases = new float[depth];
		for (int i = 0; i < weights.length; i++) {
			weights[i] = dis.readFloat();
			biases[i] = dis.readFloat();
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

		activation.export(dos);

		for (int i = 0; i < depth; i++) {
			dos.writeFloat(weights[i]);
			dos.writeFloat(biases[i]);
		}
	}

	public void setDimensions(int[] dimensions, UpdaterType updaterType) {
		if (dimensions.length == 3) {
			this.height = dimensions[0];
			this.width = dimensions[1];
			this.depth = dimensions[2];
		} else {
			height = 1;
			width = 1;

			depth = dimensions[0];
			for (int i = 1; i < dimensions.length; i++)
				depth *= dimensions[i];
		}

		mean = new float[depth];
		variance = new float[depth];

		weights = new float[depth];
		biases = new float[depth];

		int inputSize = height * width * depth;
		for (int i = 0; i < depth; i++) {
			weights[i] = initializer.initialize(inputSize);
			biases[i] = initializer.initialize(inputSize);
		}
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
		}

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
		return new float[][][]{{weights, new float[weights.length]}, {biases, new float[biases.length]},
			{mean, new float[mean.length]}, {variance, new float[variance.length]}};
	}

	public void update(int size) {
	}

	/**
	 * Builder for BatchNormalization layers.
	 */
	@SuppressWarnings({"unused", "WeakerAccess"})
	public static class Builder {
		private float epsilon;
		private Initializer initializer;
		private ActivationType activationType;

		public Builder() {
			epsilon = 1e-5f;
			activationType = ActivationType.IDENTITY;
		}

		public Builder epsilon(float epsilon) {
			this.epsilon = epsilon;
			return this;
		}

		public Builder initializer(Initializer initializer) {
			this.initializer = initializer;
			return this;
		}

		public Builder activationType(ActivationType activationType) {
			this.activationType = activationType;
			return this;
		}

		public BatchNormalization build() {
			return new BatchNormalization(epsilon, initializer, activationType);
		}
	}
}