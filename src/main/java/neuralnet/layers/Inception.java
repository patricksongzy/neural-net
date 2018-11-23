package neuralnet.layers;

import neuralnet.activations.ActivationType;
import neuralnet.costs.Cost;
import neuralnet.initializers.Initializer;
import neuralnet.optimizers.UpdaterType;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.LinkedList;

public class Inception implements Layer {
	private int batchSize;
	private int height, width, depth;
	private int[] filterAmounts;

	private float[] output;

	private Layer[] bottleneck;
	private Layer[] conv;

	private Inception(LinkedList<Integer> filterAmounts, Initializer initializer) {
		if (filterAmounts.size() != 6)
			throw new IllegalArgumentException("Filter size lengths not correct");

		this.filterAmounts = new int[6];
		for (int i = 0; i < filterAmounts.size(); i++) {
			this.filterAmounts[i] = filterAmounts.get(i);
		}

		bottleneck = new Layer[]{
			new Convolutional.Builder().pad(0).stride(1).initializer(initializer).filterSize(1).filterAmount(filterAmounts.remove())
				.activationType(ActivationType.RELU).build(),
			new Convolutional.Builder().pad(0).stride(1).initializer(initializer).filterSize(1).filterAmount(filterAmounts.remove())
				.activationType(ActivationType.RELU).build(),
			new Convolutional.Builder().pad(0).stride(1).initializer(initializer).filterSize(1).filterAmount(filterAmounts.remove())
				.activationType(ActivationType.RELU).build(),
			new Pooling.Builder().downsampleSize(3).downsampleStride(1).pad(1).build()
		};

		conv = new Layer[]{
			new Convolutional.Builder().pad(1).stride(1).initializer(initializer).filterSize(3).filterAmount(filterAmounts.remove())
				.activationType(ActivationType.RELU).build(),
			new Convolutional.Builder().pad(2).stride(1).initializer(initializer).filterSize(5).filterAmount(filterAmounts.remove())
				.activationType(ActivationType.RELU).build(),
			new Convolutional.Builder().pad(0).stride(1).initializer(initializer).filterSize(1).filterAmount(filterAmounts.remove())
				.activationType(ActivationType.RELU).build(),
		};
	}

	Inception(DataInputStream dis, UpdaterType updaterType) throws IOException {
		height = dis.readInt();
		width = dis.readInt();
		depth = dis.readInt();

		filterAmounts = new int[dis.readInt()];
		for (int i = 0; i < filterAmounts.length; i++) {
			filterAmounts[i] = dis.readInt();
		}

		bottleneck = new Layer[dis.readInt()];
		for (int i = 0; i < bottleneck.length; i++) {
			bottleneck[i] = LayerType.fromString(dis, updaterType);
		}

		conv = new Layer[dis.readInt()];
		for (int i = 0; i < conv.length; i++) {
			conv[i] = LayerType.fromString(dis, updaterType);
		}
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(height);
		dos.writeInt(width);
		dos.writeInt(depth);

		dos.writeInt(filterAmounts.length);
		for (int filterAmount : filterAmounts) {
			dos.writeInt(filterAmount);
		}

		dos.writeInt(bottleneck.length);
		for (Layer layer : bottleneck) {
			dos.writeUTF(layer.getType().toString());
			layer.export(dos);
		}

		dos.writeInt(conv.length);
		for (Layer layer : conv) {
			dos.writeUTF(layer.getType().toString());
			layer.export(dos);
		}
	}

	public void setDimensions(int[] dimensions, UpdaterType updaterType) {
		if (dimensions.length != 3)
			throw new IllegalArgumentException("Invalid input dimensions.");

		this.height = dimensions[0];
		this.width = dimensions[1];
		this.depth = dimensions[2];

		for (Layer layer : bottleneck) {
			layer.setDimensions(dimensions, updaterType);
		}

		for (int i = 0; i < conv.length; i++) {
			conv[i].setDimensions(bottleneck[i + 1].getOutputDimensions(), updaterType);
		}
	}

	public void setMode(Mode mode) {
		for (Layer layer : bottleneck) {
			layer.setMode(mode);
		}

		for (Layer layer : conv) {
			layer.setMode(mode);
		}
	}

	public LayerType getType() {
		return LayerType.INCEPTION;
	}

	public float[] forward(float[] input, int batchSize) {
		float[][] outputs = new float[bottleneck.length][];

		this.batchSize = batchSize;

		for (int i = 0; i < bottleneck.length; i++) {
			outputs[i] = bottleneck[i].forward(input, batchSize);
		}

		for (int i = 0; i < conv.length; i++) {
			outputs[i + 1] = conv[i].forward(outputs[i + 1], batchSize);
		}

		int outputDepth = getOutputDimensions()[2];
		output = new float[height * width * outputDepth * batchSize];
		int offset = 0;
		for (int i = 0; i < bottleneck.length; i++) {
			int filterAmount = filterAmounts[i == 0 ? 0 : i + 2];

			for (int b = 0; b < batchSize; b++) {
				for (int f = 0; f < filterAmount; f++) {
					for (int h = 0; h < height; h++) {
						System.arraycopy(outputs[i], width * (h + height * (f + filterAmount * b)), output,
							width * (h + height * ((f + offset) + outputDepth * b)), width);
					}
				}
			}

			offset += filterAmount;
		}

		return output;
	}

	public float[] backward(Cost cost, float[] target, boolean calculateDelta) {
		return backward(cost.derivative(output, target, batchSize), calculateDelta);
	}

	public float[] backward(float[] previousDelta, boolean calculateDelta) {
		float[][] deltas = new float[bottleneck.length][];

		int outputDepth = getOutputDimensions()[2];
		int offset = 0;

		for (int i = 0; i < bottleneck.length; i++) {
			int filterAmount = filterAmounts[i == 0 ? 0 : i + 2];
			deltas[i] = new float[filterAmount * batchSize * height * width];

			for (int b = 0; b < batchSize; b++) {
				for (int f = 0; f < filterAmount; f++) {
					for (int h = 0; h < height; h++) {
						System.arraycopy(previousDelta, width * (h + height * ((f + offset) + outputDepth * b)), deltas[i],
							width * (h + height * (f + filterAmount * b)), width);
					}
				}
			}

			offset += filterAmount;
		}

		for (int i = 0; i < conv.length; i++) {
			deltas[i + 1] = conv[i].backward(deltas[i + 1], true);
		}

		for (int i = 0; i < bottleneck.length; i++) {
			deltas[i] = bottleneck[i].backward(deltas[i], calculateDelta);
		}

		float[] delta = new float[batchSize * height * width * depth];

		for (int i = 0; i < bottleneck.length; i++) {
			for (int b = 0; b < batchSize; b++) {
				for (int k = 0; k < depth; k++) {
					for (int h = 0; h < height; h++) {
						for (int w = 0; w < width; w++) {
							delta[w + width * (h + height * (k + depth * b))] += deltas[i][w + width * (h + height * (k + depth * b))];
						}
					}
				}
			}
		}

		return delta;
	}

	public int[] getOutputDimensions() {
		return new int[]{height, width, (filterAmounts[0] + filterAmounts[3] + filterAmounts[4] + filterAmounts[5])};
	}

	public float[][][] getParameters() {
		int length = 0;

		for (Layer layer : bottleneck) {
			length += layer.getParameters().length;
		}

		for (Layer layer : conv) {
			length += layer.getParameters().length;
		}

		float[][][] parameters = new float[length][][];

		System.arraycopy(bottleneck[0].getParameters(), 0, parameters, 0, bottleneck[0].getParameters().length);
		int offset = bottleneck[0].getParameters().length;

		for (int i = 1; i < bottleneck.length; i++) {
			System.arraycopy(bottleneck[i].getParameters(), 0, parameters, offset, bottleneck[i].getParameters().length);
			offset += bottleneck[i].getParameters().length;

			System.arraycopy(conv[i - 1].getParameters(), 0, parameters, offset, conv[i - 1].getParameters().length);
			offset += conv[i - 1].getParameters().length;
		}

		return parameters;
	}

	public void update(int size) {
		for (Layer layer : bottleneck) {
			layer.update(size);
		}

		for (Layer layer : conv) {
			layer.update(size);
		}
	}

	/**
	 * Builder for Inception layers.
	 */
	@SuppressWarnings({"unused", "WeakerAccess"})
	public static class Builder {
		private Initializer initializer;
		private LinkedList<Integer> filterAmounts = new LinkedList<>();

		public Builder filterAmount(int... filterAmounts) {
			for (int value : filterAmounts)
				this.filterAmounts.add(value);
			return this;
		}

		public Builder initializer(Initializer initializer) {
			this.initializer = initializer;
			return this;
		}

		public Inception build() {
			return new Inception(filterAmounts, initializer);
		}
	}
}