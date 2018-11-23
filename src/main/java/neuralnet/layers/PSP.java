package neuralnet.layers;

import neuralnet.activations.ActivationType;
import neuralnet.costs.Cost;
import neuralnet.initializers.Initializer;
import neuralnet.optimizers.UpdaterType;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.LinkedList;

@SuppressWarnings("FieldCanBeLocal")
public class PSP implements Layer {
	private int batchSize;
	private int height, width, depth;

	private LinkedList<Integer> downsampleSizes;
	private Initializer initializer;

	private float[] output;

	private Layer[] branch1;
	private Layer[] branch2;
	private Layer[] branch3;
	private Layer[] branch4;

	private PSP(LinkedList<Integer> downsampleSizes, Initializer initializer) {
		this.downsampleSizes = downsampleSizes;
		this.initializer = initializer;
	}

	PSP(DataInputStream dis, UpdaterType updaterType) throws IOException {
		height = dis.readInt();
		width = dis.readInt();
		depth = dis.readInt();

		branch1 = new Layer[4];
		branch2 = new Layer[4];
		branch3 = new Layer[4];
		branch4 = new Layer[4];
		for (int i = 0; i < 4; i++) {
			branch1[i] = LayerType.fromString(dis, updaterType);
			branch2[i] = LayerType.fromString(dis, updaterType);
			branch3[i] = LayerType.fromString(dis, updaterType);
			branch4[i] = LayerType.fromString(dis, updaterType);
		}
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(height);
		dos.writeInt(width);
		dos.writeInt(depth);

		for (int i = 0; i < 4; i++) {
			dos.writeUTF(branch1[i].getType().toString());
			branch1[i].export(dos);
			dos.writeUTF(branch2[i].getType().toString());
			branch2[i].export(dos);
			dos.writeUTF(branch3[i].getType().toString());
			branch3[i].export(dos);
			dos.writeUTF(branch4[i].getType().toString());
			branch4[i].export(dos);
		}
	}

	public void setDimensions(int[] dimensions, UpdaterType updaterType) {
		if (dimensions.length != 3)
			throw new IllegalArgumentException("Invalid input dimensions.");

		this.height = dimensions[0];
		this.width = dimensions[1];
		this.depth = dimensions[2];

		branch1 = new Layer[]{
			new Pooling.Builder().downsampleSize(downsampleSizes.getFirst()).downsampleStride(downsampleSizes.remove()).mode(Pooling.Mode.AVERAGE).build(),
			new Convolutional.Builder().filterAmount(512).filterSize(1).stride(1).activationType(ActivationType.IDENTITY)
				.initializer(initializer).pad(0).build(),
			new BatchNormalization.Builder().initializer(initializer).activationType(ActivationType.RELU).build(),
			new Interpolation.Builder().outputHeight(height).outputWidth(width).build()
		};

		branch2 = new Layer[]{
			new Pooling.Builder().downsampleSize(downsampleSizes.getFirst()).downsampleStride(downsampleSizes.remove()).mode(Pooling.Mode.AVERAGE).build(),
			new Convolutional.Builder().filterAmount(512).filterSize(1).stride(1).activationType(ActivationType.IDENTITY)
				.initializer(initializer).pad(0).build(),
			new BatchNormalization.Builder().initializer(initializer).activationType(ActivationType.RELU).build(),
			new Interpolation.Builder().outputHeight(height).outputWidth(width).build()
		};

		branch3 = new Layer[]{
			new Pooling.Builder().downsampleSize(downsampleSizes.getFirst()).downsampleStride(downsampleSizes.remove()).mode(Pooling.Mode.AVERAGE).build(),
			new Convolutional.Builder().filterAmount(512).filterSize(1).stride(1).activationType(ActivationType.IDENTITY)
				.initializer(initializer).pad(0).build(),
			new BatchNormalization.Builder().initializer(initializer).activationType(ActivationType.RELU).build(),
			new Interpolation.Builder().outputHeight(height).outputWidth(width).build()
		};

		branch4 = new Layer[]{
			new Pooling.Builder().downsampleSize(downsampleSizes.getFirst()).downsampleStride(downsampleSizes.remove()).mode(Pooling.Mode.AVERAGE).build(),
			new Convolutional.Builder().filterAmount(512).filterSize(1).stride(1).activationType(ActivationType.IDENTITY)
				.initializer(initializer).pad(0).build(),
			new BatchNormalization.Builder().initializer(initializer).activationType(ActivationType.RELU).build(),
			new Interpolation.Builder().outputHeight(height).outputWidth(width).build()
		};

		branch1[0].setDimensions(dimensions, updaterType);
		branch2[0].setDimensions(dimensions, updaterType);
		branch3[0].setDimensions(dimensions, updaterType);
		branch4[0].setDimensions(dimensions, updaterType);

		for (int i = 1; i < 4; i++) {
			branch1[i].setDimensions(branch1[i - 1].getOutputDimensions(), updaterType);
			branch2[i].setDimensions(branch2[i - 1].getOutputDimensions(), updaterType);
			branch3[i].setDimensions(branch3[i - 1].getOutputDimensions(), updaterType);
			branch4[i].setDimensions(branch4[i - 1].getOutputDimensions(), updaterType);
		}
	}

	public void setMode(Mode mode) {
		for (int i = 0; i < 4; i++) {
			branch1[i].setMode(mode);
			branch2[i].setMode(mode);
			branch3[i].setMode(mode);
			branch4[i].setMode(mode);
		}
	}

	public LayerType getType() {
		return LayerType.PSP;
	}

	public float[] forward(float[] input, int batchSize) {
		float[][] outputs = new float[4][];

		this.batchSize = batchSize;

		outputs[0] = branch1[0].forward(input, batchSize);
		outputs[1] = branch2[0].forward(input, batchSize);
		outputs[2] = branch3[0].forward(input, batchSize);
		outputs[3] = branch4[0].forward(input, batchSize);

		for (int i = 1; i < 4; i++) {
			outputs[0] = branch1[i].forward(outputs[0], batchSize);
			outputs[1] = branch2[i].forward(outputs[1], batchSize);
			outputs[2] = branch3[i].forward(outputs[2], batchSize);
			outputs[3] = branch4[i].forward(outputs[3], batchSize);
		}

		int outputDepth = getOutputDimensions()[2];
		int offset = 0;

		output = new float[height * width * outputDepth * batchSize];
		for (float[] value : outputs) {
			int filterAmount = 512;

			for (int b = 0; b < batchSize; b++) {
				for (int f = 0; f < filterAmount; f++) {
					for (int h = 0; h < height; h++) {
						System.arraycopy(value, width * (h + height * (f + filterAmount * b)), output,
							width * (h + height * ((f + offset) + outputDepth * b)), width);
					}
				}
			}

			offset += filterAmount;
		}

		for (int b = 0; b < batchSize; b++) {
			for (int f = 0; f < depth; f++) {
				for (int h = 0; h < height; h++) {
					System.arraycopy(input, width * (h + height * (f + depth * b)), output,
						width * (h + height * ((f + offset) + outputDepth * b)), width);
				}
			}
		}

		return output;
	}

	public float[] backward(Cost cost, float[] target, boolean calculateDelta) {
		return new float[height * width * depth * batchSize];
	}

	public float[] backward(float[] previousDelta, boolean calculateDelta) {
		return new float[height * width * depth * batchSize];
	}

	public int[] getOutputDimensions() {
		return new int[]{height, width, 2048 + depth};
	}

	public float[][][] getParameters() {
		int length = 0;

		for (int i = 0; i < 4; i++) {
			length += branch1[i].getParameters().length;
			length += branch2[i].getParameters().length;
			length += branch3[i].getParameters().length;
			length += branch4[i].getParameters().length;
		}

		float[][][] parameters = new float[length][][];
		int offset = 0;
		for (Layer layer : branch1) {
			System.arraycopy(layer.getParameters(), 0, parameters, offset, layer.getParameters().length);
			offset += layer.getParameters().length;
		}

		for (Layer layer : branch2) {
			System.arraycopy(layer.getParameters(), 0, parameters, offset, layer.getParameters().length);
			offset += layer.getParameters().length;
		}

		for (Layer layer : branch3) {
			System.arraycopy(layer.getParameters(), 0, parameters, offset, layer.getParameters().length);
			offset += layer.getParameters().length;
		}

		for (Layer layer : branch4) {
			System.arraycopy(layer.getParameters(), 0, parameters, offset, layer.getParameters().length);
			offset += layer.getParameters().length;
		}

		return parameters;
	}

	public void update(int size) {
	}

	/**
	 * Builder for PSP layers.
	 */
	@SuppressWarnings({"unused", "WeakerAccess"})
	public static class Builder {
		private LinkedList<Integer> downsampleSizes = new LinkedList<>();
		private Initializer initializer;

		public Builder downsampleSizes(int... downsampleSizes) {
			for (int value : downsampleSizes)
				this.downsampleSizes.add(value);
			return this;
		}

		public Builder initializer(Initializer initializer) {
			this.initializer = initializer;
			return this;
		}

		public PSP build() {
			return new PSP(downsampleSizes, initializer);
		}
	}
}