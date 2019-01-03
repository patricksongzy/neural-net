package neuralnet.layers;

import neuralnet.activations.ActivationType;
import neuralnet.costs.Cost;
import neuralnet.initializers.Initializer;
import neuralnet.optimizers.UpdaterType;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Objects;

@SuppressWarnings("FieldCanBeLocal")
public class Residual implements Layer {
	private int batchSize;
	private int pad;
	private int stride;
	private int depth, height, width;
	private int filterAmount;
	private int outputDepth;

	private float[] output, residual;

	private Initializer initializer;
	private Layer[] branch1;
	private Layer[] branch2;

	private Residual(int filterAmount, int outputDepth, int pad, int stride, Initializer initializer) {
		Objects.requireNonNull(initializer);

		this.filterAmount = filterAmount;
		this.outputDepth = outputDepth;
		this.pad = pad;
		this.stride = stride;
		this.initializer = initializer;
	}

	Residual(DataInputStream dis, UpdaterType updaterType) throws IOException {
		depth = dis.readInt();
		height = dis.readInt();
		width = dis.readInt();

		filterAmount = dis.readInt();
		outputDepth = dis.readInt();

		branch1 = new Layer[dis.readInt()];
		for (int i = 0; i < branch1.length; i++) {
			branch1[i] = LayerType.fromString(dis, updaterType);
		}

		branch2 = new Layer[dis.readInt()];
		for (int i = 0; i < branch2.length; i++) {
			branch2[i] = LayerType.fromString(dis, updaterType);
		}
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(depth);
		dos.writeInt(height);
		dos.writeInt(width);

		dos.writeInt(filterAmount);
		dos.writeInt(outputDepth);

		dos.writeInt(branch1.length);
		for (Layer layer : branch1) {
			dos.writeUTF(layer.getType().toString());
			layer.export(dos);
		}

		dos.writeInt(branch2.length);
		for (Layer layer : branch2) {
			dos.writeUTF(layer.getType().toString());
			layer.export(dos);
		}
	}

	public void setDimensions(int[] dimensions, UpdaterType updaterType) {
		if (dimensions.length != 3)
			throw new IllegalArgumentException("Invalid input dimensions.");

		this.depth = dimensions[0];
		this.height = dimensions[1];
		this.width = dimensions[2];

		if (outputDepth == -1) {
			outputDepth = depth;
		}

		branch2 = new Layer[]{
			new Convolutional.Builder().pad(0).stride(1).initializer(initializer).filterSize(1).filterAmount(filterAmount)
				.activationType(ActivationType.IDENTITY).build(),
			new BatchNormalization.Builder().initializer(initializer).activationType(ActivationType.RELU).build(),
			new Convolutional.Builder().pad(pad).dilation(pad).stride(stride).initializer(initializer).filterSize(3).filterAmount(filterAmount)
				.activationType(ActivationType.IDENTITY).build(),
			new BatchNormalization.Builder().initializer(initializer).activationType(ActivationType.RELU).build(),
			new Convolutional.Builder().pad(0).stride(1).initializer(initializer).filterSize(1).filterAmount(outputDepth)
				.activationType(ActivationType.IDENTITY).build(),
			new BatchNormalization.Builder().initializer(initializer).build(),
		};

		if (outputDepth != depth) {
			branch1 = new Layer[]{
				new Convolutional.Builder().pad(0).stride(stride).initializer(initializer).filterSize(1).filterAmount(outputDepth)
					.activationType(ActivationType.IDENTITY).build(),
				new BatchNormalization.Builder().initializer(initializer).build(),
			};

			branch1[0].setDimensions(dimensions, updaterType);
			for (int i = 1; i < branch1.length; i++) {
				branch1[i].setDimensions(branch1[i - 1].getOutputDimensions(), updaterType);
			}
		} else {
			branch1 = new Layer[0];
		}

		branch2[0].setDimensions(dimensions, updaterType);
		for (int i = 1; i < branch2.length; i++) {
			branch2[i].setDimensions(branch2[i - 1].getOutputDimensions(), updaterType);
		}
	}

	public void setMode(Mode mode) {
		for (Layer layer : branch1) {
			layer.setMode(mode);
		}

		for (Layer layer : branch2) {
			layer.setMode(mode);
		}
	}

	public LayerType getType() {
		return LayerType.RESIDUAL;
	}

	public float[] forward(float[] input, int batchSize) {
		this.batchSize = batchSize;

		residual = input;
		for (Layer layer : branch1) {
			residual = layer.forward(residual, batchSize);
		}

		output = input;
		for (Layer layer : branch2) {
			output = layer.forward(output, batchSize);
		}

		for (int i = 0; i < output.length; i++) {
			output[i] += residual[i];
		}

		ActivationType.RELU.activation(output, batchSize);

		return output;
	}

	// TODO: implement
	public float[] backward(Cost cost, float[] target, boolean calculateDelta) {
		return new float[batchSize * depth * height * width];
	}

	// TODO: implement
	public float[] backward(float[] previousDelta, boolean calculateDelta) {
		return new float[batchSize * depth * height * width];
	}

	public int[] getOutputDimensions() {
		return branch2[branch2.length - 1].getOutputDimensions();
	}

	public float[][][] getParameters() {
		int length = 0;

		for (Layer layer : branch1) {
			length += layer.getParameters().length;
		}

		for (Layer layer : branch2) {
			length += layer.getParameters().length;
		}

		float[][][] parameters = new float[length][][];
		int offset = 0;
		if (branch1.length > 0) {
			for (Layer layer : branch2) {
				System.arraycopy(layer.getParameters(), 0, parameters, offset, layer.getParameters().length);
				offset += layer.getParameters().length;
			}

			for (Layer layer : branch1) {
				System.arraycopy(layer.getParameters(), 0, parameters, offset, layer.getParameters().length);
				offset += layer.getParameters().length;
			}
		} else {
			for (Layer layer : branch2) {
				System.arraycopy(layer.getParameters(), 0, parameters, offset, layer.getParameters().length);
				offset += layer.getParameters().length;
			}
		}

		return parameters;
	}

	public void update(int length) {
		for (Layer layer : branch1) {
			layer.update(length);
		}

		for (Layer layer : branch2) {
			layer.update(length);
		}
	}

	@SuppressWarnings({"unused", "WeakerAccess"})
	public static class Builder {
		private int filterAmount;
		private int outputDepth;
		private int pad;
		private int stride;
		private Initializer initializer;

		public Builder() {
			outputDepth = -1;
			pad = 1;
			stride = 1;
		}

		public Builder initializer(Initializer initializer) {
			this.initializer = initializer;
			return this;
		}

		public Builder filterAmount(int filterAmount) {
			this.filterAmount = filterAmount;
			return this;
		}

		public Builder outputDepth(int outputDepth) {
			this.outputDepth = outputDepth;
			return this;
		}

		public Builder pad(int pad) {
			this.pad = pad;
			return this;
		}

		public Builder stride(int stride) {
			this.stride = stride;
			return this;
		}

		public Residual build() {
			return new Residual(filterAmount, outputDepth, pad, stride, initializer);
		}
	}
}