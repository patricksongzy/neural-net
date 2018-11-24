package neuralnet.layers;

import neuralnet.costs.Cost;
import neuralnet.optimizers.UpdaterType;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Pooling layers downsample inputs. Max pooling does so by taking a max out of a certain area from the input. To back propagation,
 * upsampling is used, where the switches of the max pooling (the selected indices) are remembered, and used to place the deltas at such
 * locations.
 */
public class Pooling implements Layer {
	private final Mode mode;

	private int batchSize;
	private int pad;
	private int inputHeight, inputWidth, depth;
	private int padHeight, padWidth;
	private int downsampleHeight, downsampleWidth;
	private int downsampleSize, downsampleStride;
	private boolean[] switches;
	private float[] output;

	private Pooling(Mode mode, int downsampleSize, int downsampleStride, int pad) {
		if (downsampleSize <= 0 && downsampleStride <= 0)
			throw new IllegalArgumentException("Invalid pooling dimensions.");
		if (pad < 0)
			throw new IllegalArgumentException("Invalid pad dimensions.");
		if (mode == null)
			throw new IllegalArgumentException("Values cannot be null.");

		this.mode = mode;
		this.downsampleSize = downsampleSize;
		this.downsampleStride = downsampleStride;
		this.pad = pad;
	}

	Pooling(DataInputStream dis) throws IOException {
		System.out.println("Type: " + getType());
		mode = Mode.valueOf(dis.readUTF());
		inputHeight = dis.readInt();
		inputWidth = dis.readInt();
		pad = dis.readInt();
		padHeight = dis.readInt();
		padWidth = dis.readInt();
		depth = dis.readInt();
		System.out.println(String.format("Input Dimensions (h x w x d): %d x %d x %d", inputHeight, inputWidth, depth));
		System.out.println(String.format("Input Dimensions (h x w x d): %d x %d x %d", padHeight, padWidth, depth));

		downsampleHeight = dis.readInt();
		downsampleWidth = dis.readInt();
		System.out.println(String.format("Output Dimensions (h x w x d): %d x %d x %d", downsampleHeight, downsampleWidth, depth));

		downsampleSize = dis.readInt();
		downsampleStride = dis.readInt();
		System.out.println("Downsample Size: " + downsampleSize);
		System.out.println("Downsample Stride: " + downsampleStride);
	}

	public void setMode(Layer.Mode mode) {
	}

	public void setDimensions(int[] dimensions, UpdaterType updaterType) {
		System.out.println("Type: " + getType());

		if (dimensions.length < 3)
			throw new IllegalArgumentException("Invalid input dimensions.");

		this.inputHeight = dimensions[0];
		this.inputWidth = dimensions[1];
		this.depth = dimensions[2];

		padHeight = inputHeight + 2 * pad;
		padWidth = inputWidth + 2 * pad;

		this.downsampleWidth = (int) Math.ceil((padWidth - downsampleSize) / (float) downsampleStride) + 1;
		this.downsampleHeight = (int) Math.ceil((padHeight - downsampleSize) / (float) downsampleStride) + 1;

		if ((downsampleHeight - 1) * downsampleStride >= inputHeight * padHeight)
			downsampleHeight--;
		if ((downsampleWidth - 1) * downsampleStride >= inputWidth * padWidth)
			downsampleWidth--;

		System.out.println(String.format("Input Dimensions (h x w x d): %d x %d x %d", inputHeight, inputWidth, depth));
		System.out.println(String.format("Pad Dimensions (h x w x d): %d x %d x %d", padHeight, padWidth, depth));
		System.out.println(String.format("Output Dimensions (h x w x d): %d x %d x %d", downsampleHeight, downsampleWidth, depth));

		if ((padHeight - downsampleSize) % downsampleStride != 0 || (padWidth - downsampleSize) % downsampleStride != 0) {
			System.err.println("WARNING: Stride and filter sizes do not match");

			for (StackTraceElement element : Thread.currentThread().getStackTrace()) {
				System.err.println(element);
			}
		}

		if (depth <= 0 || inputHeight <= 0 || inputWidth <= 0)
			throw new IllegalArgumentException("Invalid input dimensions.");

		if (downsampleHeight <= 0 || downsampleWidth <= 0)
			throw new IllegalArgumentException("Invalid input dimensions.");
	}

	/**
	 * Pads the input.
	 *
	 * @param input the input
	 * @return the padded input
	 */
	private float[] pad(float[] input, int batchSize) {
		if (batchSize <= 0)
			throw new IllegalArgumentException("Batch size must be > 0.");

		return Convolutional.pad(input, batchSize, pad, depth, padHeight, padWidth, inputHeight, inputWidth);
	}

	public float[] forward(float[] x, int batchSize) {
		this.batchSize = batchSize;
		float[] input = pad(x, batchSize);

		switches = new boolean[batchSize * depth * padHeight * padWidth];
		output = new float[batchSize * depth * downsampleHeight * downsampleWidth];

		// TODO: Quick workaround to let padding work for invalid dimensions. These changes are still not reflected in backpropagation.
		int roundWidth = (padWidth - downsampleSize) % downsampleStride != 0 ? 1 : 0;
		int roundHeight = (padHeight - downsampleSize) % downsampleStride != 0 ? 1 : 0;

		for (int b = 0; b < batchSize; b++) {
			for (int f = 0; f < depth; f++) {
				for (int i = -roundHeight; i < downsampleHeight - roundHeight; i++) {
					for (int j = -roundWidth; j < downsampleWidth - roundWidth; j++) {
						int h = i * downsampleStride;
						int w = j * downsampleStride;

						int downsampleIndex = (j + roundWidth) + downsampleWidth * ((i + roundHeight) + downsampleHeight * (f + depth * b));
						if (mode == Mode.MAX) {
							int index = 0;
							float max = Float.NEGATIVE_INFINITY;

							for (int m = 0; m < downsampleSize; m++) {
								for (int n = 0; n < downsampleSize; n++) {
									if (w + n >= 0 && h + m >= 0) {
										int outputIndex = (w + n) + padWidth * ((h + m) + padHeight * (f + depth * b));
										float value = input[outputIndex];

										// finding the max value
										if (value > max) {
											max = value;
											index = outputIndex;
										}
									}
								}
							}

							switches[index] = true;
							output[downsampleIndex] = max;
						} else {
							float sum = 0;
							for (int m = 0; m < downsampleSize; m++) {
								for (int n = 0; n < downsampleSize; n++) {
									if (w + n >= 0 && h + m >= 0) {
										int outputIndex = (w + n) + padWidth * ((h + m) + padHeight * (f + depth * b));
										sum += input[outputIndex];
									}
								}
							}

							output[downsampleIndex] = sum / (downsampleSize * downsampleSize);
						}
					}
				}
			}
		}

		return output;
	}

	public float[] backward(Cost cost, float[] target, boolean calculateDelta) {
		return backward(cost.derivative(output, target, batchSize), calculateDelta);
	}

	public float[] backward(float[] previousDelta, boolean calculateDelta) {
		if (calculateDelta) {
			float[] upsampled = new float[batchSize * depth * padHeight * padWidth];

			int roundWidth = (padWidth - downsampleSize) % downsampleStride != 0 ? 1 : 0;
			int roundHeight = (padHeight - downsampleSize) % downsampleStride != 0 ? 1 : 0;

			for (int b = 0; b < batchSize; b++) {
				for (int f = 0; f < depth; f++) {
					for (int i = -roundHeight, h = 0; i < downsampleHeight - roundHeight; i++, h += downsampleStride) {
						for (int j = -roundWidth, w = 0; j < downsampleWidth - roundWidth; j++, w += downsampleStride) {
							int downsampleIndex =
								(j + roundWidth) + downsampleWidth * ((i + roundHeight) + downsampleHeight * (f + depth * b));

							if (mode == Mode.MAX) {
								for (int m = 0; m < downsampleSize; m++) {
									for (int n = 0; n < downsampleSize; n++) {
										// filling input max locations with deltas
										int index = (w + n) + padWidth * ((h + m) + padHeight * (f + depth * b));
										if (switches[index]) {
											upsampled[index] = previousDelta[downsampleIndex];
										}
									}
								}
							} else {
								for (int m = 0; m < downsampleSize; m++) {
									for (int n = 0; n < downsampleSize; n++) {
										int index = (w + n) + padWidth * ((h + m) + padHeight * (f + depth * b));
										upsampled[index] = previousDelta[downsampleIndex] / (downsampleSize * downsampleSize);
									}
								}
							}
						}
					}
				}
			}

			return removePad(upsampled, batchSize);
		}

		return null;
	}

	private float[] removePad(float[] input, int batchSize) {
		if (batchSize <= 0)
			throw new IllegalArgumentException("Batch size must be > 0.");

		return Convolutional.removePad(input, batchSize, pad, depth, padWidth, inputHeight, inputWidth);
	}

	public float[][][] getParameters() {
		return new float[0][][];
	}

	public void update() {
	}

	public int[] getOutputDimensions() {
		return new int[]{downsampleHeight, downsampleWidth, depth};
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeUTF(mode.toString());
		dos.writeInt(inputHeight);
		dos.writeInt(inputWidth);
		dos.writeInt(pad);
		dos.writeInt(padHeight);
		dos.writeInt(padWidth);
		dos.writeInt(depth);
		dos.writeInt(downsampleHeight);
		dos.writeInt(downsampleWidth);
		dos.writeInt(downsampleSize);
		dos.writeInt(downsampleStride);
	}

	public LayerType getType() {
		return LayerType.POOLING;
	}

	public enum Mode {
		MAX, AVERAGE
	}

	@SuppressWarnings({"unused", "WeakerAccess"})
	public static class Builder {
		private Mode mode;
		private int pad;
		private int downsampleSize, downsampleStride;

		public Builder() {
			mode = Mode.MAX;
		}

		public Builder mode(Mode mode) {
			this.mode = mode;
			return this;
		}

		public Builder downsampleSize(int downsampleSize) {
			this.downsampleSize = downsampleSize;
			return this;
		}

		public Builder downsampleStride(int downsampleStride) {
			this.downsampleStride = downsampleStride;
			return this;
		}

		public Builder pad(int pad) {
			this.pad = pad;
			return this;
		}

		public Pooling build() {
			return new Pooling(mode, downsampleSize, downsampleStride, pad);
		}
	}
}