package neuralnet.layers;

import neuralnet.activations.Identity;
import neuralnet.costs.Cost;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.stream.IntStream;

/**
 * Pooling layers downsample inputs. Max pooling does so by taking a max out of a certain area from the input. To back propagation,
 * upsampling is used, where the switches of the max pooling (the selected indices) are remembered, and used to place the deltas at such
 * locations.
 */
public class Pooling implements Layer {
	private int inputHeight, inputWidth, filterAmount;
	private int downsampleHeight, downsampleWidth;
	private int downsampleSize, downsampleStride;

	private byte[][] switches;
	private float[][] output;

	private Pooling(int downsampleSize, int downsampleStride) {
		this.downsampleSize = downsampleSize;
		this.downsampleStride = downsampleStride;
	}

	Pooling(DataInputStream dis) throws IOException {
		inputHeight = dis.readInt();
		inputWidth = dis.readInt();
		filterAmount = dis.readInt();
		downsampleHeight = dis.readInt();
		downsampleWidth = dis.readInt();
		downsampleSize = dis.readInt();
		downsampleStride = dis.readInt();
	}

	public void setMode(Mode mode) {
	}

	public void setDimensions(int... dimensions) {
		if (dimensions.length != 3)
			throw new IllegalArgumentException();

		this.inputHeight = dimensions[0];
		this.inputWidth = dimensions[1];
		this.filterAmount = dimensions[2];

		if (filterAmount <= 0 || inputHeight <= 0 || inputWidth <= 0)
			throw new IllegalArgumentException();

		this.downsampleWidth = (inputWidth - downsampleSize) / downsampleStride + 1;
		this.downsampleHeight = (inputHeight - downsampleSize) / downsampleStride + 1;
	}

	public float[][] forward(float[][] input) {
		switches = new byte[input.length][filterAmount * inputHeight * inputWidth];
		output = new float[input.length][filterAmount * downsampleHeight * downsampleWidth];

		IntStream.range(0, input.length).parallel().forEach(b -> {
			for (int f = 0; f < filterAmount; f++) {
				for (int i = 0; i < downsampleHeight; i++) {
					for (int j = 0; j < downsampleWidth; j++) {
						int h = i * downsampleStride;
						int w = j * downsampleStride;

						int index = 0;
						float max = Float.NEGATIVE_INFINITY;

						for (int m = 0; m < downsampleSize; m++) {
							for (int n = 0; n < downsampleSize; n++) {
								int outputIndex = (w + n) + inputWidth * ((h + m) + inputHeight * f);
								float value = input[b][outputIndex];

								// finding the max value
								if (value > max) {
									max = value;
									index = outputIndex;
								}
							}
						}

						int downsampleIndex = j + downsampleWidth * (i + downsampleHeight * f);
						switches[b][index] = 1;
						output[b][downsampleIndex] = max;
					}
				}
			}
		});

		return output;
	}

	public float[] backward(Cost cost, float[][] target) {
		float[] delta = cost.derivative(output, target, new Identity());

		return backward(delta);
	}

	public float[] backward(float[] previousDelta) {
		float[] upsampled = new float[output.length * filterAmount * inputHeight * inputWidth];

		IntStream.range(0, output.length).parallel().forEach(b -> {
			for (int f = 0; f < filterAmount; f++) {
				for (int i = 0; i < downsampleHeight; i++) {
					for (int j = 0; j < downsampleWidth; j++) {
						int h = i * downsampleStride;
						int w = j * downsampleStride;
						int downsampleIndex = j + downsampleWidth * (i + downsampleHeight * (f + filterAmount * b));

						for (int m = 0; m < downsampleSize; m++) {
							for (int n = 0; n < downsampleSize; n++) {
								// changing the dimensions of the delta, and filling the areas that had the max values with the delta
								int upsampledIndex = w + n + inputWidth * h + inputWidth * m + inputWidth * inputHeight * f;

								if (switches[b][upsampledIndex] == 1) {
									upsampled[upsampledIndex + inputWidth * inputHeight * filterAmount * b]
										= previousDelta[downsampleIndex];
								}
							}
						}
					}
				}
			}
		});

		return upsampled;
	}

	public float[][][] getParameters() {
		return new float[0][][];
	}

	public int[] getOutputDimensions() {
		return new int[]{downsampleHeight, downsampleWidth, filterAmount};
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(inputHeight);
		dos.writeInt(inputWidth);
		dos.writeInt(filterAmount);
		dos.writeInt(downsampleHeight);
		dos.writeInt(downsampleWidth);
		dos.writeInt(downsampleSize);
		dos.writeInt(downsampleStride);
	}

	public LayerType getType() {
		return LayerType.POOLING;
	}

	@SuppressWarnings({"unused", "WeakerAccess"})
	public static class Builder {
		private int downsampleSize, downsampleStride;

		public Builder downsampleSize(int downsampleSize) {
			this.downsampleSize = downsampleSize;
			return this;
		}

		public Builder downsampleStride(int downsampleStride) {
			this.downsampleStride = downsampleStride;
			return this;
		}

		public Pooling build() {
			if (downsampleSize > 0 && downsampleStride >= 0)
				return new Pooling(downsampleSize, downsampleStride);

			throw new IllegalArgumentException();
		}
	}
}