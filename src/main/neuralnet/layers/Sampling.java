package main.neuralnet.layers;

import com.aparapi.Kernel;
import com.aparapi.Range;
import main.neuralnet.activations.Identity;
import main.neuralnet.costs.Cost;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Sampling layers downsample inputs. Max pooling does so by taking a max out of a certain area from the input. To back propagation,
 * upsampling is used, where the switches of the max pooling (the selected indices) are remembered, and used to place the deltas at such
 * locations.
 */
public class Sampling implements Layer {
	private DownsampleKernel downsampleKernel;
	private UpsampleKernel upsampleKernel;

	private int inputHeight, inputWidth, filterAmount;
	private int downsampleHeight, downsampleWidth;
	private int downsampleSize, downsampleStride;

	private byte[][] switches;
	private double[][] output, upsampled;

	private Sampling(int downsampleSize, int downsampleStride) {
		this.downsampleSize = downsampleSize;
		this.downsampleStride = downsampleStride;
	}

	Sampling(DataInputStream dis) throws IOException {
		inputHeight = dis.readInt();
		inputWidth = dis.readInt();
		filterAmount = dis.readInt();
		downsampleHeight = dis.readInt();
		downsampleWidth = dis.readInt();
		downsampleSize = dis.readInt();
		downsampleStride = dis.readInt();

		downsampleKernel = new DownsampleKernel(downsampleStride, downsampleSize, inputHeight, downsampleHeight);
		upsampleKernel = new UpsampleKernel(downsampleStride, downsampleHeight, downsampleWidth, downsampleSize, inputHeight, inputWidth);
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

		downsampleKernel = new DownsampleKernel(downsampleStride, downsampleSize, inputHeight, downsampleHeight);
		upsampleKernel = new UpsampleKernel(downsampleStride, downsampleHeight, downsampleWidth, downsampleSize, inputHeight, inputWidth);
	}

	public double[][] forward(double[][] x) {
		switches = new byte[x.length][filterAmount * inputHeight * inputWidth];
		output = new double[x.length][filterAmount * downsampleHeight * downsampleWidth];

		downsampleKernel.init(x, output, switches);
		downsampleKernel.execute(Range.create3D(filterAmount, downsampleHeight, downsampleWidth), x.length);

		return output;
	}

	public double[][] backward(Cost cost, double[][] target) {
		upsampled = new double[target.length][filterAmount * inputHeight * inputWidth];
		double[][] delta = cost.derivative(output, target, new Identity());

		upsampleKernel.init(switches, upsampled, delta);
		upsampleKernel.execute(Range.create3D(filterAmount, downsampleHeight, downsampleWidth), target.length);

		return upsampled;
	}

	public double[][] backward(double[][] previousDelta) {
		upsampled = new double[output.length][filterAmount * inputHeight * inputWidth];

		upsampleKernel.init(switches, upsampled, previousDelta);
		upsampleKernel.execute(Range.create3D(filterAmount, downsampleHeight, downsampleWidth), output.length);

		return upsampled;
	}

	public double[][][] getParameters() {
		return new double[0][][];
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
		return LayerType.SAMPLING;
	}

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

		public Sampling build() {
			if (downsampleSize > 0 && downsampleStride >= 0)
				return new Sampling(downsampleSize, downsampleStride);

			throw new IllegalArgumentException();
		}
	}

	class DownsampleKernel extends Kernel {
		private int downsampleStride, downsampleSize, inputHeight, downsampleHeight;
		private byte[][] switches;
		private double[][] input, downsampled;

		DownsampleKernel(int downsampleStride, int downsampleSize, int inputHeight, int downsampleHeight) {
			this.downsampleStride = downsampleStride;
			this.downsampleSize = downsampleSize;
			this.inputHeight = inputHeight;
			this.downsampleHeight = downsampleHeight;
		}

		void init(double[][] input, double[][] downsampled, byte[][] switches) {
			this.input = input;
			this.downsampled = downsampled;
			this.switches = switches;
		}

		public void run() {
			int b = getPassId();

			int f = getGlobalId(0);
			int i = getGlobalId(1);
			int j = getGlobalId(2);

			int h = i * downsampleStride;
			int w = j * downsampleStride;

			int index = 0;
			double max = Double.NEGATIVE_INFINITY;

			for (int m = 0; m < downsampleSize; m++) {
				for (int n = 0; n < downsampleSize; n++) {
					int outputIndex = (w + n) + inputWidth * ((h + m) + inputHeight * f);
					double value = input[b][outputIndex];

					// finding the max value
					if (value > max) {
						max = value;
						index = outputIndex;
					}
				}
			}

			int downsampleIndex = j + downsampleWidth * (i + downsampleHeight * f);
			switches[b][index] = 1;
			downsampled[b][downsampleIndex] = max;
		}
	}

	class UpsampleKernel extends Kernel {
		private int downsampleStride, downsampleHeight, downsampleWidth, downsampleSize, inputHeight, inputWidth;
		private byte[][] switches;
		private double[][] upsampled, delta;

		UpsampleKernel(int downsampleStride, int downsampleHeight, int downsampleWidth, int downsampleSize, int inputHeight, int inputWidth) {
			this.downsampleStride = downsampleStride;
			this.downsampleHeight = downsampleHeight;
			this.downsampleWidth = downsampleWidth;
			this.downsampleSize = downsampleSize;
			this.inputHeight = inputHeight;
			this.inputWidth = inputWidth;
		}

		void init(byte[][] switches, double[][] upsampled, double[][] delta) {
			this.switches = switches;
			this.upsampled = upsampled;
			this.delta = delta;
		}

		public void run() {
			int b = getPassId();

			int f = getGlobalId(0);
			int i = getGlobalId(1);
			int j = getGlobalId(2);

			int h = i * downsampleStride;
			int w = j * downsampleStride;

			int downsampleIndex = j + downsampleWidth * (i + downsampleHeight * f);

			for (int m = 0; m < downsampleSize; m++) {
				for (int n = 0; n < downsampleSize; n++) {
					int upsampledIndex = (w + n) + inputWidth * ((h + m) + inputHeight * f);

					// changing the dimensions of the delta, and filling the areas that had the max values with the delta values
					if (switches[b][upsampledIndex] == 1) {
						upsampled[b][upsampledIndex] = delta[b][downsampleIndex];
					}
				}
			}
		}
	}
}