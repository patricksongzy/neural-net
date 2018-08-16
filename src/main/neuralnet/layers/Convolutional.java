package main.neuralnet.layers;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Range;
import main.neuralnet.activations.Activation;
import main.neuralnet.activations.ActivationType;
import main.neuralnet.costs.Cost;
import main.neuralnet.initializers.HeInitialization;
import main.neuralnet.initializers.Initializer;
import main.neuralnet.optimizers.Updater;
import main.neuralnet.optimizers.UpdaterType;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.stream.IntStream;

/**
 * The convolutional layer revolves around convolutions in image processing. Using a similar method, filters are convolved around an image
 * (the input) and an output is created. A bias term is added, then the output is activated Deltas are calculated a layer ahead
 * (the next layer that will be updated), so that upsampling layers are compatible
 */
public class Convolutional implements Layer {
	private Mode mode = Mode.TRAIN;

	// the filter amount is the output depth
	// the filter size is the size of the filters
	private int filterAmount, filterSize;
	private ConvolutionKernel convolutionKernel;
	private GradientKernel gradientKernel;
	private DeltaKernel deltaKernel;

	private UpdaterType updaterType;
	private Initializer initializer;
	private Activation activation;
	private Updater[] filterUpdaters;
	private Updater[] biasUpdaters;

	// the pad is the amount an input is padded
	// the pad width and pad height are the heights and widths of the input after it is padded
	private int padHeight, padWidth, pad;

	// the stride is how much a filter skips each time
	// the output width and output height are the heights and widths of the output after convolution
	private int outputHeight, outputWidth, stride;

	// the input width and input height are the heights and widths of the inputs provided to the layer
	private int inputHeight, inputWidth, depth;

	private double[] filters, biases;
	private double[] gradient, biasGradient;
	private double[][] input, output;

	private Convolutional(int pad, int stride, int filterAmount, int filterSize, Initializer initializer, UpdaterType updaterType,
						  ActivationType activationType) {
		this.pad = pad;
		this.stride = stride;
		this.filterAmount = filterAmount;
		this.filterSize = filterSize;

		this.updaterType = updaterType;
		this.initializer = initializer;

		activation = activationType.create();
	}

	/**
	 * Initializes a convolutional layer from a file.
	 *
	 * @param dis the input stream
	 */
	Convolutional(DataInputStream dis) throws IOException {
		inputHeight = dis.readInt();
		inputWidth = dis.readInt();
		depth = dis.readInt();
		padHeight = dis.readInt();
		padWidth = dis.readInt();
		pad = dis.readInt();
		outputHeight = dis.readInt();
		outputWidth = dis.readInt();
		stride = dis.readInt();
		filterAmount = dis.readInt();
		filterSize = dis.readInt();

		filterUpdaters = new Updater[filterAmount * depth * filterSize * filterSize];
		filters = new double[filterAmount * depth * filterSize * filterSize];

		biasUpdaters = new Updater[filterAmount];
		biases = new double[filterAmount];

		activation = ActivationType.fromString(dis).create();
		updaterType = UpdaterType.fromString(dis);

		for (int f = 0; f < filterAmount; f++) {
			biases[f] = dis.readDouble();
			biasUpdaters[f] = updaterType.create(dis);

			for (int k = 0; k < depth; k++) {
				for (int m = 0; m < filterSize; m++) {
					for (int n = 0; n < filterSize; n++) {
						int index = n + filterSize * (m + filterSize * (k + depth * f));

						filters[index] = dis.readDouble();
						filterUpdaters[index] = updaterType.create(dis);
					}
				}
			}
		}

		convolutionKernel = new ConvolutionKernel(padWidth, padHeight, depth, stride, outputWidth, outputHeight, filterSize, filterAmount);
		gradientKernel = new GradientKernel(padWidth, padHeight, depth, stride, outputWidth, outputHeight, filterSize);
		deltaKernel = new DeltaKernel(padWidth, padHeight, depth, outputWidth, outputHeight, filterSize, filterAmount, stride);
	}

	public void setDimensions(int... dimensions) {
		if (dimensions.length != 3)
			throw new IllegalArgumentException();

		this.inputHeight = dimensions[0];
		this.inputWidth = dimensions[1];
		this.depth = dimensions[2];

		if (inputHeight <= 0 || inputWidth <= 0 || depth <= 0)
			throw new IllegalArgumentException();

		// calculating the post padding dimensions
		padHeight = inputHeight + 2 * pad;
		padWidth = inputWidth + 2 * pad;

		// calculating the post convolution dimensions
		this.outputHeight = (padHeight - filterSize) / stride + 1;
		this.outputWidth = (padWidth - filterSize) / stride + 1;

		filterUpdaters = new Updater[filterAmount * depth * filterSize * filterSize];
		filters = new double[filterAmount * depth * filterSize * filterSize];

		biasUpdaters = new Updater[filterAmount];
		biases = new double[filterAmount];

		int inputSize = depth * filterSize * filterSize;

		IntStream.range(0, filterAmount).parallel().forEach(f -> {
			biasUpdaters[f] = updaterType.create();
			for (int k = 0; k < depth; k++) {
				for (int m = 0; m < filterSize; m++) {
					for (int n = 0; n < filterSize; n++) {
						int index = n + filterSize * (m + filterSize * (k + depth * f));

						filters[index] = initializer.initialize(inputSize);
						filterUpdaters[index] = updaterType.create();
					}
				}
			}
		});

		convolutionKernel = new ConvolutionKernel(padWidth, padHeight, depth, stride, outputWidth, outputHeight, filterSize, filterAmount);
		gradientKernel = new GradientKernel(padWidth, padHeight, depth, stride, outputWidth, outputHeight, filterSize);
		deltaKernel = new DeltaKernel(padWidth, padHeight, depth, outputWidth, outputHeight, filterSize, filterAmount, stride);
	}

	public void setMode(Mode mode) {
		this.mode = mode;
	}

	/**
	 * Pads the input.
	 *
	 * @param x the input
	 * @return the padded input
	 */
	public double[][] pad(double[][] x) {
		if (pad > 0) {
			// creating an array, with the dimensions of the padded input
			double[][] out = new double[x.length][depth * padHeight * padWidth];

			// padding the array
			for (int i = 0; i < x.length; i++) {
				int j = 0;

				for (int d = 0; d < depth; d++) {
					int offset = 0;
					int index = 0;

					offset += padWidth * pad;

					for (int h = 0; h < inputHeight; h++) {
						offset += pad;

						for (int w = 0; w < inputWidth; w++) {
							out[i][d * padHeight * padWidth + index++ + offset] = x[i][j++];
						}

						offset += pad;
					}
				}
			}

			return out;
		}

		return x;
	}

	public double[][] forward(double[][] x) {
		input = pad(x);

		output = new double[x.length][filterAmount * outputHeight * outputWidth];

		// multiplying by filters
		convolutionKernel.init(filters, biases, input, output);
		convolutionKernel.execute(Range.create3D(filterAmount * x.length, outputHeight, outputWidth));
		// activation
		activation.activation(output);

		return output;
	}

	public double[][] backward(Cost cost, double[][] target) {
		// back propagation on the Convolutional layers are calculated a layer ahead
		double[][] previousDelta = cost.derivative(output, target, activation);

		double[][] delta = new double[output.length][depth * padHeight * padWidth];
		biasGradient = new double[filterAmount];

		gradient = new double[filterAmount * depth * filterSize * filterSize];

		gradientKernel.init(previousDelta, biasGradient, gradient, input);
		gradientKernel.execute(Range.create2D(filterAmount, output.length));

		// calculating gradient
		if (mode == Mode.TRAIN) {
			// updating parameters
			update(biasGradient, gradient);
		}

		// calculating the delta
		deltaKernel.init(delta, previousDelta, filters);
		deltaKernel.execute(Range.create3D(output.length * depth, padHeight, padWidth));

		return delta;
	}

	public double[][] backward(double[][] previousDelta) {
		// back propagation on the Convolutional layers are calculated a layer ahead
		double[][] delta = new double[output.length][depth * padHeight * padWidth];
		biasGradient = new double[filterAmount];

		// derivative
		output = activation.derivative(output);

		// since deltas are calculated a layer ahead
		for (int b = 0; b < previousDelta.length; b++) {
			for (int i = 0; i < output[0].length; i++) {
				previousDelta[b][i] *= output[b][i];
			}
		}

		gradient = new double[filterAmount * depth * filterSize * filterSize];

		gradientKernel.init(previousDelta, biasGradient, gradient, input);
		gradientKernel.execute(Range.create2D(filterAmount, output.length));

		// calculating gradient
		if (mode == Mode.TRAIN) {
			// updating parameters
			update(biasGradient, gradient);
		}

		// calculating delta
		deltaKernel.init(delta, previousDelta, filters);
		deltaKernel.execute(Range.create3D(output.length * depth, padHeight, padWidth));

		return delta;
	}

	/**
	 * Update the parameters given gradients.
	 *
	 * @param delta the bias gradient
	 * @param gradient the weight gradient
	 */
	private void update(double[] delta, double[] gradient) {
		IntStream.range(0, filterAmount).parallel().forEach(f -> {
			biases[f] += biasUpdaters[f].update(delta[f] / output.length);

			for (int k = 0; k < depth; k++) {
				for (int m = 0; m < filterSize; m++) {
					for (int n = 0; n < filterSize; n++) {
						int filterIndex = n + filterSize * (m + filterSize * (k + depth * f));

						filters[filterIndex] += filterUpdaters[filterIndex].update(gradient[filterIndex] / output.length);
					}
				}
			}
		});
	}

	public double[][][] getParameters() {
		return new double[][][] {{filters, gradient}, {biases, biasGradient}};
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(inputHeight);
		dos.writeInt(inputWidth);
		dos.writeInt(depth);
		dos.writeInt(padHeight);
		dos.writeInt(padWidth);
		dos.writeInt(pad);
		dos.writeInt(outputHeight);
		dos.writeInt(outputWidth);
		dos.writeInt(stride);
		dos.writeInt(filterAmount);
		dos.writeInt(filterSize);

		activation.getType().export(dos);
		updaterType.export(dos);

		for (int f = 0; f < filterAmount; f++) {
			dos.writeDouble(biases[f]);
			biasUpdaters[f].export(dos);

			for (int k = 0; k < depth; k++) {
				for (int m = 0; m < filterSize; m++) {
					for (int n = 0; n < filterSize; n++) {
						int index = n + filterSize * (m + filterSize * (k + depth * f));

						dos.writeDouble(filters[index]);
						filterUpdaters[index].export(dos);
					}
				}
			}
		}
	}

	public int[] getOutputDimensions() {
		return new int[]{outputHeight, outputWidth, filterAmount};
	}

	public LayerType getType() {
		return LayerType.CONVOLUTIONAL;
	}

	/**
	 * Builder for Convolutional layers.
	 */
	@SuppressWarnings("unused")
	public static class Builder {
		private int pad;
		private int stride;
		private int filterAmount, filterSize;
		private Initializer initializer;
		private UpdaterType updaterType;
		private ActivationType activationType;

		public Builder() {
			initializer = new HeInitialization();
			updaterType = UpdaterType.ADAM;
			activationType = ActivationType.RELU;
		}

		/**
		 * The pad is the amount of zeroes that are padded around an input. Padding is used to preserve edge features during convolutions,
		 * by allowing filters to traverse such areas.
		 *
		 * @param pad the pad
		 */
		public Builder pad(int pad) {
			this.pad = pad;
			return this;
		}

		/**
		 * The stride is the amount a filter moves by, each time, when performing convolution. Strides can be used to downsample images,
		 * instead
		 * of using pooling
		 *
		 * @param stride the stride
		 */
		public Builder stride(int stride) {
			this.stride = stride;
			return this;
		}

		/**
		 * The filter amount is the amount of kernels that can be learned and applied during convolution.
		 *
		 * @param filterAmount the filter amount
		 */
		public Builder filterAmount(int filterAmount) {
			this.filterAmount = filterAmount;
			return this;
		}

		/**
		 * The filter size is the size of filters. Larger filter sizes decrease output dimensions faster.
		 *
		 * @param filterSize the filter size
		 */
		public Builder filterSize(int filterSize) {
			this.filterSize = filterSize;
			return this;
		}

		/**
		 * The initializer initializes weights.
		 *
		 * @param initializer the initializer
		 */
		public Builder initializer(Initializer initializer) {
			if (initializer != null)
				this.initializer = initializer;
			else
				throw new IllegalArgumentException();

			return this;
		}

		/**
		 * The updater updates parameters.
		 *
		 * @param updaterType the updater type
		 */
		public Builder updaterType(UpdaterType updaterType) {
			if (updaterType != null)
				this.updaterType = updaterType;
			else
				throw new IllegalArgumentException();

			return this;
		}

		/**
		 * The activation simulates a neuron firing.
		 *
		 * @param activationType the activation type
		 */
		public Builder activationType(ActivationType activationType) {
			if (activationType != null)
				this.activationType = activationType;
			else
				throw new IllegalArgumentException();

			return this;
		}

		public Convolutional build() {
			if (pad >= 0 && stride > 0 && filterAmount > 0 && filterSize > 0)
				return new Convolutional(pad, stride, filterAmount, filterSize, initializer, updaterType, activationType);

			throw new IllegalArgumentException();
		}
	}

	/**
	 * The ConvolutionKernel does convolution with the input and the filters.
	 */
	class ConvolutionKernel extends Kernel {
		private int padWidth, padHeight, depth, stride, outputWidth, outputHeight, filterSize, filterAmount;
		private double[] filters, biases;
		private double[][] input, preActivated;

		ConvolutionKernel(int padWidth, int padHeight, int depth, int stride, int outputWidth, int outputHeight, int filterSize, int
				filterAmount) {
			this.padWidth = padWidth;
			this.padHeight = padHeight;
			this.depth = depth;
			this.stride = stride;
			this.filterSize = filterSize;
			this.filterAmount = filterAmount;
			this.outputWidth = outputWidth;
			this.outputHeight = outputHeight;
		}

		void init(double[] filters, double[] biases, double[][] input, double[][] preActivated) {
			this.filters = filters;
			this.biases = biases;
			this.input = input;
			this.preActivated = preActivated;
		}

		public void run() {
			int index = getGlobalId(0);
			int f = index % filterAmount;
			int b = index / filterAmount;
			int i = getGlobalId(1);
			int j = getGlobalId(2);

			// performing strides
			int h = i * stride;
			int w = j * stride;

			// convoluted value is the sum of the filters multiplied against the inputs at a certain position
			double conv = 0;

			for (int k = 0; k < depth; k++) {
				for (int m = 0; m < filterSize; m++) {
					for (int n = 0; n < filterSize; n++) {
						int filterIndex = n + filterSize * (m + filterSize * (k + depth * f));
						int inputIndex = (w + n) + padWidth * ((h + m) + padHeight * k);

						conv += filters[filterIndex] * input[b][inputIndex];
					}
				}
			}

			// adding biases to shift the activation function
			int activatedIndex = j + outputWidth * (i + outputHeight * f);
			preActivated[b][activatedIndex] = (conv + biases[f]);
		}
	}

	/**
	 * The GradientKernel calculates the gradients to update parameters on.
	 */
	class GradientKernel extends Kernel {
		private int padWidth, padHeight, depth, stride, outputWidth, outputHeight, filterSize;
		private double[] gradient, biasGradient;

		// the previous delta is really the delta of the current layer, calculated from the previous layer in back propagation
		private double[][] previousDelta, input;

		GradientKernel(int padWidth, int padHeight, int depth, int stride, int outputWidth, int outputHeight, int filterSize) {
			this.padWidth = padWidth;
			this.padHeight = padHeight;
			this.depth = depth;
			this.stride = stride;
			this.outputWidth = outputWidth;
			this.outputHeight = outputHeight;
			this.filterSize = filterSize;
		}

		void init(double[][] previousDelta, double[] biasGradient, double[] gradient, double[][] input) {
			this.previousDelta = previousDelta;
			this.biasGradient = biasGradient;
			this.gradient = gradient;
			this.input = input;
		}

		public void run() {
			int f = getGlobalId(0);
			int b = getGlobalId(1);

			for (int i = 0, h = 0; i < outputHeight; i++, h += stride) {
				for (int j = 0, w = 0; j < outputWidth; j++, w += stride) {
					int index = j + outputWidth * (i + outputHeight * f);

					// the bias gradient is the delta, since biases are just added to the output
					double d = previousDelta[b][index];
					biasGradient[f] += d;

					for (int k = 0; k < depth; k++) {
						for (int m = 0; m < filterSize; m++) {
							for (int n = 0; n < filterSize; n++) {
								int gradientIndex = n + filterSize * (m + filterSize * (k + depth * f));
								int inputIndex = (w + n) + padWidth * ((h + m) + padHeight * k);

								// the gradient is the delta multiplied against the input, since the filters are multiplied with the input
								gradient[gradientIndex] += d * input[b][inputIndex];
							}
						}
					}
				}
			}
		}
	}

	/**
	 * The DeltaKernel calculates the delta of the next layer in back propagation, given the current delta, calculated from the previous
	 * layer.
	 */
	class DeltaKernel extends Kernel {
		private int padWidth, padHeight, depth, outputWidth, outputHeight, filterSize, filterAmount, stride;
		private double[] filters;
		private double[][] delta, previousDelta;

		DeltaKernel(int padWidth, int padHeight, int depth, int outputWidth, int outputHeight, int filterSize, int filterAmount, int
				stride) {
			this.padWidth = padWidth;
			this.padHeight = padHeight;
			this.depth = depth;
			this.outputWidth = outputWidth;
			this.outputHeight = outputHeight;
			this.filterSize = filterSize;
			this.filterAmount = filterAmount;
			this.stride = stride;
		}

		void init(double[][] delta, double[][] previousDelta, double[] filters) {
			this.delta = delta;
			this.previousDelta = previousDelta;
			this.filters = filters;
		}

		public void run() {
			int index = getGlobalId(0);
			int b = index / depth;
			int k = index % depth;

			int i = getGlobalId(1);
			int j = getGlobalId(2);

			int h = i * stride;
			int w = j * stride;

			int deltaIndex = j + padWidth * (i + padHeight * k);

			for (int f = 0; f < filterAmount; f++) {
				for (int m = 0; m < filterSize; m++) {
					for (int n = 0; n < filterSize; n++) {
						if ((w - n) < outputWidth && (h - m) < outputHeight && (w - n) >= 0 && (h - m) >= 0) {
							int upsampledIndex = (w - n) + outputWidth * ((h - m) + outputHeight * f);
							int filterIndex = n + filterSize * (m + filterSize * (k + depth * f));

							// same as forward propagation, except the activation derivative is multiplied later
							delta[b][deltaIndex] += previousDelta[b][upsampledIndex] * filters[filterIndex];
						}
					}
				}
			}
		}
	}
}