package neuralnet.layers;

import neuralnet.GPU;
import neuralnet.activations.Activation;
import neuralnet.activations.ActivationType;
import neuralnet.costs.Cost;
import neuralnet.initializers.Initializer;
import neuralnet.optimizers.Updater;
import neuralnet.optimizers.UpdaterType;
import org.jocl.CL;
import org.jocl.blast.CLBlastTranspose;
import org.jocl.cl_mem;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.IntStream;

/**
 * The convolutional layer revolves around convolutions in image processing. Using a similar method, filters are convolved around an image
 * (the input) and an output is created. A bias term is added, then the output is activated Deltas are calculated a layer ahead
 * (the next layer that will be updated), so that upsampling layers are compatible
 */
public class Convolutional implements Layer {
	private Mode mode;
	private int batchSize;

	// the filter amount is the output depth
	// the filter size is the size of the filters
	private int filterAmount, filterSize;
	private int dilation, dilatedSize;

	private Initializer initializer;
	private Activation activation;
	private Updater filterUpdater;
	private Updater biasUpdater;

	// the pad is the amount an input is padded
	// the pad width and pad height are the heights and widths of the input after it is padded
	private int padHeight, padWidth, pad;

	// the stride is how much a filter skips each time
	// the output width and output height are the heights and widths of the output after convolution
	private int outputHeight, outputWidth, stride;

	// the input width and input height are the heights and widths of the inputs provided to the layer
	private int depth, inputHeight, inputWidth;

	private float[] filters, biases;
	private float[] gradient, biasGradient;
	private float[] output;
	private cl_mem inputBuffer;

	private Convolutional(int pad, int stride, int filterAmount, int filterSize, int dilation, Initializer initializer,
						  ActivationType activationType) {
		Objects.requireNonNull(initializer);
		Objects.requireNonNull(activationType);
		if (pad < 0)
			throw new IllegalArgumentException("Pad must be > 0");
		if (stride <= 0 || filterAmount <= 0 || filterSize <= 0)
			throw new IllegalArgumentException("Stride, filter amount and filter size must be > 0");

		this.dilation = dilation;

		this.pad = pad;
		this.stride = stride;
		this.filterAmount = filterAmount;
		this.filterSize = filterSize;
		dilatedSize = (filterSize - 1) * (dilation - 1) + filterSize;

		this.initializer = initializer;

		activation = activationType;
	}

	/**
	 * Initializes a convolutional layer from a file.
	 *
	 * @param dis the input stream
	 */
	Convolutional(DataInputStream dis, UpdaterType updaterType) throws IOException {
		depth = dis.readInt();
		inputHeight = dis.readInt();
		inputWidth = dis.readInt();
		padHeight = dis.readInt();
		padWidth = dis.readInt();
		pad = dis.readInt();
		outputHeight = dis.readInt();
		outputWidth = dis.readInt();
		stride = dis.readInt();
		filterAmount = dis.readInt();
		filterSize = dis.readInt();
		dilation = dis.readInt();
		dilatedSize = dis.readInt();

		activation = Activation.fromString(dis);
		filterUpdater = updaterType.create(dis);
		filters = new float[filterAmount * depth * filterSize * filterSize];

		biasUpdater = updaterType.create(dis);
		biases = new float[filterAmount];

		for (int f = 0; f < filterAmount; f++) {
			biases[f] = dis.readFloat();

			for (int k = 0; k < depth; k++) {
				for (int m = 0; m < filterSize; m++) {
					for (int n = 0; n < filterSize; n++) {
						int index = n + filterSize * (m + filterSize * (k + depth * f));

						filters[index] = dis.readFloat();
					}
				}
			}
		}
	}

	static float[] pad(float[] input, int batchSize, int pad, int depth, int padHeight, int padWidth, int inputHeight, int inputWidth) {
		if (pad > 0) {
			// creating an array, with the dimensions of the padded input
			float[] output = new float[batchSize * depth * padHeight * padWidth];

			// padding the array
			int position = 0;
			for (int b = 0; b < batchSize; b++) {
				for (int j = 0; j < depth; j++) {
					position += pad * padWidth;

					for (int k = 0; k < inputWidth * inputHeight; k += inputWidth) {
						System.arraycopy(input, k + (inputWidth * inputHeight) * (j + depth * b), output, pad + position, inputWidth);
						position += padWidth;
					}

					position += pad * padWidth;
				}
			}

			return output;
		}

		return input;
	}

	static float[] removePad(float[] input, int batchSize, int pad, int depth, int padWidth, int inputHeight, int inputWidth) {
		if (pad > 0) {
			// creating an array, with the dimensions of the padded input
			float[] output = new float[batchSize * depth * inputHeight * inputWidth];

			// padding the array
			int position = 0;
			for (int b = 0; b < batchSize; b++) {
				for (int i = 0; i < depth; i++) {
					position += pad * padWidth;

					for (int j = 0; j < inputWidth * inputHeight; j += inputWidth) {
						System.arraycopy(input, pad + position, output, j + (inputWidth * inputHeight) * (i + depth * b), inputWidth);
						position += padWidth;
					}

					position += pad * padWidth;
				}
			}

			return output;
		}

		return input;
	}

	private static float[] dilate(float[] input, int dilation, int amount, int depth, int height, int width) {
		if (dilation > 1) {
			int dilatedHeight = (height - 1) * (dilation - 1) + height;
			int dilatedWidth = (width - 1) * (dilation - 1) + width;
			float[] output = new float[amount * depth * dilatedHeight * dilatedWidth];

			for (int b = 0; b < amount; b++) {
				for (int i = 0; i < depth; i++) {
					for (int j = 0, h = 0; j < height; j++, h += dilation) {
						for (int k = 0, w = 0; k < width; k++, w += dilation) {
							output[w + dilatedWidth * (h + dilatedHeight * (i + depth * b))] =
								input[k + width * (j + height * (i + depth * b))];
						}
					}
				}
			}

			return output;
		}

		return input;
	}

	public void setMode(Mode mode) {
		this.mode = mode;
	}

	public void setDimensions(int[] dimensions, UpdaterType updaterType) {
		if (dimensions.length < 3)
			throw new IllegalArgumentException();

		this.depth = dimensions[0];
		this.inputHeight = dimensions[1];
		this.inputWidth = dimensions[2];

		if (depth <= 0 || inputHeight <= 0 || inputWidth <= 0)
			throw new IllegalArgumentException("Invalid input dimensions.");

		// calculating the post padding dimensions
		padHeight = inputHeight + 2 * pad;
		padWidth = inputWidth + 2 * pad;

		if ((padHeight - filterSize) % stride != 0 || (padWidth - filterSize) % stride != 0) {
			Logger.getGlobal().log(Level.WARNING, "Filter sizes and stride do not match", new IllegalArgumentException());
		}

		// calculating the post convolution dimensions
		this.outputHeight = (padHeight - dilatedSize) / stride + 1;
		this.outputWidth = (padWidth - dilatedSize) / stride + 1;

		if (outputHeight <= 0 || outputWidth <= 0 || filterAmount <= 0)
			throw new IllegalArgumentException("Invalid output dimensions.");

		if (filterSize <= 0)
			throw new IllegalArgumentException("Invalid filter dimensions.");

		filters = new float[filterAmount * depth * filterSize * filterSize];
		filterUpdater = updaterType.create(filters.length, true);

		biases = new float[filterAmount];
		biasUpdater = updaterType.create(biases.length, false);

		int inputSize = depth * filterSize * filterSize;

		for (int f = 0; f < filterAmount; f++) {
			for (int k = 0; k < depth; k++) {
				for (int m = 0; m < filterSize; m++) {
					for (int n = 0; n < filterSize; n++) {
						int index = n + filterSize * (m + filterSize * (k + depth * f));

						filters[index] = initializer.initialize(inputSize);
					}
				}
			}
		}
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

		return pad(input, batchSize, pad, depth, padHeight, padWidth, inputHeight, inputWidth);
	}

	public float[] forward(float[] input, int batchSize) {
		this.batchSize = batchSize;

		input = pad(input, batchSize);
		output = new float[batchSize * filterAmount * outputHeight * outputWidth];

		int patchSize = dilatedSize * dilatedSize * depth;

		float[] biasMatrix = new float[batchSize * filterAmount * outputHeight * outputWidth];
		for (int b = 0; b < batchSize; b++) {
			for (int f = 0; f < filterAmount; f++) {
				for (int i = 0; i < outputHeight * outputWidth; i++) {
					biasMatrix[f + filterAmount * (i + outputHeight * outputWidth * b)] = biases[f];
				}
			}
		}

		// TODO: more efficient padding and dilations
		float[] dilated = dilate(filters, dilation, filterAmount, depth, filterSize, filterSize);

		float[] inputMatrix = new float[patchSize * outputHeight * outputWidth * batchSize];
		int inputIndex = 0;
		for (int b = 0; b < batchSize; b++) {
			for (int i = 0, h = 0; i < outputHeight; i++, h += stride) {
				for (int j = 0, w = 0; j < outputWidth; j++, w += stride) {
					for (int k = 0; k < depth; k++) {
						for (int m = 0; m < dilatedSize; m++) {
							for (int n = 0; n < dilatedSize; n++) {
								inputMatrix[inputIndex++] = input[(w + n) + padWidth * ((h + m) + padHeight * (k + depth * b))];
							}
						}
					}
				}
			}
		}

		inputBuffer = GPU.gpuAlloc(CL.CL_MEM_READ_ONLY, inputMatrix.length, inputMatrix);
		cl_mem dilatedBuffer = GPU.gpuAlloc(CL.CL_MEM_READ_ONLY, dilated.length, dilated);

		float[] conv = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeYes,
			outputHeight * outputWidth * batchSize, filterAmount, patchSize, inputBuffer, patchSize, dilatedBuffer, patchSize, biasMatrix,
			filterAmount);

		CL.clReleaseMemObject(dilatedBuffer);
		if (mode == Mode.EVAL)
			CL.clReleaseMemObject(inputBuffer);

		for (int f = 0; f < filterAmount; f++) {
			for (int i = 0; i < outputHeight * outputWidth; i++) {
				for (int b = 0; b < batchSize; b++) {
					output[i + (outputWidth * outputHeight) * (f + filterAmount * b)] =
						conv[f + filterAmount * (i + outputHeight * outputWidth * b)];
				}
			}
		}

		// activation
		activation.activation(output, batchSize);

		return output;
	}

	public float[] backward(Cost cost, float[] target, boolean calculateDelta) {
		return backward(cost.derivative(output, target, batchSize), calculateDelta);
	}

	public float[] backward(float[] previousDelta, boolean calculateDelta) {
		gradient = new float[filterAmount * depth * filterSize * filterSize];
		biasGradient = new float[filterAmount];

		// derivative
		output = activation.derivative(output);

		int patchSize = filterSize * filterSize * depth;

		float[] deltaMatrix = new float[batchSize * filterAmount * outputHeight * outputWidth];

		IntStream.range(0, batchSize).parallel().forEach(b -> {
			for (int f = 0; f < filterAmount; f++) {
				for (int i = 0, h = 0; i < outputHeight; i++, h += stride) {
					for (int j = 0, w = 0; j < outputWidth; j++, w += stride) {
						int index = j + outputWidth * (i + outputHeight * (f + filterAmount * b));

						// the bias gradient is the delta, since biases are just added to the output
						previousDelta[index] *= output[index];
						biasGradient[f] += previousDelta[index];
						deltaMatrix[f + filterAmount * (j + outputWidth * (i + outputHeight * b))] = previousDelta[index];
					}
				}
			}
		});

		cl_mem deltaBuffer = GPU.gpuAlloc(CL.CL_MEM_READ_ONLY, deltaMatrix.length, deltaMatrix);
		float[] result = GPU.sgemm(CLBlastTranspose.CLBlastTransposeYes, CLBlastTranspose.CLBlastTransposeNo,
			patchSize, filterAmount, outputHeight * outputWidth * batchSize, inputBuffer, patchSize,
			deltaBuffer, filterAmount, new float[patchSize * filterAmount], filterAmount);

		CL.clReleaseMemObject(deltaBuffer);
		CL.clReleaseMemObject(inputBuffer);

		for (int f = 0; f < filterAmount; f++) {
			for (int i = 0; i < patchSize; i++) {
				gradient[i + patchSize * f] = result[f + filterAmount * i];
			}
		}

		if (calculateDelta) {
			return calculateDelta(previousDelta);
		}

		return null;
	}

	private float[] calculateDelta(float[] previousDelta) {
		float[] delta = new float[batchSize * depth * padHeight * padWidth];

		previousDelta = dilate(previousDelta, stride, batchSize, filterAmount, outputHeight, outputWidth);

		int dilatedHeight = (outputHeight - 1) * (stride - 1) + outputHeight;
		int dilatedWidth = (outputWidth - 1) * (stride - 1) + outputWidth;

		int patchSize = filterSize * filterSize * filterAmount;

		float[] deltaMatrix = new float[patchSize * padHeight * padWidth * batchSize];
		int deltaIndex = 0;
		for (int b = 0; b < batchSize; b++) {
			for (int i = 0; i < padHeight; i++) {
				for (int j = 0; j < padWidth; j++) {
					for (int f = 0; f < filterAmount; f++) {
						for (int m = 0; m < filterSize; m++) {
							for (int n = 0; n < filterSize; n++) {
								// checks for full convolution
								if ((j - n) < dilatedWidth && (i - m) < dilatedHeight && (j - n) >= 0 && (i - m) >= 0) {
									int outputIndex = (j - n) + dilatedWidth * ((i - m) + dilatedHeight * (f + filterAmount * b));
									deltaMatrix[deltaIndex] = previousDelta[outputIndex];
								}

								deltaIndex++;
							}
						}
					}
				}
			}
		}

		float[] transposed = new float[filters.length];
		for (int k = 0; k < depth; k++) {
			for (int f = 0; f < filterAmount; f++) {
				for (int m = 0; m < filterSize; m++) {
					for (int n = 0; n < filterSize; n++) {
						int index = n + filterSize * (m + filterSize * (k + depth * f));
						transposed[k + depth * (n + filterSize * (m + filterSize * f))] = filters[index];
					}
				}
			}
		}

		float[] conv = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo,
			padHeight * padWidth * batchSize, depth, patchSize, deltaMatrix, patchSize, transposed, depth,
			new float[batchSize * padHeight * padWidth * depth], depth);

		for (int k = 0; k < depth; k++) {
			for (int i = 0; i < padHeight * padWidth; i++) {
				for (int b = 0; b < batchSize; b++) {
					delta[i + padHeight * padWidth * (k + depth * b)] = conv[k + depth * (i + padHeight * padWidth * b)];
				}
			}
		}

		return removePad(delta, batchSize);
	}

	private float[] removePad(float[] input, int batchSize) {
		if (batchSize <= 0)
			throw new IllegalArgumentException("Batch size must be > 0.");

		return removePad(input, batchSize, pad, depth, padWidth, inputHeight, inputWidth);
	}

	public void update(int length) {
		biasUpdater.update(biases, biasGradient, length);
		filterUpdater.update(filters, gradient, length);
	}

	public float[][][] getParameters() {
		return new float[][][]{{filters, gradient}, {biases, biasGradient}};
	}

	@SuppressWarnings("Duplicates")
	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(depth);
		dos.writeInt(inputHeight);
		dos.writeInt(inputWidth);
		dos.writeInt(padHeight);
		dos.writeInt(padWidth);
		dos.writeInt(pad);
		dos.writeInt(outputHeight);
		dos.writeInt(outputWidth);
		dos.writeInt(stride);
		dos.writeInt(filterAmount);
		dos.writeInt(filterSize);
		dos.writeInt(dilation);
		dos.writeInt(dilatedSize);

		activation.export(dos);

		filterUpdater.export(dos);
		biasUpdater.export(dos);

		for (int f = 0; f < filterAmount; f++) {
			dos.writeFloat(biases[f]);

			for (int k = 0; k < depth; k++) {
				for (int m = 0; m < filterSize; m++) {
					for (int n = 0; n < filterSize; n++) {
						int index = n + filterSize * (m + filterSize * (k + depth * f));

						dos.writeFloat(filters[index]);
					}
				}
			}
		}
	}

	public int[] getOutputDimensions() {
		return new int[]{filterAmount, outputHeight, outputWidth};
	}

	public LayerType getType() {
		return LayerType.CONVOLUTIONAL;
	}

	/**
	 * Builder for Convolutional layers.
	 */
	@SuppressWarnings({"unused", "WeakerAccess"})
	public static class Builder {
		private int pad;
		private int stride;
		private int filterAmount, filterSize;
		private int dilation;
		private Initializer initializer;
		private ActivationType activationType;

		public Builder() {
			dilation = 1;
		}

		/**
		 * The pad is the amount of zeroes that are padded around an input. Padding is used to preserve edge features during convolutions,
		 * by allowing filters to traverse such areas.
		 *
		 * @param pad the pad
		 * @return the builder
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
		 * @return the builder
		 */
		public Builder stride(int stride) {
			this.stride = stride;

			return this;
		}

		/**
		 * The filter amount is the amount of kernels that can be learned and applied during convolution.
		 *
		 * @param filterAmount the filter amount
		 * @return the builder
		 */
		public Builder filterAmount(int filterAmount) {
			this.filterAmount = filterAmount;

			return this;
		}

		/**
		 * The filter size is the size of filters. Larger filter sizes decrease output dimensions faster.
		 *
		 * @param filterSize the filter size
		 * @return the builder
		 */
		public Builder filterSize(int filterSize) {
			this.filterSize = filterSize;

			return this;
		}

		/**
		 * Sets the dilation for the filters.
		 *
		 * @param dilation the dilation
		 * @return the builder
		 */
		public Builder dilation(int dilation) {
			this.dilation = dilation;

			return this;
		}

		/**
		 * The initializer initializes weights.
		 *
		 * @param initializer the initializer
		 * @return the builder
		 */
		public Builder initializer(Initializer initializer) {
			this.initializer = initializer;

			return this;
		}

		/**
		 * The activation simulates a neuron firing.
		 *
		 * @param activationType the activation type
		 * @return the builder
		 */
		public Builder activationType(ActivationType activationType) {
			this.activationType = activationType;

			return this;
		}

		/**
		 * Builds the layer.
		 *
		 * @return the layer
		 */
		public Convolutional build() {
			return new Convolutional(pad, stride, filterAmount, filterSize, dilation, initializer, activationType);
		}
	}
}
