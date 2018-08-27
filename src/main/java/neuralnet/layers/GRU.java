package neuralnet.layers;

import neuralnet.GPU;
import neuralnet.activations.Activation;
import neuralnet.activations.ActivationType;
import neuralnet.costs.Cost;
import neuralnet.initializers.HeInitialization;
import neuralnet.initializers.Initializer;
import neuralnet.optimizers.Updater;
import neuralnet.optimizers.UpdaterType;
import org.jocl.blast.CLBlastTranspose;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.stream.IntStream;

public class GRU implements Layer {
	private Mode mode = Mode.TRAIN;

	private int batchSize;
	private int inputSize, outputSize;

	private float[] wz, wr, wh;
	private float[] dWz, dWr, dWh;
	private float[] bz, br, bh;
	private float[] dBz, dBr, dBh;

	private float[][] h, y;
	private float[][] xh, xrh, z, r, hc;

	private Initializer initializer;
	private UpdaterType updaterType;
	private Updater[] updaters;
	private Activation outputActivation, activation;

	private GRU(int outputSize, Initializer initializer, UpdaterType updaterType, ActivationType outputActivation,
				ActivationType activation) {
		if (outputSize <= 0)
			throw new IllegalArgumentException("Invalid output dimensions.");
		if (initializer == null || activation == null || outputActivation == null || updaterType == null)
			throw new IllegalArgumentException("Values cannot be null.");

		this.outputSize = outputSize;
		this.outputActivation = outputActivation;
		this.activation = activation;
		this.updaterType = updaterType;
		this.initializer = initializer;

		bz = new float[outputSize];
		br = new float[outputSize];
		bh = new float[outputSize];
	}

	GRU(DataInputStream dis) throws IOException {
		inputSize = dis.readInt();
		outputSize = dis.readInt();

		updaters = new Updater[3 * outputSize * inputSize + 3 * outputSize * outputSize + 3 * outputSize];

		wz = new float[outputSize * inputSize + outputSize * outputSize];
		wr = new float[outputSize * inputSize + outputSize * outputSize];
		wh = new float[outputSize * inputSize + outputSize * outputSize];

		bz = new float[outputSize];
		br = new float[outputSize];
		bh = new float[outputSize];

		outputActivation = Activation.fromString(dis);
		activation = Activation.fromString(dis);
		updaterType = UpdaterType.fromString(dis);

		int position = 0;
		for (int i = 0; i < outputSize * inputSize + outputSize * outputSize; i++) {
			wz[i] = dis.readFloat();
			updaters[position++] = updaterType.create(dis);
			wr[i] = dis.readFloat();
			updaters[position++] = updaterType.create(dis);
			wh[i] = dis.readFloat();
			updaters[position++] = updaterType.create(dis);
		}

		for (int i = 0; i < outputSize; i++) {
			bz[i] = dis.readFloat();
			updaters[position++] = updaterType.create(dis);
			br[i] = dis.readFloat();
			updaters[position++] = updaterType.create(dis);
			bh[i] = dis.readFloat();
			updaters[position++] = updaterType.create(dis);
		}
	}

	public void setDimensions(int... dimensions) {
		if (dimensions.length != 1) {
			inputSize = 1;

			for (int dimension : dimensions) {
				inputSize *= dimension;
			}
		} else {
			inputSize = dimensions[0];
		}

		if (inputSize <= 0)
			throw new IllegalArgumentException("Invalid input dimensions.");

		updaters = new Updater[3 * outputSize * inputSize + 3 * outputSize * outputSize + 3 * outputSize];

		for (int i = 0; i < updaters.length; i++)
			updaters[i] = updaterType.create();

		wz = new float[outputSize * inputSize + outputSize * outputSize];
		wr = new float[outputSize * inputSize + outputSize * outputSize];
		wh = new float[outputSize * inputSize + outputSize * outputSize];

		for (int i = 0; i < outputSize * inputSize; i++) {
			wz[i] = initializer.initialize(inputSize);
			wr[i] = initializer.initialize(inputSize);
			wh[i] = initializer.initialize(inputSize);
		}

		for (int i = outputSize * inputSize; i < outputSize * inputSize + outputSize * outputSize; i++) {
			wz[i] = initializer.initialize(outputSize);
			wr[i] = initializer.initialize(outputSize);
			wh[i] = initializer.initialize(outputSize);
		}
	}

	public int[] getOutputDimensions() {
		return new int[]{outputSize};
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(inputSize);
		dos.writeInt(outputSize);

		outputActivation.export(dos);
		activation.export(dos);
		updaterType.export(dos);

		int position = 0;

		position = exportParameters(position, outputSize * inputSize + outputSize * outputSize, wz, wr, wh, dos);
		exportParameters(position, outputSize, bz, br, bh, dos);
	}

	private int exportParameters(int position, int length, float[] z, float[] r, float[] p, DataOutputStream dos) throws IOException {
		for (int i = 0; i < length; i++) {
			dos.writeFloat(z[i]);
			updaters[position++].export(dos);
			dos.writeFloat(r[i]);
			updaters[position++].export(dos);
			dos.writeFloat(p[i]);
			updaters[position++].export(dos);
		}

		return position;
	}

	public void setMode(Mode mode) {
		this.mode = mode;
	}

	public LayerType getType() {
		return LayerType.GRU;
	}

	public float[][] forward(float[][] input, int batchSize) {
		this.batchSize = batchSize;

		h = new float[input.length][batchSize * outputSize];
		if (mode == Mode.GRADIENT_CHECK)
			IntStream.range(0, input.length).parallel().forEach(t -> Arrays.fill(h[t], 0.1f));

		xh = new float[input.length][batchSize * (inputSize + outputSize)];
		xrh = new float[input.length][batchSize * (inputSize + outputSize)];
		hc = new float[input.length][batchSize * outputSize];
		z = new float[input.length][batchSize * outputSize];
		r = new float[input.length][batchSize * outputSize];
		y = new float[input.length][batchSize * outputSize];

		for (int t = 0; t < input.length; t++) {
			for (int b = 0; b < batchSize; b++) {
				System.arraycopy(bz, 0, z[t], outputSize * b, outputSize);
				System.arraycopy(br, 0, r[t], outputSize * b, outputSize);
				System.arraycopy(bh, 0, hc[t], outputSize * b, outputSize);
				System.arraycopy(input[t], inputSize * b, xh[t], (inputSize + outputSize) * b, inputSize);
				System.arraycopy(h[t], outputSize * b, xh[t], inputSize + (inputSize + outputSize) * b, outputSize);
			}

			z[t] = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeYes, batchSize,
				outputSize, inputSize + outputSize, xh[t], inputSize + outputSize, wz, inputSize + outputSize, z[t], outputSize);
			r[t] = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeYes, batchSize,
				outputSize, inputSize + outputSize, xh[t], inputSize + outputSize, wr, inputSize + outputSize, r[t], outputSize);

			outputActivation.activation(z[t], batchSize);
			outputActivation.activation(r[t], batchSize);

			for (int b = 0; b < batchSize; b++) {
				System.arraycopy(input[t], inputSize * b, xrh[t], (inputSize + outputSize) * b, inputSize);
				for (int i = 0; i < outputSize; i++) {
					int index = i + outputSize * b;
					xrh[t][(inputSize + i) + (inputSize + outputSize) * b] += r[t][index] * h[t][index];
				}
			}

			hc[t] = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeYes, batchSize,
				outputSize, inputSize + outputSize, xrh[t], inputSize + outputSize, wh, inputSize + outputSize, hc[t], outputSize);
			activation.activation(hc[t], batchSize);

			for (int b = 0; b < batchSize; b++) {
				for (int i = 0; i < outputSize; i++) {
					int index = i + outputSize * b;
					h[t][index] = z[t][index] * h[t][index] + (1 - z[t][index]) * hc[t][index];
				}

				System.arraycopy(h[t], outputSize * b, y[t], outputSize * b, outputSize);
			}
		}

		return y;
	}

	public float[][] backward(Cost cost, float[][] target) {
		float[][] dy = new float[xh.length][];

		for (int t = 0; t < xh.length; t++)
			dy[t] = cost.derivative(y[t], target[t], batchSize);

		return backward(dy);
	}

	public float[][] backward(float[][] previousDelta) {
		float[][] dx = new float[xh.length][batchSize * inputSize];

		dWz = new float[outputSize * inputSize + outputSize * outputSize];
		dWr = new float[outputSize * inputSize + outputSize * outputSize];
		dWh = new float[outputSize * inputSize + outputSize * outputSize];

		dBz = new float[outputSize];
		dBr = new float[outputSize];
		dBh = new float[outputSize];

		// these variable represent before-activation derivatives
		float[] dh = new float[batchSize * outputSize];
		float[] dr = new float[batchSize * outputSize];
		float[] dz = new float[batchSize * outputSize];
		float[] dhc = new float[batchSize * outputSize];

		for (int t = xh.length - 1; t >= 0; t--) {
			float[] derivative = activation.derivative(hc[t]);
			float[] dzActivation = outputActivation.derivative(z[t]);
			float[] drActivation = outputActivation.derivative(r[t]);

			for (int b = 0; b < batchSize; b++) {
				for (int i = 0; i < outputSize; i++) {
					int index = i + outputSize * b;

					dh[index] += previousDelta[t][index];
					dhc[index] = dh[index] * (1 - z[t][index]) * derivative[index];
				}
			}

			float[] delta = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, batchSize,
				outputSize + inputSize,
				outputSize, dhc, outputSize, wh, outputSize + inputSize, new float[batchSize * (outputSize + inputSize)],
				outputSize + inputSize);

			for (int b = 0; b < batchSize; b++) {
				for (int i = 0; i < outputSize; i++) {
					int index = i + outputSize * b;
					int inputIndex = (inputSize + i) + (inputSize + outputSize) * b;

					dr[index] = xh[t][inputIndex] * delta[inputIndex] * drActivation[index];
					dz[index] = dh[index] * (xh[t][inputIndex] - hc[t][index]) * dzActivation[index];

					delta[inputIndex] = dh[index] * z[t][index] + r[t][index] * delta[inputIndex];
				}

			}

			delta = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, batchSize, outputSize + inputSize,
				outputSize, dz, outputSize, wz, outputSize + inputSize, delta, outputSize + inputSize);
			delta = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, batchSize, outputSize + inputSize,
				outputSize, dr, outputSize, wr, outputSize + inputSize, delta, outputSize + inputSize);

			for (int b = 0; b < batchSize; b++) {
				System.arraycopy(delta, (inputSize + outputSize) * b, dx[t], inputSize * b, inputSize);
				System.arraycopy(delta, inputSize + (inputSize + outputSize) * b, dh, outputSize * b, outputSize);
			}

			dWr = GPU.sgemm(CLBlastTranspose.CLBlastTransposeYes, CLBlastTranspose.CLBlastTransposeNo, outputSize,
				inputSize + outputSize, batchSize, dr, outputSize, xh[t], inputSize + outputSize, dWr, inputSize + outputSize);
			dWz = GPU.sgemm(CLBlastTranspose.CLBlastTransposeYes, CLBlastTranspose.CLBlastTransposeNo, outputSize,
				inputSize + outputSize, batchSize, dz, outputSize, xh[t], inputSize + outputSize, dWz, inputSize + outputSize);
			dWh = GPU.sgemm(CLBlastTranspose.CLBlastTransposeYes, CLBlastTranspose.CLBlastTransposeNo, outputSize,
				inputSize + outputSize, batchSize, dhc, outputSize, xrh[t], inputSize + outputSize, dWh, inputSize + outputSize);

			IntStream.range(0, batchSize).parallel().forEach(b -> {
				for (int i = 0; i < outputSize; i++) {
					int index = i + outputSize * b;

					dBr[i] += dr[index];
					dBz[i] += dz[index];
					dBh[i] += dhc[index];
				}
			});
		}

		if (mode != Mode.GRADIENT_CHECK)
			update();

		return dx;
	}

	public float[][][] getParameters() {
		return new float[][][]{{wz, dWz}, {wr, dWr}, {wh, dWh}, {bz, dBz}, {br, dBr}, {bh, dBh}};
	}

	private void update() {
		int position = 0;

		for (int i = 0; i < outputSize * inputSize + outputSize * outputSize; i++) {
			wz[i] += Math.max(Math.min(updaters[position++].update(dWz[i] / (xh.length * batchSize)), 1), -1);
			wr[i] += Math.max(Math.min(updaters[position++].update(dWr[i] / (xh.length * batchSize)), 1), -1);
			wh[i] += Math.max(Math.min(updaters[position++].update(dWh[i] / (xh.length * batchSize)), 1), -1);
		}

		for (int i = 0; i < outputSize; i++) {
			br[i] += Math.max(Math.min(updaters[position++].update(dBr[i] / (xh.length * batchSize)), 1), -1);
			bz[i] += Math.max(Math.min(updaters[position++].update(dBz[i] / (xh.length * batchSize)), 1), -1);
			bh[i] += Math.max(Math.min(updaters[position++].update(dBh[i] / (xh.length * batchSize)), 1), -1);
		}
	}

	@SuppressWarnings({"unused", "WeakerAccess"})
	public static class Builder {
		private int hiddenSize;
		private Initializer initializer;
		private ActivationType hiddenActivation;
		private ActivationType activation;
		private UpdaterType updaterType;

		public Builder() {
			initializer = new HeInitialization();
			hiddenActivation = ActivationType.SIGMOID;
			activation = ActivationType.TANH;
			updaterType = UpdaterType.ADAM;
		}

		public Builder hiddenSize(int hiddenSize) {
			this.hiddenSize = hiddenSize;

			return this;
		}

		public Builder activation(ActivationType hiddenActivation, ActivationType activation) {
			this.hiddenActivation = hiddenActivation;
			this.activation = activation;

			return this;
		}

		public Builder initializer(Initializer initializer) {
			this.initializer = initializer;

			return this;
		}

		public Builder updaterType(UpdaterType updaterType) {
			this.updaterType = updaterType;

			return this;
		}

		public GRU build() {
			return new GRU(hiddenSize, initializer, updaterType, hiddenActivation, activation);
		}
	}
}