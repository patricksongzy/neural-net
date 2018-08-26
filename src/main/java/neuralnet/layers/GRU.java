package neuralnet.layers;

import neuralnet.GPU;
import neuralnet.activations.Activation;
import neuralnet.activations.ActivationType;
import neuralnet.activations.Identity;
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
	private int inputSize, hiddenSize;

	private float[] wz, wr, wh;
	private float[] dWz, dWr, dWh;
	private float[] bz, br, bh;
	private float[] dBz, dBr, dBh;

	private float[][] h, y;
	private float[][] xh, xrh, z, r, hc;

	private Initializer initializer;
	private UpdaterType updaterType;
	private Updater[] updaters;
	private Activation hiddenActivation, activation;

	private GRU(int hiddenSize, Initializer initializer, UpdaterType updaterType,
				ActivationType hiddenActivation, ActivationType activation) {
		this.hiddenSize = hiddenSize;
		this.hiddenActivation = hiddenActivation.create();
		this.activation = activation.create();
		this.updaterType = updaterType;
		this.initializer = initializer;

		bz = new float[hiddenSize];
		br = new float[hiddenSize];
		bh = new float[hiddenSize];
	}

	GRU(DataInputStream dis) throws IOException {
		inputSize = dis.readInt();
		hiddenSize = dis.readInt();

		updaters = new Updater[3 * hiddenSize * inputSize + 3 * hiddenSize * hiddenSize + 3 * hiddenSize];

		wz = new float[hiddenSize * inputSize + hiddenSize * hiddenSize];
		wr = new float[hiddenSize * inputSize + hiddenSize * hiddenSize];
		wh = new float[hiddenSize * inputSize + hiddenSize * hiddenSize];

		bz = new float[hiddenSize];
		br = new float[hiddenSize];
		bh = new float[hiddenSize];

		hiddenActivation = ActivationType.fromString(dis).create();
		activation = ActivationType.fromString(dis).create();
		updaterType = UpdaterType.fromString(dis);

		int position = 0;
		for (int i = 0; i < hiddenSize * inputSize + hiddenSize * hiddenSize; i++) {
			wz[i] = dis.readFloat();
			updaters[position++] = updaterType.create(dis);
			wr[i] = dis.readFloat();
			updaters[position++] = updaterType.create(dis);
			wh[i] = dis.readFloat();
			updaters[position++] = updaterType.create(dis);
		}

		for (int i = 0; i < hiddenSize; i++) {
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

		updaters = new Updater[3 * hiddenSize * inputSize + 3 * hiddenSize * hiddenSize + 3 * hiddenSize];

		for (int i = 0; i < updaters.length; i++)
			updaters[i] = updaterType.create();

		wz = new float[hiddenSize * inputSize + hiddenSize * hiddenSize];
		wr = new float[hiddenSize * inputSize + hiddenSize * hiddenSize];
		wh = new float[hiddenSize * inputSize + hiddenSize * hiddenSize];

		for (int i = 0; i < hiddenSize * inputSize; i++) {
			wz[i] = initializer.initialize(inputSize);
			wr[i] = initializer.initialize(inputSize);
			wh[i] = initializer.initialize(inputSize);
		}

		for (int i = hiddenSize * inputSize; i < hiddenSize * inputSize + hiddenSize * hiddenSize; i++) {
			wz[i] = initializer.initialize(hiddenSize);
			wr[i] = initializer.initialize(hiddenSize);
			wh[i] = initializer.initialize(hiddenSize);
		}
	}

	public int[] getOutputDimensions() {
		return new int[]{hiddenSize};
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(inputSize);
		dos.writeInt(hiddenSize);

		hiddenActivation.getType().export(dos);
		activation.getType().export(dos);
		updaterType.export(dos);

		int position = 0;

		position = exportParameters(position, hiddenSize * inputSize + hiddenSize * hiddenSize, wz, wr, wh, dos);
		exportParameters(position, hiddenSize, bz, br, bh, dos);
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

		h = new float[input.length][batchSize * hiddenSize];
		if (mode == Mode.GRADIENT_CHECK)
			IntStream.range(0, input.length).parallel().forEach(t -> Arrays.fill(h[t], 0.1f));

		xh = new float[input.length][batchSize * (inputSize + hiddenSize)];
		xrh = new float[input.length][batchSize * (inputSize + hiddenSize)];
		hc = new float[input.length][batchSize * hiddenSize];
		z = new float[input.length][batchSize * hiddenSize];
		r = new float[input.length][batchSize * hiddenSize];
		y = new float[input.length][batchSize * hiddenSize];

		for (int t = 0; t < input.length; t++) {
			for (int b = 0; b < batchSize; b++) {
				System.arraycopy(bz, 0, z[t], hiddenSize * b, hiddenSize);
				System.arraycopy(br, 0, r[t], hiddenSize * b, hiddenSize);
				System.arraycopy(bh, 0, hc[t], hiddenSize * b, hiddenSize);
				System.arraycopy(input[t], inputSize * b, xh[t], (inputSize + hiddenSize) * b, inputSize);
				System.arraycopy(h[t], hiddenSize * b, xh[t], inputSize + (inputSize + hiddenSize) * b, hiddenSize);
			}

			z[t] = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeYes, batchSize,
				hiddenSize, inputSize + hiddenSize, xh[t], inputSize + hiddenSize, wz, inputSize + hiddenSize, z[t], hiddenSize);
			r[t] = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeYes, batchSize,
				hiddenSize, inputSize + hiddenSize, xh[t], inputSize + hiddenSize, wr, inputSize + hiddenSize, r[t], hiddenSize);

			hiddenActivation.activation(z[t], batchSize);
			hiddenActivation.activation(r[t], batchSize);

			for (int b = 0; b < batchSize; b++) {
				System.arraycopy(input[t], inputSize * b, xrh[t], (inputSize + hiddenSize) * b, inputSize);
				for (int i = 0; i < hiddenSize; i++) {
					int index = i + hiddenSize * b;
					xrh[t][(inputSize + i) + (inputSize + hiddenSize) * b] += r[t][index] * h[t][index];
				}
			}

			hc[t] = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeYes, batchSize,
				hiddenSize, inputSize + hiddenSize, xrh[t], inputSize + hiddenSize, wh, inputSize + hiddenSize, hc[t], hiddenSize);
			activation.activation(hc[t], batchSize);

			for (int b = 0; b < batchSize; b++) {
				for (int i = 0; i < hiddenSize; i++) {
					int index = i + hiddenSize * b;
					h[t][index] = z[t][index] * h[t][index] + (1 - z[t][index]) * hc[t][index];
				}

				System.arraycopy(h[t], hiddenSize * b, y[t], hiddenSize * b, hiddenSize);
			}
		}

		return y;
	}

	public float[][] backward(Cost cost, float[][] target) {
		float[][] dy = new float[xh.length][];

		for (int t = 0; t < xh.length; t++)
			dy[t] = cost.derivative(y[t], target[t], new Identity(), batchSize);

		return backward(dy);
	}

	public float[][] backward(float[][] previousDelta) {
		float[][] dx = new float[xh.length][batchSize * inputSize];

		dWz = new float[hiddenSize * inputSize + hiddenSize * hiddenSize];
		dWr = new float[hiddenSize * inputSize + hiddenSize * hiddenSize];
		dWh = new float[hiddenSize * inputSize + hiddenSize * hiddenSize];

		dBz = new float[hiddenSize];
		dBr = new float[hiddenSize];
		dBh = new float[hiddenSize];

		// these variable represent before-activation derivatives
		float[] dh = new float[batchSize * hiddenSize];
		float[] dr = new float[batchSize * hiddenSize];
		float[] dz = new float[batchSize * hiddenSize];
		float[] dhc = new float[batchSize * hiddenSize];

		for (int t = xh.length - 1; t >= 0; t--) {
			float[] derivative = activation.derivative(hc[t]);
			float[] dzActivation = hiddenActivation.derivative(z[t]);
			float[] drActivation = hiddenActivation.derivative(r[t]);

			for (int b = 0; b < batchSize; b++) {
				for (int i = 0; i < hiddenSize; i++) {
					int index = i + hiddenSize * b;

					dh[index] += previousDelta[t][index];
					dhc[index] = dh[index] * (1 - z[t][index]) * derivative[index];
				}
			}

			float[] delta = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, batchSize,
				hiddenSize + inputSize,
				hiddenSize, dhc, hiddenSize, wh, hiddenSize + inputSize, new float[batchSize * (hiddenSize + inputSize)],
				hiddenSize + inputSize);

			for (int b = 0; b < batchSize; b++) {
				for (int i = 0; i < hiddenSize; i++) {
					int index = i + hiddenSize * b;
					int inputIndex = (inputSize + i) + (inputSize + hiddenSize) * b;

					dr[index] = xh[t][inputIndex] * delta[inputIndex] * drActivation[index];
					dz[index] = dh[index] * (xh[t][inputIndex] - hc[t][index]) * dzActivation[index];

					delta[inputIndex] = dh[index] * z[t][index] + r[t][index] * delta[inputIndex];
				}

			}

			delta = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, batchSize, hiddenSize + inputSize,
				hiddenSize, dz, hiddenSize, wz, hiddenSize + inputSize, delta, hiddenSize + inputSize);
			delta = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, batchSize, hiddenSize + inputSize,
				hiddenSize, dr, hiddenSize, wr, hiddenSize + inputSize, delta, hiddenSize + inputSize);

			for (int b = 0; b < batchSize; b++) {
				System.arraycopy(delta, (inputSize + hiddenSize) * b, dx[t], inputSize * b, inputSize);
				System.arraycopy(delta, inputSize + (inputSize + hiddenSize) * b, dh, hiddenSize * b, hiddenSize);
			}

			dWr = GPU.sgemm(CLBlastTranspose.CLBlastTransposeYes, CLBlastTranspose.CLBlastTransposeNo, hiddenSize,
				inputSize + hiddenSize, batchSize, dr, hiddenSize, xh[t], inputSize + hiddenSize, dWr, inputSize + hiddenSize);
			dWz = GPU.sgemm(CLBlastTranspose.CLBlastTransposeYes, CLBlastTranspose.CLBlastTransposeNo, hiddenSize,
				inputSize + hiddenSize, batchSize, dz, hiddenSize, xh[t], inputSize + hiddenSize, dWz, inputSize + hiddenSize);
			dWh = GPU.sgemm(CLBlastTranspose.CLBlastTransposeYes, CLBlastTranspose.CLBlastTransposeNo, hiddenSize,
				inputSize + hiddenSize, batchSize, dhc, hiddenSize, xrh[t], inputSize + hiddenSize, dWh, inputSize + hiddenSize);

			IntStream.range(0, batchSize).parallel().forEach(b -> {
				for (int i = 0; i < hiddenSize; i++) {
					int index = i + hiddenSize * b;

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

		for (int i = 0; i < hiddenSize * inputSize + hiddenSize * hiddenSize; i++) {
			wz[i] += Math.max(Math.min(updaters[position++].update(dWz[i] / (xh.length * batchSize)), 1), -1);
			wr[i] += Math.max(Math.min(updaters[position++].update(dWr[i] / (xh.length * batchSize)), 1), -1);
			wh[i] += Math.max(Math.min(updaters[position++].update(dWh[i] / (xh.length * batchSize)), 1), -1);
		}

		for (int i = 0; i < hiddenSize; i++) {
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
			if (hiddenActivation != null && activation != null) {
				this.hiddenActivation = hiddenActivation;
				this.activation = activation;
			} else {
				throw new IllegalArgumentException();
			}

			return this;
		}

		public Builder initializer(Initializer initializer) {
			if (initializer != null)
				this.initializer = initializer;
			else
				throw new IllegalArgumentException();

			return this;
		}

		public Builder updaterType(UpdaterType updaterType) {
			if (updaterType != null)
				this.updaterType = updaterType;
			else
				throw new IllegalArgumentException();

			return this;
		}

		public GRU build() {
			if (hiddenSize > 0)
				return new GRU(hiddenSize, initializer, updaterType, hiddenActivation, activation);

			throw new IllegalArgumentException();
		}
	}
}