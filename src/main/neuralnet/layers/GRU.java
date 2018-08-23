package main.neuralnet.layers;

import main.GPU;
import main.neuralnet.activations.Activation;
import main.neuralnet.activations.ActivationType;
import main.neuralnet.activations.Identity;
import main.neuralnet.costs.Cost;
import main.neuralnet.initializers.HeInitialization;
import main.neuralnet.initializers.Initializer;
import main.neuralnet.optimizers.Updater;
import main.neuralnet.optimizers.UpdaterType;
import org.jocl.blast.CLBlastTranspose;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

public class GRU implements Layer {
	private Mode mode = Mode.TRAIN;
	private float[] wz, wr, w;
	private float[] dWz, dWr, dW;
	private float[] uz, ur, u;
	private float[] dUz, dUr, dU;
	private float[] bz, br, b;
	private float[] dBz, dBr, dB;
	private float[] state;
	private float[][] x, z, r, hc, h, rh, y;

	private int inputSize, hiddenSize;

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

		state = new float[hiddenSize];

		uz = new float[hiddenSize * hiddenSize];
		ur = new float[hiddenSize * hiddenSize];
		u = new float[hiddenSize * hiddenSize];

		bz = new float[hiddenSize];
		br = new float[hiddenSize];
		b = new float[hiddenSize];

		for (int i = 0; i < hiddenSize * hiddenSize; i++) {
			uz[i] = initializer.initialize(hiddenSize);
			ur[i] = initializer.initialize(hiddenSize);
			u[i] = initializer.initialize(hiddenSize);
		}
	}

	GRU(DataInputStream dis) throws IOException {
		inputSize = dis.readInt();
		hiddenSize = dis.readInt();

		updaters = new Updater[3 * hiddenSize * inputSize + 3 * hiddenSize * hiddenSize + 3 * hiddenSize];

		wz = new float[hiddenSize * inputSize];
		wr = new float[hiddenSize * inputSize];
		w = new float[hiddenSize * inputSize];

		uz = new float[hiddenSize * hiddenSize];
		ur = new float[hiddenSize * hiddenSize];
		u = new float[hiddenSize * hiddenSize];

		bz = new float[hiddenSize];
		br = new float[hiddenSize];
		b = new float[hiddenSize];

		hiddenActivation = ActivationType.fromString(dis).create();
		activation = ActivationType.fromString(dis).create();
		updaterType = UpdaterType.fromString(dis);

		int position = 0;
		for (int i = 0; i < hiddenSize * inputSize; i++) {
			wz[i] = dis.readFloat();
			updaters[position++] = updaterType.create(dis);
			wr[i] = dis.readFloat();
			updaters[position++] = updaterType.create(dis);
			w[i] = dis.readFloat();
			updaters[position++] = updaterType.create(dis);
		}

		for (int i = 0; i < hiddenSize * inputSize; i++) {
			uz[i] = dis.readFloat();
			updaters[position++] = updaterType.create(dis);
			ur[i] = dis.readFloat();
			updaters[position++] = updaterType.create(dis);
			u[i] = dis.readFloat();
			updaters[position++] = updaterType.create(dis);
		}

		for (int i = 0; i < hiddenSize * inputSize; i++) {
			bz[i] = dis.readFloat();
			updaters[position++] = updaterType.create(dis);
			br[i] = dis.readFloat();
			updaters[position++] = updaterType.create(dis);
			b[i] = dis.readFloat();
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

		wz = new float[hiddenSize * inputSize];
		wr = new float[hiddenSize * inputSize];
		w = new float[hiddenSize * inputSize];

		for (int i = 0; i < hiddenSize * inputSize; i++) {
			wz[i] = initializer.initialize(inputSize);
			wr[i] = initializer.initialize(inputSize);
			w[i] = initializer.initialize(inputSize);
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

		position = exportParameters(position, hiddenSize * inputSize, wz, wr, w, dos);
		position = exportParameters(position, hiddenSize * hiddenSize, uz, ur, u, dos);
		exportParameters(position, hiddenSize, bz, br, b, dos);
	}

	private int exportParameters(int position, int length, float[] pz, float[] pr, float[] p, DataOutputStream dos) throws IOException {
		for (int i = 0; i < length; i++) {
			dos.writeFloat(pz[i]);
			updaters[position++].export(dos);
			dos.writeFloat(pr[i]);
			updaters[position++].export(dos);
			dos.writeFloat(p[i]);
			updaters[position++].export(dos);
		}

		return position;
	}

	public void setMode(Mode mode) {
		if (mode == Mode.GRADIENT_CHECK) {
			for (int i = 0; i < state.length; i++) {
				state[i] = ThreadLocalRandom.current().nextFloat();
			}
		} else {
			state = new float[hiddenSize];
		}

		this.mode = mode;
	}

	public LayerType getType() {
		return LayerType.GRU;
	}

	public float[][] forward(float[][] x) {
		this.x = x;

		hc = new float[x.length][hiddenSize];
		z = new float[x.length][hiddenSize];
		r = new float[x.length][hiddenSize];
		h = new float[x.length + 1][hiddenSize];
		rh = new float[x.length][hiddenSize];
		y = new float[x.length][];

		h[0] = state;

		for (int t = 0; t < x.length; t++) {
			final int time = t;

			z[time] = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeYes, 1,
				hiddenSize, inputSize, x[time], inputSize, wz, inputSize, bz, hiddenSize);
			r[time] = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeYes, 1,
				hiddenSize, inputSize, x[time], inputSize, wr, inputSize, br, hiddenSize);
			z[time] = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeYes, 1,
				hiddenSize, hiddenSize, h[time], hiddenSize, uz, hiddenSize, z[time], hiddenSize);
			r[time] = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeYes, 1,
				hiddenSize, hiddenSize, h[time], hiddenSize, ur, hiddenSize, r[time], hiddenSize);

			hiddenActivation.activation(z[time]);
			hiddenActivation.activation(r[time]);

			rh[time] = new float[hiddenSize];
			IntStream.range(0, hiddenSize).parallel().forEach(i -> rh[time][i] += r[time][i] * h[time][i]);

			hc[time] = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeYes, 1,
				hiddenSize, inputSize, x[time], inputSize, w, inputSize, b, hiddenSize);
			hc[time] = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeYes, 1,
				hiddenSize, hiddenSize, rh[time], hiddenSize, u, hiddenSize, hc[time], hiddenSize);

			activation.activation(hc[time]);

			IntStream.range(0, hiddenSize).parallel().forEach(i -> h[time + 1][i] += z[time][i] * h[time][i] + (1 - z[time][i]) * hc[time][i]);

			y[time] = h[time + 1];
		}

		if (mode != Mode.GRADIENT_CHECK)
			state = h[x.length];

		return y;
	}

	public float[][] backward(Cost cost, float[][] target) {
		float[][] dx = new float[x.length][inputSize];
		float[][] dy = cost.derivative(y, target, new Identity());

		backward(dy, dx);

		return dx;
	}

	public float[][] backward(float[][] previousDelta) {
		float[][] dx = new float[x.length][inputSize];

		backward(previousDelta, dx);

		return dx;
	}

	private void backward(float[][] previousDelta, float[][] dx) {
		dWz = new float[hiddenSize * inputSize];
		dWr = new float[hiddenSize * inputSize];
		dW = new float[hiddenSize * inputSize];

		dUz = new float[hiddenSize * hiddenSize];
		dUr = new float[hiddenSize * hiddenSize];
		dU = new float[hiddenSize * hiddenSize];

		dBz = new float[hiddenSize];
		dBr = new float[hiddenSize];
		dB = new float[hiddenSize];

		// these variable represent before-activation derivatives
		float[] dh = new float[hiddenSize];
		float[] dr = new float[hiddenSize];
		float[] dz = new float[hiddenSize];
		float[] dhc = new float[hiddenSize];

		// activation derivatives
		float[][] derivative = activation.derivative(hc);
		float[][] dzActivation = hiddenActivation.derivative(z);
		float[][] drActivation = hiddenActivation.derivative(r);

		// TODO: Stack input and state for more parallel execution
		for (int t = x.length - 1; t >= 0; t--) {
			for (int i = 0; i < hiddenSize; i++) {
				dh[i] += previousDelta[t][i];
				dhc[i] = dh[i] * (1 - z[t][i]) * derivative[t][i];
			}

			// dhc/dht-1
			float[] dhNext = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, 1, hiddenSize,
				hiddenSize, dhc, hiddenSize, u, hiddenSize, new float[hiddenSize], hiddenSize);

			dx[t] = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, 1,
				inputSize, hiddenSize, dhc, hiddenSize, w, inputSize, new float[inputSize], inputSize);

			for (int i = 0; i < hiddenSize; i++) {
				dr[i] = h[t][i] * dhNext[i] * drActivation[t][i];
				dz[i] = dh[i] * (h[t][i] - hc[t][i]) * dzActivation[t][i];

				dh[i] = dh[i] * z[t][i] + r[t][i] * dhNext[i];
			}

			dx[t] = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, 1,
				inputSize, hiddenSize, dz, hiddenSize, wz, inputSize, dx[t], inputSize);
			dx[t] = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, 1,
				inputSize, hiddenSize, dr, hiddenSize, wr, inputSize, dx[t], inputSize);

			dh = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, 1,
				hiddenSize, hiddenSize, dz, hiddenSize, uz, hiddenSize, dh, hiddenSize);
			dh = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, 1,
				hiddenSize, hiddenSize, dr, hiddenSize, ur, hiddenSize, dh, hiddenSize);

			dWr = GPU.sger(hiddenSize, inputSize, dr, x[t], dWr, inputSize);
			dWz = GPU.sger(hiddenSize, inputSize, dz, x[t], dWz, inputSize);
			dW = GPU.sger(hiddenSize, inputSize, dhc, x[t], dW, inputSize);

			dUr = GPU.sger(hiddenSize, hiddenSize, dr, h[t], dUr, hiddenSize);
			dUz = GPU.sger(hiddenSize, hiddenSize, dz, h[t], dUz, hiddenSize);
			dU = GPU.sger(hiddenSize, hiddenSize, dhc, rh[t], dU, hiddenSize);

			IntStream.range(0, hiddenSize).parallel().forEach(i -> {
				dBr[i] += dr[i];
				dBz[i] += dz[i];
				dB[i] += dhc[i];
			});
		}

		if (mode == Mode.TRAIN)
			update();
	}

	public float[][][] getParameters() {
		return new float[][][]{{wz, dWz}, {wr, dWr}, {w, dW}, {uz, dUz}, {ur, dUr}, {u, dU}, {bz, dBz}, {br, dBr}, {b, dB}};
	}

	private void update() {
		int position = 0;

		for (int i = 0; i < hiddenSize * inputSize; i++) {
			wz[i] += updaters[position++].update(dWz[i] / x.length);
			wr[i] += updaters[position++].update(dWr[i] / x.length);
			w[i] += updaters[position++].update(dW[i] / x.length);
		}

		for (int i = 0; i < hiddenSize * hiddenSize; i++) {
			uz[i] += updaters[position++].update(dUz[i] / x.length);
			ur[i] += updaters[position++].update(dUr[i] / x.length);
			u[i] += updaters[position++].update(dU[i] / x.length);
		}

		for (int i = 0; i < hiddenSize; i++) {
			bz[i] += updaters[position++].update(dBz[i] / x.length);
			br[i] += updaters[position++].update(dBr[i] / x.length);
			b[i] += updaters[position++].update(dB[i] / x.length);
		}
	}

	@SuppressWarnings("unused")
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