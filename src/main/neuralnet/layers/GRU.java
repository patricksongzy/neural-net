package main.neuralnet.layers;

import main.neuralnet.activations.Activation;
import main.neuralnet.activations.ActivationType;
import main.neuralnet.activations.Identity;
import main.neuralnet.costs.Cost;
import main.neuralnet.initializers.HeInitialization;
import main.neuralnet.initializers.Initializer;
import main.neuralnet.optimizers.Updater;
import main.neuralnet.optimizers.UpdaterType;

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
	private float[][] x, z, r, hc, h, y;

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
		y = new float[x.length][];

		h[0] = state;

		for (int t = 0; t < x.length; t++) {
			final int time = t;

			IntStream.range(0, hiddenSize).parallel().forEach(i -> {
				for (int j = 0; j < inputSize; j++) {
					z[time][i] += wz[j + inputSize * i] * x[time][j];
					r[time][i] += wr[j + inputSize * i] * x[time][j];
				}

				for (int j = 0; j < hiddenSize; j++) {
					z[time][i] += uz[j + hiddenSize * i] * h[time][j];
					r[time][i] += ur[j + hiddenSize * i] * h[time][j];
				}

				z[time][i] += bz[i];
				r[time][i] += br[i];
			});

			hiddenActivation.activation(z[time]);
			hiddenActivation.activation(r[time]);

			float[] product = new float[hiddenSize];
			IntStream.range(0, hiddenSize).parallel().forEach(i -> product[i] += r[time][i] * h[time][i]);

			IntStream.range(0, hiddenSize).parallel().forEach(i -> {
				for (int j = 0; j < inputSize; j++) {
					hc[time][i] += w[j + inputSize * i] * x[time][j];
				}

				for (int j = 0; j < hiddenSize; j++) {
					hc[time][i] += u[j + hiddenSize * i] * product[j];
				}

				hc[time][i] += b[i];
			});

			activation.activation(hc[time]);

			IntStream.range(0, hiddenSize).parallel().forEach(i -> h[time + 1][i] += z[time][i] * h[time][i] + (1 - z[time][i]) * hc[time][i]);

			y[time] = h[time + 1];
		}

		if (mode != Mode.GRADIENT_CHECK)
			state = h[x.length];

		return y;
	}

	public float[][] backward(Cost cost, float[][] target) {
		dWz = new float[hiddenSize * inputSize];
		dWr = new float[hiddenSize * inputSize];
		dW = new float[hiddenSize * inputSize];

		dUz = new float[hiddenSize * hiddenSize];
		dUr = new float[hiddenSize * hiddenSize];
		dU = new float[hiddenSize * hiddenSize];

		dBz = new float[hiddenSize];
		dBr = new float[hiddenSize];
		dB = new float[hiddenSize];

		float[][] dx = new float[x.length][inputSize];
		float[][] dy = cost.derivative(y, target, new Identity());

		backward(dy, dx);

		return dx;
	}

	public float[][] backward(float[][] previousDelta) {
		dWz = new float[hiddenSize * inputSize];
		dWr = new float[hiddenSize * inputSize];
		dW = new float[hiddenSize * inputSize];

		dUz = new float[hiddenSize * hiddenSize];
		dUr = new float[hiddenSize * hiddenSize];
		dU = new float[hiddenSize * hiddenSize];

		dBz = new float[hiddenSize];
		dBr = new float[hiddenSize];
		dB = new float[hiddenSize];

		float[][] dx = new float[x.length][inputSize];

		backward(previousDelta, dx);

		return dx;
	}

	private void backward(float[][] previousDelta, float[][] dx) {
		// these variable represent before-activation derivatives
		float[] dh = new float[hiddenSize];
		float[] dr = new float[hiddenSize];
		float[] dz = new float[hiddenSize];
		float[] dhc = new float[hiddenSize];

		// activation derivatives
		float[][] derivative = activation.derivative(hc);
		float[][] dzActivation = hiddenActivation.derivative(z);
		float[][] drActivation = hiddenActivation.derivative(r);

		for (int t = x.length - 1; t >= 0; t--) {
			final int time = t;

			IntStream.range(0, hiddenSize).parallel().forEach(i -> {
				dh[i] += previousDelta[time][i];
				dhc[i] = dh[i] * (1 - z[time][i]) * derivative[time][i];
			});

			IntStream.range(0, hiddenSize).parallel().forEach(i -> {
				// dhc/dht-1
				float dhNext = 0;

				for (int j = 0; j < hiddenSize; j++) {
					dhNext += u[i + hiddenSize * j] * dhc[j];
				}

				dr[i] = h[time][i] * dhNext * drActivation[time][i];
				dz[i] = dh[i] * (h[time][i] - hc[time][i]) * dzActivation[time][i];

				dh[i] = dh[i] * z[time][i] + r[time][i] * dhNext;

				for (int j = 0; j < inputSize; j++) {
					dx[time][j] += w[j + inputSize * i] * dhc[i];
				}
			});

			IntStream.range(0, hiddenSize).parallel().forEach(i -> {
				for (int j = 0; j < inputSize; j++) {
					dx[time][j] += wz[j + inputSize * i] * dz[i];
					dx[time][j] += wr[j + inputSize * i] * dr[i];
				}

				for (int j = 0; j < hiddenSize; j++) {
					dh[i] += uz[i + hiddenSize * j] * dz[j];
					dh[i] += ur[i + hiddenSize * j] * dr[j];
				}
			});

			float[] product = new float[hiddenSize];

			IntStream.range(0, hiddenSize).parallel().forEach(i -> product[i] = r[time][i] * h[time][i]);

			IntStream.range(0, hiddenSize).parallel().forEach(i -> {
				for (int j = 0; j < inputSize; j++) {
					dWr[j + inputSize * i] += dr[i] * x[time][j];
					dWz[j + inputSize * i] += dz[i] * x[time][j];
					dW[j + inputSize * i] += dhc[i] * x[time][j];
				}

				for (int j = 0; j < hiddenSize; j++) {
					dUr[j + hiddenSize * i] += dr[i] * h[time][j];
					dUz[j + hiddenSize * i] += dz[i] * h[time][j];
					dU[j + hiddenSize * i] += dhc[i] * product[j];
				}

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