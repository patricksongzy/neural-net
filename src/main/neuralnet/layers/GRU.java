package main.neuralnet.layers;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Range;
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
	private HiddenKernel hiddenKernel;
	private WeightGradientKernel weightGradientKernel;
	private StateGradientKernel stateGradientKernel;
	private HiddenDeltaKernel hiddenDeltaKernel;
	private DeltaKernel deltaKernel;

	private Mode mode = Mode.TRAIN;
	private double[] wz, wr, w;
	private double[] dWz, dWr, dW;
	private double[] uz, ur, u;
	private double[] dUz, dUr, dU;
	private double[] bz, br, b;
	private double[] dBz, dBr, dB;
	private double[] state;
	private double[][] x, z, r, hc, h, y;

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

		state = new double[hiddenSize];

		uz = new double[hiddenSize * hiddenSize];
		ur = new double[hiddenSize * hiddenSize];
		u = new double[hiddenSize * hiddenSize];

		bz = new double[hiddenSize];
		br = new double[hiddenSize];
		b = new double[hiddenSize];

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

		wz = new double[hiddenSize * inputSize];
		wr = new double[hiddenSize * inputSize];
		w = new double[hiddenSize * inputSize];

		uz = new double[hiddenSize * hiddenSize];
		ur = new double[hiddenSize * hiddenSize];
		u = new double[hiddenSize * hiddenSize];

		bz = new double[hiddenSize];
		br = new double[hiddenSize];
		b = new double[hiddenSize];

		hiddenActivation = ActivationType.fromString(dis).create();
		activation = ActivationType.fromString(dis).create();
		updaterType = UpdaterType.fromString(dis);

		int position = 0;
		for (int i = 0; i < hiddenSize * inputSize; i++) {
			wz[i] = dis.readDouble();
			updaters[position++] = updaterType.create(dis);
			wr[i] = dis.readDouble();
			updaters[position++] = updaterType.create(dis);
			w[i] = dis.readDouble();
			updaters[position++] = updaterType.create(dis);
		}

		for (int i = 0; i < hiddenSize * inputSize; i++) {
			uz[i] = dis.readDouble();
			updaters[position++] = updaterType.create(dis);
			ur[i] = dis.readDouble();
			updaters[position++] = updaterType.create(dis);
			u[i] = dis.readDouble();
			updaters[position++] = updaterType.create(dis);
		}

		for (int i = 0; i < hiddenSize * inputSize; i++) {
			bz[i] = dis.readDouble();
			updaters[position++] = updaterType.create(dis);
			br[i] = dis.readDouble();
			updaters[position++] = updaterType.create(dis);
			b[i] = dis.readDouble();
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

		wz = new double[hiddenSize * inputSize];
		wr = new double[hiddenSize * inputSize];
		w = new double[hiddenSize * inputSize];

		for (int i = 0; i < hiddenSize * inputSize; i++) {
			wz[i] = initializer.initialize(inputSize);
			wr[i] = initializer.initialize(inputSize);
			w[i] = initializer.initialize(inputSize);
		}

		hiddenKernel = new HiddenKernel(inputSize, hiddenSize);
		weightGradientKernel = new WeightGradientKernel(inputSize);
		stateGradientKernel = new StateGradientKernel(hiddenSize);
		hiddenDeltaKernel = new HiddenDeltaKernel(hiddenSize);
		deltaKernel = new DeltaKernel(inputSize);
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

	private int exportParameters(int position, int length, double[] pz, double[] pr, double[] p, DataOutputStream dos) throws IOException {
		for (int i = 0; i < length; i++) {
			dos.writeDouble(pz[i]);
			updaters[position++].export(dos);
			dos.writeDouble(pr[i]);
			updaters[position++].export(dos);
			dos.writeDouble(p[i]);
			updaters[position++].export(dos);
		}

		return position;
	}

	public void setMode(Mode mode) {
		if (mode == Mode.GRADIENT_CHECK) {
			for (int i = 0; i < state.length; i++) {
				state[i] = ThreadLocalRandom.current().nextDouble();
			}
		} else {
			state = new double[hiddenSize];
		}

		this.mode = mode;
	}

	public LayerType getType() {
		return LayerType.GRU;
	}

	public double[][] forward(double[][] x) {
		this.x = x;

		hc = new double[x.length][hiddenSize];
		z = new double[x.length][hiddenSize];
		r = new double[x.length][hiddenSize];
		h = new double[x.length + 1][hiddenSize];
		y = new double[x.length][];

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

			double[] product = new double[hiddenSize];
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

	public double[][] backward(Cost cost, double[][] target) {
		dWz = new double[hiddenSize * inputSize];
		dWr = new double[hiddenSize * inputSize];
		dW = new double[hiddenSize * inputSize];

		dUz = new double[hiddenSize * hiddenSize];
		dUr = new double[hiddenSize * hiddenSize];
		dU = new double[hiddenSize * hiddenSize];

		dBz = new double[hiddenSize];
		dBr = new double[hiddenSize];
		dB = new double[hiddenSize];

		double[][] dx = new double[x.length][inputSize];
		double[][] dy = cost.derivative(y, target, new Identity());

		backward(dy, dx);

		return dx;
	}

	public double[][] backward(double[][] previousDelta) {
		dWz = new double[hiddenSize * inputSize];
		dWr = new double[hiddenSize * inputSize];
		dW = new double[hiddenSize * inputSize];

		dUz = new double[hiddenSize * hiddenSize];
		dUr = new double[hiddenSize * hiddenSize];
		dU = new double[hiddenSize * hiddenSize];

		dBz = new double[hiddenSize];
		dBr = new double[hiddenSize];
		dB = new double[hiddenSize];

		double[][] dx = new double[x.length][inputSize];

		backward(previousDelta, dx);

		return dx;
	}

	private void backward(double[][] previousDelta, double[][] dx) {
		// these variable represent before-activation derivatives
		double[] dh = new double[hiddenSize];
		double[] dr = new double[hiddenSize];
		double[] dz = new double[hiddenSize];
		double[] dhc = new double[hiddenSize];

		// activation derivatives
		double[][] derivative = activation.derivative(hc);
		double[][] dzActivation = hiddenActivation.derivative(z);
		double[][] drActivation = hiddenActivation.derivative(r);

		for (int t = x.length - 1; t >= 0; t--) {
			final int time = t;

			IntStream.range(0, hiddenSize).parallel().forEach(i -> {
				dh[i] += previousDelta[time][i];
				dhc[i] = dh[i] * (1 - z[time][i]) * derivative[time][i];
			});

			hiddenKernel.init(time, w, u, dr, dz, dh, dhc, h, z, r, hc, drActivation, dzActivation, dx);
			hiddenKernel.execute(Range.create(hiddenSize));

			hiddenDeltaKernel.init(dh, uz, ur, dz, dr);
			hiddenDeltaKernel.execute(Range.create2D(hiddenSize, hiddenSize));

			deltaKernel.init(time, wz, wr, dz, dr, dx);
			deltaKernel.execute(Range.create2D(hiddenSize, inputSize));

			double[] product = new double[hiddenSize];

			IntStream.range(0, hiddenSize).parallel().forEach(i -> product[i] = r[time][i] * h[time][i]);

			weightGradientKernel.init(time, dr, dz, dhc, dWr, dWz, dW, x);
			weightGradientKernel.execute(Range.create2D(hiddenSize, inputSize));

			stateGradientKernel.init(time, dr, dz, dhc, dUr, dUz, dU, product, h);
			stateGradientKernel.execute(Range.create2D(hiddenSize, hiddenSize));

			IntStream.range(0, hiddenSize).parallel().forEach(i -> {
				dBr[i] += dr[i];
				dBz[i] += dz[i];
				dB[i] += dhc[i];
			});
		}

		if (mode == Mode.TRAIN)
			update();
	}

	public double[][][] getParameters() {
		return new double[][][]{{wz, dWz}, {wr, dWr}, {w, dW}, {uz, dUz}, {ur, dUr}, {u, dU}, {bz, dBz}, {br, dBr}, {b, dB}};
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

	class HiddenDeltaKernel extends Kernel {
		private int hiddenSize;
		private double[] dh, uz, ur, dz, dr;

		HiddenDeltaKernel(int hiddenSize) {
			this.hiddenSize = hiddenSize;
		}

		void init(double[] dh, double[] uz, double[] ur, double[] dz, double[] dr) {
			this.dh = dh;
			this.uz = uz;
			this.ur = ur;
			this.dz = dz;
			this.dr = dr;
		}

		public void run() {
			int i = getGlobalId(0);
			int j = getGlobalId(1);

			dh[i] += uz[i + hiddenSize * j] * dz[j];
			dh[i] += ur[i + hiddenSize * j] * dr[j];
		}
	}

	class DeltaKernel extends Kernel {
		private int inputSize;
		private int time;
		private double[] wz, wr, dz, dr;
		private double[][] dx;

		DeltaKernel(int inputSize) {
			this.inputSize = inputSize;
		}

		void init(int time, double[] wz, double[] wr, double[] dz, double[] dr, double[][] dx) {
			this.time = time;
			this.wz = wz;
			this.wr = wr;
			this.dz = dz;
			this.dr = dr;
			this.dx = dx;
		}

		public void run() {
			int i = getGlobalId(0);
			int j = getGlobalId(1);

			dx[time][j] += wz[j + inputSize * i] * dz[i];
			dx[time][j] += wr[j + inputSize * i] * dr[i];
		}
	}

	class WeightGradientKernel extends Kernel {
		private int inputSize;
		private int time;
		private double[] dr, dz, dhc, dWr, dWz, dW;
		private double[][] x;

		WeightGradientKernel(int inputSize) {
			this.inputSize = inputSize;
		}

		void init(int time, double[] dr, double[] dz, double[] dhc, double[] dWr, double[] dWz, double[] dW, double[][] x) {
			this.time = time;
			this.dr = dr;
			this.dz = dz;
			this.dhc = dhc;
			this.dWr = dWr;
			this.dWz = dWz;
			this.dW = dW;
			this.x = x;
		}

		public void run() {
			int i = getGlobalId(0);
			int j = getGlobalId(1);

			dWr[j + inputSize * i] += dr[i] * x[time][j];
			dWz[j + inputSize * i] += dz[i] * x[time][j];
			dW[j + inputSize * i] += dhc[i] * x[time][j];
		}
	}

	class StateGradientKernel extends Kernel {
		private int hiddenSize;
		private int time;
		private double[] dr, dz, dhc, dUr, dUz, dU, product;
		private double[][] h;

		StateGradientKernel(int hiddenSize) {
			this.hiddenSize = hiddenSize;
		}

		void init(int time, double[] dr, double[] dz, double[] dhc, double[] dUr, double[] dUz, double[] dU, double[] product,
				  double[][] h) {
			this.time = time;
			this.dr = dr;
			this.dz = dz;
			this.dhc = dhc;
			this.dUr = dUr;
			this.dUz = dUz;
			this.dU = dU;
			this.product = product;
			this.h = h;
		}

		public void run() {
			int i = getGlobalId(0);
			int j = getGlobalId(1);

			dUr[j + hiddenSize * i] += dr[i] * h[time][j];
			dUz[j + hiddenSize * i] += dz[i] * h[time][j];
			dU[j + hiddenSize * i] += dhc[i] * product[j];
		}
	}

	class HiddenKernel extends Kernel {
		private int inputSize, hiddenSize;
		private int time;
		private double[] w, u;
		private double[] dr, dz, dh, dhc;
		private double[][] h, z, r, hc;
		private double[][] drActivation, dzActivation, dx;

		HiddenKernel(int inputSize, int hiddenSize) {
			this.inputSize = inputSize;
			this.hiddenSize = hiddenSize;
		}

		void init(int time, double[] w, double[] u, double[] dr, double[] dz, double[] dh, double[] dhc, double[][] h, double[][] z,
				  double[][] r, double[][] hc, double[][] drActivation, double[][] dzActivation, double[][] dx) {
			this.time = time;
			this.w = w;
			this.u = u;
			this.dr = dr;
			this.dz = dz;
			this.dh = dh;
			this.dhc = dhc;
			this.h = h;
			this.z = z;
			this.r = r;
			this.hc = hc;
			this.drActivation = drActivation;
			this.dzActivation = dzActivation;
			this.dx = dx;
		}

		public void run() {
			int i = getGlobalId();

			double dot = 0;

			for (int j = 0; j < hiddenSize; j++) {
				dot += u[i + hiddenSize * j] * dhc[j];
			}

			dr[i] = h[time][i] * dot * drActivation[time][i];
			dz[i] = dh[i] * (h[time][i] - hc[time][i]) * dzActivation[time][i];

			dh[i] = dh[i] * z[time][i] + r[time][i] * dot;

			for (int j = 0; j < inputSize; j++) {
				dx[time][j] += w[j + inputSize * i] * dhc[i];
			}
		}
	}
}