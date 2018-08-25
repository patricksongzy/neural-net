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
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

public class GRU implements Layer {
	private Mode mode = Mode.TRAIN;
	private float[] wz, wr, w;
	private float[] dWz, dWr, dW;
	private float[] bz, br, b;
	private float[] dBz, dBr, dB;
	private float[] h;
	private float[][] xh, xrh, z, r, hc, y;

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

		bz = new float[hiddenSize];
		br = new float[hiddenSize];
		b = new float[hiddenSize];
	}

	GRU(DataInputStream dis) throws IOException {
		inputSize = dis.readInt();
		hiddenSize = dis.readInt();

		updaters = new Updater[3 * hiddenSize * inputSize + 3 * hiddenSize * hiddenSize + 3 * hiddenSize];

		wz = new float[hiddenSize * inputSize + hiddenSize * hiddenSize];
		wr = new float[hiddenSize * inputSize + hiddenSize * hiddenSize];
		w = new float[hiddenSize * inputSize + hiddenSize * hiddenSize];

		bz = new float[hiddenSize];
		br = new float[hiddenSize];
		b = new float[hiddenSize];

		hiddenActivation = ActivationType.fromString(dis).create();
		activation = ActivationType.fromString(dis).create();
		updaterType = UpdaterType.fromString(dis);

		int position = 0;
		for (int i = 0; i < hiddenSize * inputSize + hiddenSize * hiddenSize; i++) {
			wz[i] = dis.readFloat();
			updaters[position++] = updaterType.create(dis);
			wr[i] = dis.readFloat();
			updaters[position++] = updaterType.create(dis);
			w[i] = dis.readFloat();
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

		wz = new float[hiddenSize * inputSize + hiddenSize * hiddenSize];
		wr = new float[hiddenSize * inputSize + hiddenSize * hiddenSize];
		w = new float[hiddenSize * inputSize + hiddenSize * hiddenSize];

		for (int i = 0; i < hiddenSize * inputSize; i++) {
			wz[i] = initializer.initialize(inputSize);
			wr[i] = initializer.initialize(inputSize);
			w[i] = initializer.initialize(inputSize);
		}

		for (int i = hiddenSize * inputSize; i < hiddenSize * inputSize + hiddenSize * hiddenSize; i++) {
			wz[i] = initializer.initialize(hiddenSize);
			wr[i] = initializer.initialize(hiddenSize);
			w[i] = initializer.initialize(hiddenSize);
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

		position = exportParameters(position, hiddenSize * inputSize + hiddenSize * hiddenSize, wz, wr, w, dos);
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
		h = new float[hiddenSize];

		if (mode == Mode.GRADIENT_CHECK) {
			for (int i = 0; i < h.length; i++) {
				h[i] = ThreadLocalRandom.current().nextFloat();
			}
		}

		this.mode = mode;
	}

	public LayerType getType() {
		return LayerType.GRU;
	}

	public float[][] forward(float[][] x) {
		xh = new float[x.length][inputSize + hiddenSize];
		xrh = new float[x.length][inputSize + hiddenSize];
		hc = new float[x.length][hiddenSize];
		z = new float[x.length][hiddenSize];
		r = new float[x.length][hiddenSize];
		y = new float[x.length][hiddenSize];

		for (int t = 0; t < x.length; t++) {
			System.arraycopy(x[t], 0, xh[t], 0, inputSize);
			System.arraycopy(h, 0, xh[t], inputSize, hiddenSize);

			z[t] = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeYes, 1,
				hiddenSize, inputSize + hiddenSize, xh[t], inputSize + hiddenSize, wz, inputSize + hiddenSize, bz, hiddenSize);
			r[t] = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeYes, 1,
				hiddenSize, inputSize + hiddenSize, xh[t], inputSize + hiddenSize, wr, inputSize + hiddenSize, br, hiddenSize);

			hiddenActivation.activation(z[t]);
			hiddenActivation.activation(r[t]);

			xrh[t] = new float[inputSize + hiddenSize];
			System.arraycopy(x[t], 0, xrh[t], 0, inputSize);
			for (int i = 0; i < hiddenSize; i++)
				xrh[t][inputSize + i] += r[t][i] * h[i];

			hc[t] = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeYes, 1,
				hiddenSize, inputSize + hiddenSize, xrh[t], inputSize + hiddenSize, w, inputSize + hiddenSize, b, hiddenSize);

			activation.activation(hc[t]);

			for (int i = 0; i < hiddenSize; i++)
				h[i] = z[t][i] * h[i] + (1 - z[t][i]) * hc[t][i];

			System.arraycopy(h, 0, y[t], 0, hiddenSize);
		}

		if (mode == Mode.GRADIENT_CHECK)
			System.arraycopy(xh[0], inputSize, h, 0, hiddenSize);

		return y;
	}

	public float[] backward(Cost cost, float[][] target) {
		float[] dy = cost.derivative(y, target, new Identity());

		return backward(dy);
	}

	public float[] backward(float[] previousDelta) {
		float[] dx = new float[xh.length * inputSize];

		dWz = new float[hiddenSize * inputSize + hiddenSize * hiddenSize];
		dWr = new float[hiddenSize * inputSize + hiddenSize * hiddenSize];
		dW = new float[hiddenSize * inputSize + hiddenSize * hiddenSize];

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

		for (int t = xh.length - 1; t >= 0; t--) {
			for (int i = 0; i < hiddenSize; i++) {
				dh[i] += previousDelta[i + hiddenSize * t];
				dhc[i] = dh[i] * (1 - z[t][i]) * derivative[t][i];
			}

			float[] delta = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, 1, hiddenSize + inputSize,
				hiddenSize, dhc, hiddenSize, w, hiddenSize + inputSize, new float[hiddenSize + inputSize], hiddenSize + inputSize);

			for (int i = 0; i < hiddenSize; i++) {
				dr[i] = xh[t][inputSize + i] * delta[inputSize + i] * drActivation[t][i];
				dz[i] = dh[i] * (xh[t][inputSize + i] - hc[t][i]) * dzActivation[t][i];

				delta[inputSize + i] = dh[i] * z[t][i] + r[t][i] * delta[inputSize + i];
			}

			delta = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, 1, hiddenSize + inputSize,
				hiddenSize, dz, hiddenSize, wz, hiddenSize + inputSize, delta, hiddenSize + inputSize);
			delta = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, 1, hiddenSize + inputSize,
				hiddenSize, dr, hiddenSize, wr, hiddenSize + inputSize, delta, hiddenSize + inputSize);

			System.arraycopy(delta, 0, dx, inputSize * t, inputSize);
			System.arraycopy(delta, inputSize, dh, 0, hiddenSize);

			dWr = GPU.sger(hiddenSize, inputSize + hiddenSize, dr, xh[t], dWr, inputSize + hiddenSize);
			dWz = GPU.sger(hiddenSize, inputSize + hiddenSize, dz, xh[t], dWz, inputSize + hiddenSize);
			dW = GPU.sger(hiddenSize, inputSize + hiddenSize, dhc, xrh[t], dW, inputSize + hiddenSize);

			IntStream.range(0, hiddenSize).parallel().forEach(i -> {
				dBr[i] += dr[i];
				dBz[i] += dz[i];
				dB[i] += dhc[i];
			});
		}

		if (mode != Mode.GRADIENT_CHECK)
			update();

		return dx;
	}

	public float[][][] getParameters() {
		return new float[][][]{{wz, dWz}, {wr, dWr}, {w, dW}, {bz, dBz}, {br, dBr}, {b, dB}};
	}

	private void update() {
		int position = 0;

		for (int i = 0; i < hiddenSize * inputSize + hiddenSize * hiddenSize; i++) {
			wz[i] += updaters[position++].update(dWz[i] / xh.length);
			wr[i] += updaters[position++].update(dWr[i] / xh.length);
			w[i] += updaters[position++].update(dW[i] / xh.length);
		}

		for (int i = 0; i < hiddenSize; i++) {
			bz[i] += updaters[position++].update(dBz[i] / xh.length);
			br[i] += updaters[position++].update(dBr[i] / xh.length);
			b[i] += updaters[position++].update(dB[i] / xh.length);
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