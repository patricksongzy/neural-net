package neuralnet.layers;

import neuralnet.GPU;
import neuralnet.activations.Activation;
import neuralnet.activations.ActivationType;
import neuralnet.costs.Cost;
import neuralnet.initializers.Initializer;
import neuralnet.optimizers.Updater;
import neuralnet.optimizers.UpdaterType;
import org.jocl.blast.CLBlastTranspose;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;

public class GRU implements Layer {
	private Mode mode = Mode.TRAIN;

	private int batchSize;
	private int inputSize, outputSize;

	// weights
	private float[] wz, wr, wh;
	// pre-transposed weights
	private float[] wzT, wrT, whT;
	// weight gradients
	private float[] dWz, dWr, dWh;
	// biases
	private float[] bz, br, bh;
	// bias gradients
	private float[] dBz, dBr, dBh;

	// state
	private float[] h;
	// derivative
	private float[] dh;
	// outputs stored in linked list
	private LinkedList<float[]> xh, xrh, z, r, hc, y;

	private Initializer initializer;
	private UpdaterType updaterType;
	private Updater[] weightUpdaters;
	private Updater[] biasUpdaters;
	private Activation hiddenActivation, activation;

	private GRU(int outputSize, Initializer initializer, UpdaterType updaterType, ActivationType hiddenActivation,
				ActivationType activation) {
		if (initializer == null || activation == null || hiddenActivation == null || updaterType == null)
			throw new IllegalArgumentException("Values cannot be null.");

		this.outputSize = outputSize;
		this.hiddenActivation = hiddenActivation;
		this.activation = activation;
		this.updaterType = updaterType;
		this.initializer = initializer;

		bz = new float[outputSize];
		br = new float[outputSize];
		bh = new float[outputSize];
	}

	GRU(DataInputStream dis) throws IOException {
		System.out.println("Type: " + getType());

		inputSize = dis.readInt();
		outputSize = dis.readInt();

		System.out.println("Input Size: " + inputSize);
		System.out.println("Output Size: " + outputSize);

		wz = new float[outputSize * inputSize + outputSize * outputSize];
		wr = new float[outputSize * inputSize + outputSize * outputSize];
		wh = new float[outputSize * inputSize + outputSize * outputSize];

		bz = new float[outputSize];
		br = new float[outputSize];
		bh = new float[outputSize];

		hiddenActivation = Activation.fromString(dis);
		activation = Activation.fromString(dis);
		System.out.println("Activations (z/r, hc): " + hiddenActivation.getType() + ", " + activation.getType());

		updaterType = UpdaterType.fromString(dis);
		System.out.println("Updater type: " + updaterType);

		biasUpdaters = new Updater[3];
		weightUpdaters = new Updater[3];
		for (int i = 0; i < 3; i++) {
			weightUpdaters[i] = updaterType.create(dis);
			biasUpdaters[i] = updaterType.create(dis);
		}

		System.out.println("Importing weights.");
		for (int i = 0; i < outputSize * inputSize + outputSize * outputSize; i++) {
			wz[i] = dis.readFloat();
			wr[i] = dis.readFloat();
			wh[i] = dis.readFloat();
		}

		for (int i = 0; i < outputSize; i++) {
			bz[i] = dis.readFloat();
			br[i] = dis.readFloat();
			bh[i] = dis.readFloat();
		}

		init();
		System.out.println("Done importing weights.");
	}

	public void setDimensions(int... dimensions) {
		System.out.println("Type: " + getType());

		inputSize = dimensions[0];
		for (int i = 1; i < dimensions.length; i++)
			inputSize *= dimensions[i];

		System.out.println("Input Size: " + inputSize);
		System.out.println("Output Size: " + outputSize);

		if (inputSize <= 0)
			throw new IllegalArgumentException("Invalid input dimensions.");
		if (outputSize <= 0)
			throw new IllegalArgumentException("Invalid output dimensions.");

		System.out.println("Activations (z/r, hc): " + hiddenActivation.getType() + ", " + activation.getType());
		System.out.println("Updater type: " + updaterType);

		wz = new float[outputSize * inputSize + outputSize * outputSize];
		wr = new float[outputSize * inputSize + outputSize * outputSize];
		wh = new float[outputSize * inputSize + outputSize * outputSize];

		weightUpdaters = new Updater[3];
		biasUpdaters = new Updater[3];
		for (int i = 0; i < 3; i++) {
			weightUpdaters[i] = updaterType.create(outputSize * inputSize + outputSize * outputSize);
			biasUpdaters[i] = updaterType.create(outputSize);
		}

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

		init();
	}

	private void init() {
		wzT = new float[outputSize * (inputSize + outputSize)];
		wrT = new float[outputSize * (inputSize + outputSize)];
		whT = new float[outputSize * (inputSize + outputSize)];

		transposeWeights();

		dWz = new float[outputSize * inputSize + outputSize * outputSize];
		dWr = new float[outputSize * inputSize + outputSize * outputSize];
		dWh = new float[outputSize * inputSize + outputSize * outputSize];

		dBz = new float[outputSize];
		dBr = new float[outputSize];
		dBh = new float[outputSize];

		xh = new LinkedList<>();
		xrh = new LinkedList<>();
		hc = new LinkedList<>();
		z = new LinkedList<>();
		r = new LinkedList<>();
		y = new LinkedList<>();
	}

	private void transposeWeights() {
		for (int i = 0; i < outputSize; i++) {
			for (int j = 0; j < inputSize + outputSize; j++) {
				int index = j + (inputSize + outputSize) * i;
				int transposedIndex = i + outputSize * j;

				wzT[transposedIndex] = wz[index];
				wrT[transposedIndex] = wr[index];
				whT[transposedIndex] = wh[index];
			}
		}
	}

	public int[] getOutputDimensions() {
		return new int[]{outputSize};
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(inputSize);
		dos.writeInt(outputSize);

		hiddenActivation.export(dos);
		activation.export(dos);
		updaterType.export(dos);

		for (int i = 0; i < 3; i++) {
			weightUpdaters[i].export(dos);
			biasUpdaters[i].export(dos);
		}

		exportParameters(outputSize * inputSize + outputSize * outputSize, wz, wr, wh, dos);
		exportParameters(outputSize, bz, br, bh, dos);
	}

	private void exportParameters(int length, float[] z, float[] r, float[] h, DataOutputStream dos) throws IOException {
		for (int i = 0; i < length; i++) {
			dos.writeFloat(z[i]);
			dos.writeFloat(r[i]);
			dos.writeFloat(h[i]);
		}
	}

	public void setMode(Mode mode) {
		this.mode = mode;

		xh.clear();
		xrh.clear();
		hc.clear();
		z.clear();
		r.clear();
		y.clear();

		if (mode == Mode.GRADIENT_CHECK) {
			if (h != null)
				Arrays.fill(h, 0.1f);
			return;
		}

		h = null;
		dh = null;

		dWz = new float[outputSize * inputSize + outputSize * outputSize];
		dWr = new float[outputSize * inputSize + outputSize * outputSize];
		dWh = new float[outputSize * inputSize + outputSize * outputSize];

		dBz = new float[outputSize];
		dBr = new float[outputSize];
		dBh = new float[outputSize];
	}

	public LayerType getType() {
		return LayerType.GRU;
	}

	public float[] forward(float[] input, int batchSize) {
		this.batchSize = batchSize;

		// checking if first in series
		if (h == null) {
			h = new float[batchSize * outputSize];

			// initializing h for gradient check
			if (mode == Mode.GRADIENT_CHECK)
				Arrays.fill(h, 0.1f);
		}

		if (mode == Mode.GRADIENT_CHECK)
			transposeWeights();

		float[] xh = new float[batchSize * (inputSize + outputSize)];
		float[] xrh = new float[batchSize * (inputSize + outputSize)];
		float[] hc = new float[batchSize * outputSize];
		float[] z = new float[batchSize * outputSize];
		float[] r = new float[batchSize * outputSize];
		float[] y = new float[batchSize * outputSize];

		// copying biases to outputs
		for (int b = 0; b < batchSize; b++) {
			System.arraycopy(bz, 0, z, outputSize * b, outputSize);
			System.arraycopy(br, 0, r, outputSize * b, outputSize);
			System.arraycopy(bh, 0, hc, outputSize * b, outputSize);
			System.arraycopy(input, inputSize * b, xh, (inputSize + outputSize) * b, inputSize);
			System.arraycopy(h, outputSize * b, xh, inputSize + (inputSize + outputSize) * b, outputSize);
		}

		z = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, batchSize,
			outputSize, inputSize + outputSize, xh, inputSize + outputSize, wzT, outputSize, z, outputSize);
		r = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, batchSize,
			outputSize, inputSize + outputSize, xh, inputSize + outputSize, wrT, outputSize, r, outputSize);

		hiddenActivation.activation(z, batchSize);
		hiddenActivation.activation(r, batchSize);

		for (int b = 0; b < batchSize; b++) {
			System.arraycopy(input, inputSize * b, xrh, (inputSize + outputSize) * b, inputSize);
			for (int i = 0; i < outputSize; i++) {
				int index = i + outputSize * b;
				xrh[(inputSize + i) + (inputSize + outputSize) * b] += r[index] * h[index];
			}
		}

		hc = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, batchSize,
			outputSize, inputSize + outputSize, xrh, inputSize + outputSize, whT, outputSize, hc, outputSize);
		activation.activation(hc, batchSize);

		for (int b = 0; b < batchSize; b++) {
			for (int i = 0; i < outputSize; i++) {
				int index = i + outputSize * b;
				h[index] = z[index] * h[index] + (1 - z[index]) * hc[index];
			}
		}

		System.arraycopy(h, 0, y, 0, batchSize * outputSize);

		// adding items to linked list for backpropagation
		this.xh.push(xh);
		this.xrh.push(xrh);
		this.hc.push(hc);
		this.z.push(z);
		this.r.push(r);
		this.y.push(y);

		return y;
	}

	public float[] backward(Cost cost, float[] target, boolean calculateDelta) {
		return backward(cost.derivative(y.pop(), target, batchSize), calculateDelta);
	}

	public float[] backward(float[] previousDelta, boolean calculateDelta) {
		float[] dx = new float[batchSize * inputSize];

		if (dh == null)
			dh = new float[batchSize * outputSize];

		// these variable represent before-activation derivatives
		float[] dr = new float[batchSize * outputSize];
		float[] dz = new float[batchSize * outputSize];
		float[] dhc = new float[batchSize * outputSize];

		// popping values to go backwards from time
		float[] xh = this.xh.pop();
		float[] xrh = this.xrh.pop();
		float[] hc = this.hc.pop();
		float[] z = this.z.pop();
		float[] r = this.r.pop();

		float[] derivative = activation.derivative(hc);
		float[] dzActivation = hiddenActivation.derivative(z);
		float[] drActivation = hiddenActivation.derivative(r);

		for (int b = 0; b < batchSize; b++) {
			for (int i = 0; i < outputSize; i++) {
				int index = i + outputSize * b;

				dh[index] += previousDelta[index];
				dhc[index] = dh[index] * (1 - z[index]) * derivative[index];
			}
		}

		float[] delta = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, batchSize,
			outputSize + inputSize, outputSize, dhc, outputSize, wh, outputSize + inputSize,
			new float[batchSize * (outputSize + inputSize)], outputSize + inputSize);

		for (int b = 0; b < batchSize; b++) {
			for (int i = 0; i < outputSize; i++) {
				int index = i + outputSize * b;
				int inputIndex = (inputSize + i) + (inputSize + outputSize) * b;

				dr[index] = xh[inputIndex] * delta[inputIndex] * drActivation[index];
				dz[index] = dh[index] * (xh[inputIndex] - hc[index]) * dzActivation[index];

				delta[inputIndex] = dh[index] * z[index] + r[index] * delta[inputIndex];
			}
		}

		delta = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, batchSize, outputSize + inputSize,
			outputSize, dz, outputSize, wz, outputSize + inputSize, delta, outputSize + inputSize);
		delta = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, batchSize, outputSize + inputSize,
			outputSize, dr, outputSize, wr, outputSize + inputSize, delta, outputSize + inputSize);

		for (int b = 0; b < batchSize; b++) {
			if (calculateDelta) {
				System.arraycopy(delta, (inputSize + outputSize) * b, dx, inputSize * b, inputSize);
			}

			System.arraycopy(delta, inputSize + (inputSize + outputSize) * b, dh, outputSize * b, outputSize);
		}

		// updating parameters
		dWz = GPU.sgemm(CLBlastTranspose.CLBlastTransposeYes, CLBlastTranspose.CLBlastTransposeNo, outputSize,
			inputSize + outputSize, batchSize, dz, outputSize, xh, inputSize + outputSize, dWz, inputSize + outputSize);
		dWr = GPU.sgemm(CLBlastTranspose.CLBlastTransposeYes, CLBlastTranspose.CLBlastTransposeNo, outputSize,
			inputSize + outputSize, batchSize, dr, outputSize, xh, inputSize + outputSize, dWr, inputSize + outputSize);
		dWh = GPU.sgemm(CLBlastTranspose.CLBlastTransposeYes, CLBlastTranspose.CLBlastTransposeNo, outputSize,
			inputSize + outputSize, batchSize, dhc, outputSize, xrh, inputSize + outputSize, dWh, inputSize + outputSize);

		for (int b = 0; b < batchSize; b++) {
			for (int i = 0; i < outputSize; i++) {
				int index = i + outputSize * b;

				dBz[i] += dz[index];
				dBr[i] += dr[index];
				dBh[i] += dhc[index];
			}
		}

		return dx;
	}

	public float[][][] getParameters() {
		return new float[][][]{{wz, dWz}, {wr, dWr}, {wh, dWh}, {bz, dBz}, {br, dBr}, {bh, dBh}};
	}

	public void update(int scale) {
		weightUpdaters[0].update(wz, dWz, scale);
		weightUpdaters[1].update(wr, dWr, scale);
		weightUpdaters[2].update(wh, dWh, scale);

		biasUpdaters[0].update(bz, dBz, scale);
		biasUpdaters[1].update(br, dBr, scale);
		biasUpdaters[2].update(bh, dBh, scale);

		transposeWeights();

		// clearing states
		h = null;
		dh = null;

		// clearing gradients
		dWz = new float[outputSize * inputSize + outputSize * outputSize];
		dWr = new float[outputSize * inputSize + outputSize * outputSize];
		dWh = new float[outputSize * inputSize + outputSize * outputSize];

		dBz = new float[outputSize];
		dBr = new float[outputSize];
		dBh = new float[outputSize];

		// clearing history
		xh.clear();
		xrh.clear();
		hc.clear();
		z.clear();
		r.clear();
		y.clear();
	}

	@SuppressWarnings({"unused", "WeakerAccess"})
	public static class Builder {
		private int hiddenSize;
		private Initializer initializer;
		private ActivationType hiddenActivation;
		private ActivationType activation;
		private UpdaterType updaterType;

		public Builder() {
			hiddenActivation = ActivationType.SIGMOID;
			activation = ActivationType.TANH;
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