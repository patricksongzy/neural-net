package neuralnet.layers;

import neuralnet.costs.Cost;
import neuralnet.optimizers.UpdaterType;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class LRN implements Layer {
	private int batchSize;
	private int n;
	private float k, alpha, beta;
	private int height, width, depth;
	private float[] output;

	private LRN(int n, float k, float alpha, float beta) {
		this.n = n;
		this.k = k;
		this.alpha = alpha;
		this.beta = beta;
	}

	LRN(DataInputStream dis) throws IOException {
		height = dis.readInt();
		width = dis.readInt();
		depth = dis.readInt();

		n = dis.readInt();
		k = dis.readFloat();
		alpha = dis.readFloat();
		beta = dis.readFloat();
	}

	public void setDimensions(int[] dimensions, UpdaterType updaterType) {
		if (dimensions.length < 3)
			throw new IllegalArgumentException();

		this.height = dimensions[0];
		this.width = dimensions[1];
		this.depth = dimensions[2];
	}

	public void setMode(Layer.Mode mode) {
	}

	public LayerType getType() {
		return LayerType.LRN;
	}

	public float[] forward(float[] input, int batchSize) {
		this.batchSize = batchSize;
		output = new float[input.length];

		for (int b = 0; b < batchSize; b++) {
			for (int h = 0; h < height; h++) {
				for (int w = 0; w < width; w++) {
					for (int d = 0; d < depth; d++) {
						float sum = 0;

						for (int i = Math.max(0, d - n / 2); i <= Math.min(depth - 1, i + n / 2); i++) {
							sum += Math.pow(input[w + width * (h + height * (i + depth * b))], 2);
						}

						int index = w + width * (h + height * (d + depth * b));
						output[index] = input[index] / (float) Math.pow(k + (alpha / n) * sum, beta);
					}
				}
			}
		}

		return output;
	}

	// TODO: implement
	public float[] backward(Cost cost, float[] target, boolean calculateDelta) {
		return cost.derivative(output, target, batchSize);
	}

	// TODO: implement
	public float[] backward(float[] previousDelta, boolean calculateDelta) {
		return new float[batchSize * height * width * depth];
	}

	public int[] getOutputDimensions() {
		return new int[]{height, width, depth};
	}

	public float[][][] getParameters() {
		return new float[0][][];
	}

	public void update(int length) {
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(height);
		dos.writeInt(width);
		dos.writeInt(depth);

		dos.writeInt(n);
		dos.writeFloat(k);
		dos.writeFloat(alpha);
		dos.writeFloat(beta);
	}

	/**
	 * Builder for LRN layers.
	 */
	@SuppressWarnings({"unused", "WeakerAccess"})
	public static class Builder {
		private int n;
		private float alpha, beta, k;

		public Builder n(int n) {
			this.n = n;
			return this;
		}

		public Builder k(float k) {
			this.k = k;
			return this;
		}

		public Builder alpha(float alpha) {
			this.alpha = alpha;
			return this;
		}

		public Builder beta(float beta) {
			this.beta = beta;
			return this;
		}

		public LRN build() {
			return new LRN(n, k, alpha, beta);
		}
	}
}