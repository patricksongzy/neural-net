package neuralnet.layers;

import neuralnet.costs.Cost;
import neuralnet.optimizers.UpdaterType;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

@SuppressWarnings("FieldCanBeLocal")
public class Interpolation implements Layer {
	private int zoomFactor;
	private int batchSize;
	private int height, width, depth;
	private int outputHeight, outputWidth;
	private float[] output;

	private Interpolation(int outputHeight, int outputWidth) {
		this.outputHeight = outputHeight;
		this.outputWidth = outputWidth;
	}

	private Interpolation(int zoomFactor) {
		this.zoomFactor = zoomFactor;
	}

	Interpolation(DataInputStream dis) throws IOException {
		depth = dis.readInt();
		height = dis.readInt();
		width = dis.readInt();

		outputHeight = dis.readInt();
		outputWidth = dis.readInt();
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(depth);
		dos.writeInt(height);
		dos.writeInt(width);

		dos.writeInt(outputHeight);
		dos.writeInt(outputWidth);
	}

	public void setDimensions(int dimensions[], UpdaterType updaterType) {
		this.depth = dimensions[0];
		this.height = dimensions[1];
		this.width = dimensions[2];

		if (zoomFactor > 0) {
			outputHeight = height + (height - 1) * (zoomFactor - 1);
			outputWidth = width + (width - 1) * (zoomFactor - 1);
		}
	}

	public void setMode(Mode mode) {
	}

	public LayerType getType() {
		return LayerType.INTERPOLATION;
	}

	public float[] forward(float[] input, int batchSize) {
		this.batchSize = batchSize;

		output = new float[batchSize * depth * outputHeight * outputWidth];
		if (height == 1 && width == 1) {
			for (int b = 0; b < batchSize; b++) {
				for (int k = 0; k < depth; k++) {
					float value = input[k + depth * b];

					for (int y = 0; y < outputHeight; y++) {
						for (int x = 0; x < outputWidth; x++) {
							output[x + outputWidth * (y + outputHeight * (k + depth * b))] = value;
						}
					}
				}
			}

			return output;
		}

		float xRatio = ((float) width) / outputWidth;
		float yRatio = ((float) height) / outputHeight;
		for (int y = 1; y <= outputHeight; y++) {
			float yi = (yRatio * y) + (0.5f * (1 - 1.0f / ((float) outputHeight / height)));
			if (yi < 1)
				yi = 1;
			if (yi > height - 1e-5f)
				yi = height - 1e-5f;
			int y1 = (int) yi;
			int y2 = y1 + 1;

			for (int x = 1; x <= outputWidth; x++) {
				float xi = (xRatio * x) + (0.5f * (1 - 1.0f / ((float) outputWidth / width)));
				if (xi < 1)
					xi = 1;
				if (xi > width - 1e-5f)
					xi = width - 1e-5f;
				int x1 = (int) xi;
				int x2 = x1 + 1;


				float pw0 = (y2 - yi) * (x2 - xi);
				float pw1 = (y2 - yi) * (xi - x1);
				float pw2 = (x2 - xi) * (yi - y1);
				float pw3 = (yi - y1) * (xi - x1);

				for (int b = 0; b < batchSize; b++) {
					for (int k = 0; k < depth; k++) {
						float c0 = input[(x1 - 1) + width * ((y1 - 1) + height * (k + depth * b))];
						float c1 = input[(x2 - 1) + width * ((y1 - 1) + height * (k + depth * b))];
						float c2 = input[(x1 - 1) + width * ((y2 - 1) + height * (k + depth * b))];
						float c3 = input[(x2 - 1) + width * ((y2 - 1) + height * (k + depth * b))];

						output[(x - 1) + outputWidth * ((y - 1) + outputHeight * (k + depth * b))] =
							c0 * pw0 + c1 * pw1 + c2 * pw2 + c3 * pw3;
					}
				}
			}
		}

		return output;
	}

	public float[] backward(Cost cost, float[] target, boolean calculateDelta) {
		return new float[batchSize * depth * height * width];
	}

	public float[] backward(float[] previousDelta, boolean calculateDelta) {
		return new float[batchSize * depth * height * width];
	}

	public int[] getOutputDimensions() {
		return new int[]{depth, outputHeight, outputWidth};
	}

	public float[][][] getParameters() {
		return new float[0][][];
	}

	public void update(int length) {
	}

	/**
	 * Builder for Interpolation branch2.
	 */
	@SuppressWarnings({"unused", "WeakerAccess"})
	public static class Builder {
		private int zoomFactor;
		private int outputHeight, outputWidth;

		public Builder zoomFactor(int zoomFactor) {
			this.zoomFactor = zoomFactor;
			return this;
		}

		public Builder outputHeight(int outputHeight) {
			this.outputHeight = outputHeight;
			return this;
		}

		public Builder outputWidth(int outputWidth) {
			this.outputWidth = outputWidth;
			return this;
		}

		public Interpolation build() {
			if (outputHeight == 0 || outputWidth == 0)
				return new Interpolation(zoomFactor);
			return new Interpolation(outputHeight, outputWidth);
		}
	}
}
