package neuralnet.layers;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * The LayerType is used for exporting and importing layers.
 */
public enum LayerType {
	// TODO: R-CNN, GAN, INCEPTION-RESNET
	CONVOLUTIONAL, DROPOUT, FEED_FORWARD, POOLING, GRU, INCEPTION, RESIDUAL, PSP, BATCH_NORMALIZATION, INTERPOLATION, LRN, L2;

	/**
	 * Imports a layer given an input stream.
	 *
	 * @param dis the input stream
	 * @return the layer
	 */
	public static Layer fromString(DataInputStream dis) throws IOException {
		switch (valueOf(dis.readUTF())) {
			case CONVOLUTIONAL:
				return new Convolutional(dis);
			case DROPOUT:
				return new Dropout(dis);
			case FEED_FORWARD:
				return new Dense(dis);
			case POOLING:
				return new Pooling(dis);
			case GRU:
				return new GRU(dis);
			case INCEPTION:
				return new Inception(dis);
			case RESIDUAL:
				return new Residual(dis);
			case PSP:
				return new PSP(dis);
			case BATCH_NORMALIZATION:
				return new BatchNormalization(dis);
			case INTERPOLATION:
				return new Interpolation(dis);
			case LRN:
				return new LRN(dis);
			case L2:
				return new L2(dis);
			default:
				throw new IllegalArgumentException();
		}
	}

	/**
	 * Exports a layer given an output stream.
	 *
	 * @param dos the output stream
	 */
	public void export(DataOutputStream dos) throws IOException {
		dos.writeUTF(toString());
	}
}