package neuralnet.layers;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * The LayerType is used for exporting and importing layers.
 */
public enum LayerType {
	CONVOLUTIONAL, DROPOUT, FEED_FORWARD, POOLING, GRU, INCEPTION, RESIDUAL, BATCH_NORMALIZATION, SCALE, LRN, L2;

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
				return new FeedForward(dis);
			case POOLING:
				return new Pooling(dis);
			case GRU:
				return new GRU(dis);
			case INCEPTION:
				return new Inception(dis);
			case RESIDUAL:
				return new Residual(dis);
			case BATCH_NORMALIZATION:
				return new BatchNormalization(dis);
			case SCALE:
				return new Scale(dis);
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