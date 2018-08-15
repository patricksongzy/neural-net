package main.neuralnet.layers;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * The LayerType is used for exporting and importing layers.
 */
public enum LayerType {
	CONVOLUTIONAL, DROPOUT, FEED_FORWARD, SAMPLING, GRU;

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
			case SAMPLING:
				return new Sampling(dis);
			case GRU:
				return new GRU(dis);
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