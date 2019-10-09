package neuralnet.layers;

import neuralnet.optimizers.UpdaterType;

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
	 * @param updaterType the updater type
	 * @return the layer
	 * @throws IOException if there is an error reading from the file
	 */
	public static Layer fromString(DataInputStream dis, UpdaterType updaterType) throws IOException {
		switch (valueOf(dis.readUTF())) {
			case CONVOLUTIONAL:
				return new Convolutional(dis, updaterType);
			case DROPOUT:
				return new Dropout(dis);
			case FEED_FORWARD:
				return new Dense(dis, updaterType);
			case POOLING:
				return new Pooling(dis);
			case GRU:
				return new GRU(dis, updaterType);
			case INCEPTION:
				return new Inception(dis, updaterType);
			case RESIDUAL:
				return new Residual(dis, updaterType);
			case PSP:
				return new PSP(dis, updaterType);
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
	 * @throws IOException if there is an error writing to the file
	 */
	public void export(DataOutputStream dos) throws IOException {
		dos.writeUTF(toString());
	}
}
