package neuralnet.optimizers;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * The UpdaterType is used for exporting and importing neural networks, and for repeatedly creating instances of an updater.
 */
@SuppressWarnings("SameReturnValue")
public enum UpdaterType {
	ADAM;

	/**
	 * Creates an UpdaterType, given an input stream.
	 *
	 * @param dis the input stream
	 * @return the UpdaterType
	 */
	public static UpdaterType fromString(DataInputStream dis) throws IOException {
		switch (valueOf(dis.readUTF())) {
			default:
				Adam.importParameters(dis);
				return ADAM;
		}
	}

	/**
	 * Creates an instance, given the current UpdaterType.
	 *
	 * @return an instance of the current UpdaterType
	 */
	public Updater create(int size) {
		switch (this) {
			default:
				return new Adam(size);
		}
	}

	/**
	 * Creates an Updater, given the current type, then imports it's parameters given an input stream.
	 *
	 * @param dis the input stream
	 * @return the updater
	 */
	public Updater create(DataInputStream dis) throws IOException {
		switch (this) {
			default:
				return new Adam(dis);
		}
	}

	/**
	 * Exports the UpdaterType.
	 *
	 * @param dos the output stream
	 */
	public void export(DataOutputStream dos) throws IOException {
		dos.writeUTF(toString());

		switch (this) {
			default:
				Adam.exportParameters(dos);
		}
	}
}