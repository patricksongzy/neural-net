package neuralnet.optimizers;

import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Updaters calculate deltas to update parameters.
 */
public interface Updater {
	/**
	 * Calculates the delta for the parameters.
	 *
	 * @param parameters the parameters
	 * @param gradient the parameter gradient
	 * @param scale the scale
	 */
	void update(float[] parameters, float[] gradient, int scale);

	/**
	 * Export the parameter specific parameters to an output stream.
	 *
	 * @param dos the output stream
	 */
	void export(DataOutputStream dos) throws IOException;
}