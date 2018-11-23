package neuralnet.optimizers;

import neuralnet.GPU;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.stream.IntStream;

/**
 * The AMSGrad updater stores <code>m</code> and <code>v</code> as parameters for each layer. Although other updaters may outperform
 * AMSGrad,
 * AMSGrad is commonly used, due to the fact that it works well without adjusting any hyper-parameters.
 */
@SuppressWarnings("unused")
public class AMSGrad implements Updater {
	private static float beta1 = 0.9f;
	private static float beta2 = 0.999f;
	private static float epsilon = 1e-8f;
	private static float decay = 0.001f;
	private static float learningRate = 0.0001f;

	private int size;
	private float[] m, v;

	AMSGrad(int size) {
		this.size = size;

		m = new float[size];
		v = new float[size];
	}

	AMSGrad(DataInputStream dis) throws IOException {
		size = dis.readInt();

		m = new float[size];
		v = new float[size];

		for (int i = 0; i < size; i++) {
			m[i] = dis.readFloat();
			v[i] = dis.readFloat();
		}
	}

	/**
	 * Initializes parameters. Parameters are ordered as follows: <code>beta1, beta2, epsilon</code>
	 *
	 * @param parameters the parameters
	 */
	@SuppressWarnings("WeakerAccess")
	public static void init(float[] parameters) {
		if (parameters.length != 3)
			throw new IllegalArgumentException("Invalid parameters.");

		AMSGrad.beta1 = parameters[0];
		AMSGrad.beta2 = parameters[1];
		AMSGrad.epsilon = parameters[2];
	}

	/**
	 * The learning rate reduces updates overshooting.
	 *
	 * @param learningRate the learning rate
	 */
	static void setLearningRate(float learningRate) {
		AMSGrad.learningRate = learningRate;
	}

	static void setDecay(float decay) {
		AMSGrad.decay = decay;
	}

	/**
	 * Imports parameters from an input stream.
	 *
	 * @param dis the input stream
	 */
	static void importParameters(DataInputStream dis) throws IOException {
		beta1 = dis.readFloat();
		beta2 = dis.readFloat();
		epsilon = dis.readFloat();
		learningRate = dis.readFloat();
	}

	/**
	 * Exports parameters to an output stream.
	 *
	 * @param dos the output stream
	 */
	static void exportParameters(DataOutputStream dos) throws IOException {
		dos.writeFloat(beta1);
		dos.writeFloat(beta2);
		dos.writeFloat(epsilon);
		dos.writeFloat(learningRate);
	}

	public void update(float[] parameters, float[] gradient, int scale) {
		float b1 = 1 - beta1;
		float b2 = 1 - beta2;

		m = GPU.saxpy(size, beta1, m, GPU.sscal(size, b1, gradient));

		IntStream.range(0, size).parallel().forEach(i -> {
			v[i] = Math.max(v[i], v[i] * beta2 + b2 * gradient[i] * gradient[i]);
			parameters[i] -= learningRate * (m[i] / (scale * (Math.sqrt(v[i]) + epsilon)) + decay * parameters[i]);
		});
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(size);

		for (int i = 0; i < size; i++) {
			dos.writeFloat(m[i]);
			dos.writeFloat(v[i]);
		}
	}
}