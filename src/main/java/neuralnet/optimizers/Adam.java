package neuralnet.optimizers;

import neuralnet.GPU;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.stream.IntStream;

/**
 * The ADAM updater stores <code>m</code> and <code>v</code> as parameters for each layer. Although other updaters may outperform ADAM,
 * ADAM is commonly used, due to the fact that it works well without adjusting any hyper-parameters.
 */
public class Adam implements Updater {
	private static float beta1 = 0.9f;
	private static float beta2 = 0.999f;
	private static float epsilon = 1e-8f;
	private static float learningRate = 0.0001f;

	private int size;
	private int t = 0;
	private float[] m, v;

	public Adam(int size) {
		this.size = size;

		m = new float[size];
		v = new float[size];
	}

	public Adam(DataInputStream dis) throws IOException {
		size = dis.readInt();
		t = dis.readInt();

		m = new float[size];
		v = new float[size];

		for (int i = 0; i < size; i++) {
			m[i] = dis.readFloat();
			v[i] = dis.readFloat();
		}
	}

	/**
	 * The learning rate reduces updates overshooting.
	 *
	 * @param learningRate the learning rate
	 */
	@SuppressWarnings("unused")
	public static void init(float learningRate) {
		Adam.learningRate = learningRate;
	}

	@SuppressWarnings("unused")
	public static void init(float beta1, float beta2, float epsilon, float learningRate) {
		Adam.beta1 = beta1;
		Adam.beta2 = beta2;
		Adam.epsilon = epsilon;
		Adam.learningRate = learningRate;
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

	public float[] update(float[] gradient) {
		t++;
		float[] update = new float[size];

		float b1 = 1 - beta1;
		float b2 = 1 - beta2;

		float mt = 1 - (float) Math.pow(beta1, t);
		float vt = 1 - (float) Math.pow(beta2, t);

		m = GPU.saxpy(size, beta1, m, GPU.saxpy(size, b1, gradient, new float[size]));

		float[] bv = GPU.saxpy(size, beta2, v, new float[size]);

		IntStream.range(0, size).parallel().forEach(i -> {
			v[i] = bv[i] + b2 * gradient[i] * gradient[i];
			update[i] = (float) -((learningRate * (m[i] / mt)) / (Math.sqrt(v[i] / vt) + epsilon));
		});

		return update;
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(size);
		dos.writeInt(t);

		for (int i = 0; i < size; i++) {
			dos.writeFloat(m[i]);
			dos.writeFloat(v[i]);
		}
	}
}