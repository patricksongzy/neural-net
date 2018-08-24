package neuralnet.layers;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class PoolingTest {
	@Test
	void forward() {
		Pooling pooling = new Pooling.Builder().downsampleSize(2).downsampleStride(2).build();
		pooling.setDimensions(4, 4, 1);
		float[][] input = new float[][]{{1, 1, 2, 4, 5, 6, 7, 8, 3, 2, 1, 0, 1, 2, 3, 4}};
		assertArrayEquals(new float[]{6, 8, 3, 4}, pooling.forward(input)[0]);
	}

	@Test
	void backward() {
		Pooling pooling = new Pooling.Builder().downsampleSize(2).downsampleStride(2).build();
		pooling.setDimensions(4, 4, 1);
		pooling.forward(new float[][]{{1, 1, 2, 4, 5, 6, 7, 8, 3, 2, 1, 0, 1, 2, 3, 4}});
		assertArrayEquals(new float[]{0, 0, 0, 0, 0, 1, 0, 2, 3, 0, 0, 0, 0, 0, 0, 4}, pooling.backward(new float[][]{{1, 2, 3, 4}})[0]);
	}
}