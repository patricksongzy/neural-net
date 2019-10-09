package neuralnet;

import org.jocl.*;
import org.jocl.blast.CLBlast;
import org.jocl.blast.CLBlastLayout;

import static org.jocl.CL.*;
import static org.jocl.blast.CLBlast.*;

public class GPU {
	private static cl_context context;
	private static cl_command_queue commandQueue;
	private static Thread shutdownHook;

	static {
		CL.setExceptionsEnabled(true);
		CLBlast.setExceptionsEnabled(true);

		// initialize using first GPU
		init(0, 0);
	}

	/**
	 * Initialize with certain GPU.
	 *
	 * @param platformIndex the platform index
	 * @param deviceIndex   the device index
	 */
	@SuppressWarnings("unused, WeakerAccess")
	public static void init(int platformIndex, int deviceIndex) {
		if (commandQueue != null)
			clReleaseCommandQueue(commandQueue);
		if (context != null)
			clReleaseContext(context);

		final long deviceType = CL_DEVICE_TYPE_ALL;

		// obtain number of platforms
		int[] amounts = new int[1];
		clGetPlatformIDs(0, null, amounts);
		int platformAmount = amounts[0];

		// obtain platform
		cl_platform_id[] platforms = new cl_platform_id[platformAmount];
		clGetPlatformIDs(platformAmount, platforms, null);

		if (platformIndex > platformAmount)
			throw new IllegalArgumentException("Invalid platform.");

		cl_platform_id platform = platforms[platformIndex];
		getPlatformName(platform);

		// initialize context properties
		cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

		// obtain the number of devices for platform
		clGetDeviceIDs(platform, deviceType, 0, null, amounts);
		int deviceAmount = amounts[0];

		// obtain device
		cl_device_id[] devices = new cl_device_id[deviceAmount];
		clGetDeviceIDs(platform, deviceType, deviceAmount, devices, null);

		if (deviceIndex > deviceAmount)
			throw new IllegalArgumentException("Invalid device.");

		cl_device_id device = devices[deviceIndex];
		getDeviceName(device);

		// create a context for device
		context = clCreateContext(contextProperties, 1,
			new cl_device_id[]{device}, null, null, null);

		commandQueue = clCreateCommandQueueWithProperties(context, device, new cl_queue_properties(), null);

		if (shutdownHook != null)
			Runtime.getRuntime().removeShutdownHook(shutdownHook);

		shutdownHook = new Thread(() -> {
			clFlush(commandQueue);
			clReleaseCommandQueue(commandQueue);
			clReleaseContext(context);
		});

		Runtime.getRuntime().addShutdownHook(shutdownHook);
	}

	/**
	 * Single precision general matrix multiplication.
	 *
	 * @param aTranspose the a transpose
	 * @param bTranspose the b transpose
	 * @param m the m dimension
	 * @param n the n dimension
	 * @param k the k dimension
	 * @param a the a array
	 * @param lda the a leading dimension
	 * @param b the b array
	 * @param ldb the b leading dimension
	 * @param c the c array
	 * @param ldc the c leading dimension
	 * @return the output
	 */
	public static float[] sgemm(int aTranspose, int bTranspose, int m, int n, int k, float[] a, int lda, float[] b, int ldb,
								float[] c, int ldc) {
		cl_mem aBuffer = gpuAlloc(CL_MEM_READ_ONLY, m * k, a);
		cl_mem bBuffer = gpuAlloc(CL_MEM_READ_ONLY, k * n, b);
		cl_mem cBuffer = gpuAlloc(CL_MEM_READ_WRITE, m * n, c);

		cl_event event = new cl_event();
		CLBlastSgemm(CLBlastLayout.CLBlastLayoutRowMajor, aTranspose, bTranspose,
			m, n, k, 1, aBuffer, 0, lda, bBuffer, 0, ldb, 1, cBuffer, 0, ldc, commandQueue, event);

		// Copy the result data back to the host
		float[] result = new float[m * n];
		clEnqueueReadBuffer(commandQueue, cBuffer, true, 0, m * n
			* Sizeof.cl_float, Pointer.to(result), 0, null, null);

		// Clean up
		clReleaseMemObject(aBuffer);
		clReleaseMemObject(bBuffer);
		clReleaseMemObject(cBuffer);
		clReleaseEvent(event);

		return result;
	}

	/**
	 * Single precision general matrix multiplication that takes buffers as inputs.
	 *
	 * @param aTranspose the a transpose
	 * @param bTranspose the b transpose
	 * @param m the m dimension
	 * @param n the n dimension
	 * @param k the k dimension
	 * @param aBuffer the a buffer
	 * @param lda the a leading dimension
	 * @param bBuffer the b buffer
	 * @param ldb the b leading dimension
	 * @param c the c buffer
	 * @param ldc the c leading dimension
	 * @return the output
	 */
	public static float[] sgemm(int aTranspose, int bTranspose, int m, int n, int k, cl_mem aBuffer, int lda, cl_mem bBuffer, int ldb,
								float[] c, int ldc) {
		cl_mem cBuffer = gpuAlloc(CL_MEM_READ_WRITE, m * n, c);

		cl_event event = new cl_event();
		CLBlastSgemm(CLBlastLayout.CLBlastLayoutRowMajor, aTranspose, bTranspose,
			m, n, k, 1, aBuffer, 0, lda, bBuffer, 0, ldb, 1, cBuffer, 0, ldc, commandQueue, event);

		// Copy the result data back to the host
		float[] result = new float[m * n];
		clEnqueueReadBuffer(commandQueue, cBuffer, true, 0, m * n
			* Sizeof.cl_float, Pointer.to(result), 0, null, null);

		// Clean up
		clReleaseMemObject(cBuffer);
		clReleaseEvent(event);

		return result;
	}

	/**
	 * Single precision ax + b.
	 *
	 * @param n       the size
	 * @param alpha   the coefficient a
	 * @param x       the x array
	 * @param yBuffer the b values
	 * @return the output
	 */
	public static float[] saxpy(int n, float alpha, float[] x, cl_mem yBuffer) {
		// Create the device input buffers
		cl_mem xBuffer = gpuAlloc(CL_MEM_READ_ONLY, n, x);

		cl_event event = new cl_event();
		CLBlastSaxpy(n, alpha, xBuffer, 0, 1, yBuffer, 0, 1, commandQueue, event);

		// Copy the result data back to the host
		float[] result = new float[n];
		clEnqueueReadBuffer(commandQueue, yBuffer, true, 0, n
			* Sizeof.cl_float, Pointer.to(result), 0, null, null);

		// Clean up
		clReleaseMemObject(xBuffer);
		clReleaseMemObject(yBuffer);
		clReleaseEvent(event);

		return result;
	}

	/**
	 * Single precision ax
	 *
	 * @param n     the size
	 * @param alpha the coefficient a
	 * @param x     the x array
	 * @return the output
	 */
	public static cl_mem sscal(int n, float alpha, float[] x) {
		// Create the device input buffers
		cl_mem xBuffer = gpuAlloc(CL_MEM_READ_ONLY, n, x);

		cl_event event = new cl_event();
		CLBlastSscal(n, alpha, xBuffer, 0, 1, commandQueue, event);

		clReleaseEvent(event);

		return xBuffer;
	}

	/**
	 * Allocates float arrays to buffers.
	 *
	 * @param flags  the flags
	 * @param size   the size
	 * @param values the values
	 * @return the memory buffer
	 */
	public static cl_mem gpuAlloc(long flags, int size, float[] values) {
		cl_mem buffer = clCreateBuffer(context, flags, size
			* Sizeof.cl_float, null, null);

		clEnqueueWriteBuffer(commandQueue, buffer, true, 0, size
			* Sizeof.cl_float, Pointer.to(values), 0, null, null);

		return buffer;
	}

	private static void getPlatformName(cl_platform_id platform) {
		long size[] = new long[1];
		clGetPlatformInfo(platform, CL.CL_PLATFORM_NAME, 0, null, size);
		byte buffer[] = new byte[(int) size[0]];
		clGetPlatformInfo(platform, CL.CL_PLATFORM_NAME, buffer.length, Pointer.to(buffer), null);

		// Create a string from the buffer (excluding the trailing \0 byte)
		System.out.println("platform: " + new String(buffer, 0, buffer.length - 1));
	}

	private static void getDeviceName(cl_device_id device) {
		long size[] = new long[1];
		clGetDeviceInfo(device, CL.CL_DEVICE_NAME, 0, null, size);
		byte buffer[] = new byte[(int) size[0]];
		clGetDeviceInfo(device, CL.CL_DEVICE_NAME, buffer.length, Pointer.to(buffer), null);

		// Create a string from the buffer (excluding the trailing \0 byte)
		System.out.println("device: " + new String(buffer, 0, buffer.length - 1));
	}
}
