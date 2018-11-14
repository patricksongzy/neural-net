package neuralnet;

import org.jocl.*;
import org.jocl.blast.CLBlast;
import org.jocl.blast.CLBlastLayout;

import static org.jocl.CL.*;
import static org.jocl.blast.CLBlast.CLBlastSaxpy;
import static org.jocl.blast.CLBlast.CLBlastSgemm;

public class GPU {
	private static cl_context context;
	private static cl_command_queue commandQueue;

	static {
		CL.setExceptionsEnabled(true);
		CLBlast.setExceptionsEnabled(true);

		// The platform, device type and device number
		// that will be used
		final int platformIndex = 0;
		final long deviceType = CL_DEVICE_TYPE_ALL;
		final int deviceIndex = 0;

		// Obtain the number of platforms
		int numPlatformsArray[] = new int[1];
		clGetPlatformIDs(0, null, numPlatformsArray);
		int numPlatforms = numPlatformsArray[0];

		// Obtain a platform ID
		cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
		clGetPlatformIDs(platforms.length, platforms, null);
		cl_platform_id platform = platforms[platformIndex];

		// Initialize the context properties
		cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

		// Obtain the number of devices for the platform
		int numDevicesArray[] = new int[1];
		clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
		int numDevices = numDevicesArray[0];

		// Obtain a device ID
		cl_device_id devices[] = new cl_device_id[numDevices];
		clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
		cl_device_id device = devices[deviceIndex];

		// Create a context for the selected device
		context = clCreateContext(contextProperties, 1,
			new cl_device_id[]{device}, null, null, null);

		// Create a command-queue
		commandQueue = clCreateCommandQueueWithProperties(context, device, new cl_queue_properties(), null);

		Runtime.getRuntime().addShutdownHook(new Thread(() -> {
			clFlush(commandQueue);
			clReleaseCommandQueue(commandQueue);
			clReleaseContext(context);
		}));
	}

	@SuppressWarnings("unused")
	public static void init(int platformIndex, int deviceIndex) {
		clReleaseCommandQueue(commandQueue);
		clReleaseContext(context);

		final long deviceType = CL_DEVICE_TYPE_ALL;

		// Obtain the number of platforms
		int numPlatformsArray[] = new int[1];
		clGetPlatformIDs(0, null, numPlatformsArray);
		int numPlatforms = numPlatformsArray[0];

		// Obtain a platform ID
		cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
		clGetPlatformIDs(platforms.length, platforms, null);
		if (platformIndex > platforms.length)
			throw new IllegalArgumentException("Invalid platform.");
		cl_platform_id platform = platforms[platformIndex];

		// Initialize the context properties
		cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

		// Obtain the number of devices for the platform
		int numDevicesArray[] = new int[1];
		clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
		int numDevices = numDevicesArray[0];

		// Obtain a device ID
		cl_device_id devices[] = new cl_device_id[numDevices];
		clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
		if (deviceIndex > devices.length)
			throw new IllegalArgumentException("Invalid device.");
		cl_device_id device = devices[deviceIndex];

		// Create a context for the selected device
		context = clCreateContext(contextProperties, 1,
			new cl_device_id[]{device}, null, null, null);

		// Create a command-queue
		commandQueue = clCreateCommandQueueWithProperties(context, device, new cl_queue_properties(), null);

		Runtime.getRuntime().addShutdownHook(new Thread(() -> {
			clReleaseCommandQueue(commandQueue);
			clReleaseContext(context);
		}));
	}

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
		gpuFree(aBuffer);
		gpuFree(bBuffer);
		gpuFree(cBuffer);
		clReleaseEvent(event);

		return result;
	}

	public static cl_mem gpuAlloc(long flags, int size, float[] values) {
		cl_mem buffer = clCreateBuffer(context, flags, size
			* Sizeof.cl_float, null, null);

		clEnqueueWriteBuffer(commandQueue, buffer, true, 0, size
			* Sizeof.cl_float, Pointer.to(values), 0, null, null);

		return buffer;
	}

	public static void gpuFree(cl_mem buffer) {
		clReleaseMemObject(buffer);
	}

	public static float[] sgemm(int aTranspose, int bTranspose, int m, int n, int k, float[] a, int lda, cl_mem b, int ldb,
								cl_mem c, int ldc) {
		// Create the device input buffers
		cl_mem aBuffer = gpuAlloc(CL_MEM_READ_ONLY, m * k, a);

		cl_event event = new cl_event();
		CLBlastSgemm(CLBlastLayout.CLBlastLayoutRowMajor, aTranspose, bTranspose,
			m, n, k, 1, aBuffer, 0, lda, b, 0, ldb, 1, c, 0, ldc, commandQueue, event);

		// Copy the result data back to the host
		float[] result = new float[m * n];
		clEnqueueReadBuffer(commandQueue, c, true, 0, m * n
			* Sizeof.cl_float, Pointer.to(result), 0, null, null);

		// Clean up
		gpuFree(aBuffer);
		clReleaseEvent(event);

		return result;
	}

	public static float[] saxpy(int n, float alpha, float[] x, float[] y) {
		// Create the device input buffers
		cl_mem xBuffer = gpuAlloc(CL_MEM_READ_ONLY, n, x);
		cl_mem yBuffer = gpuAlloc(CL_MEM_READ_ONLY, n, y);

		cl_event event = new cl_event();
		CLBlastSaxpy(n, alpha, xBuffer, 0, 1, yBuffer, 0, 1, commandQueue, event);

		// Copy the result data back to the host
		float[] result = new float[n];
		clEnqueueReadBuffer(commandQueue, yBuffer, true, 0, n
			* Sizeof.cl_float, Pointer.to(result), 0, null, null);

		// Clean up
		gpuFree(xBuffer);
		gpuFree(yBuffer);
		clReleaseEvent(event);

		return result;
	}
}