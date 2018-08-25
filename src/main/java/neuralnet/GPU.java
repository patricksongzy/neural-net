package neuralnet;

import org.jocl.*;
import org.jocl.blast.CLBlast;
import org.jocl.blast.CLBlastLayout;

import static org.jocl.CL.*;
import static org.jocl.blast.CLBlast.CLBlastSgemm;
import static org.jocl.blast.CLBlast.CLBlastSger;

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
			clFinish(commandQueue);
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
			clReleaseCommandQueue(commandQueue);
			clReleaseContext(context);
		}));
	}

	public static float[] sgemm(int aTranspose, int bTranspose, int m, int n, int k, float[] a, int lda, float[] b, int ldb,
								float[] c, int ldc) {
		// Create the device input buffers
		cl_mem aBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, m * k
			* Sizeof.cl_float, null, null);
		cl_mem bBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, k * n
			* Sizeof.cl_float, null, null);
		cl_mem cBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, m * n
			* Sizeof.cl_float, null, null);

		// Copy the host data to the device
		clEnqueueWriteBuffer(commandQueue, aBuffer, true, 0, m * k
			* Sizeof.cl_float, Pointer.to(a), 0, null, null);
		clEnqueueWriteBuffer(commandQueue, bBuffer, true, 0, k * n
			* Sizeof.cl_float, Pointer.to(b), 0, null, null);
		clEnqueueWriteBuffer(commandQueue, cBuffer, true, 0, m * n
			* Sizeof.cl_float, Pointer.to(c), 0, null, null);

		cl_event event = new cl_event();
		CLBlastSgemm(CLBlastLayout.CLBlastLayoutRowMajor, aTranspose, bTranspose,
			m, n, k, 1, aBuffer, 0, lda, bBuffer, 0, ldb, 1, cBuffer, 0, ldc, commandQueue, event);

		// Copy the result data back to the host
		float result[] = new float[m * n];
		clEnqueueReadBuffer(commandQueue, cBuffer, true, 0, m * n
			* Sizeof.cl_float, Pointer.to(result), 0, null, null);

		// Clean up
		clReleaseMemObject(aBuffer);
		clReleaseMemObject(bBuffer);
		clReleaseMemObject(cBuffer);
		clReleaseEvent(event);

		return result;
	}

	public static float[] sger(int m, int n, float[] x, float[] y, float[] a, int lda) {
		// Create the device input buffers
		cl_mem aBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, m
			* Sizeof.cl_float, null, null);
		cl_mem bBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, n
			* Sizeof.cl_float, null, null);
		cl_mem cBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, m * n
			* Sizeof.cl_float, null, null);

		// Copy the host data to the device
		clEnqueueWriteBuffer(commandQueue, aBuffer, true, 0, m
			* Sizeof.cl_float, Pointer.to(x), 0, null, null);
		clEnqueueWriteBuffer(commandQueue, bBuffer, true, 0, n
			* Sizeof.cl_float, Pointer.to(y), 0, null, null);
		clEnqueueWriteBuffer(commandQueue, cBuffer, true, 0, m * n
			* Sizeof.cl_float, Pointer.to(a), 0, null, null);

		cl_event event = new cl_event();
		CLBlastSger(CLBlastLayout.CLBlastLayoutRowMajor, m, n, 1, aBuffer, 0, 1, bBuffer, 0, 1,
			cBuffer, 0, lda, commandQueue, event);

		// Copy the result data back to the host
		float result[] = new float[m * n];
		clEnqueueReadBuffer(commandQueue, cBuffer, true, 0, m * n
			* Sizeof.cl_float, Pointer.to(result), 0, null, null);

		// Clean up
		clReleaseMemObject(aBuffer);
		clReleaseMemObject(bBuffer);
		clReleaseMemObject(cBuffer);
		clReleaseEvent(event);

		return result;
	}
}