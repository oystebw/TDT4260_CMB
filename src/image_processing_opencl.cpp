#pragma GCC optimize ("Ofast")
// __attribute__((optimize("prefetch-loop-arrays")))

#include <iostream>
#include <fstream>
#include <cstddef>
#include <cmath>
#include <vector>
#include <string>
#include <cassert>
#ifdef ON_VM
#include <CL/cl2.hpp>
#else
#include <CL/cl.hpp>
#endif
extern "C"{
    #include "ppm.h"
}

#include <math.h>
#include <stdlib.h>
#include <omp.h>

#define BLOCKSIZE 8
#define PF_OFFSET 128

typedef float v4Accurate __attribute__((vector_size(16)));
typedef __uint32_t v4Int __attribute__((vector_size(16)));

PPMPixel result_data[1920 * 1200];
v4Accurate one[1920 * 1200];
v4Accurate two[1920 * 1200];
v4Accurate scratch[1920 * 1200];

using namespace std;
using namespace cl;

void errorAndExit(string error_message) {
    cerr << error_message << endl;
    exit(1);
}

// class OpenClBlur{
// public:
//     OpenClBlur() {
        
//         // choose a platform containing this string if available
//         string preferred_platform = "Intel";
//         // select platform
//         std::vector<Platform> all_platforms;
//         Platform::get(&all_platforms);
//         if (all_platforms.empty()){
//             errorAndExit("No platforms found.\n");
//         }
//         Platform default_platform = all_platforms[0];
//         for(const auto& platform: all_platforms){
//             cerr << "Found platform: " << platform.getInfo<CL_PLATFORM_NAME>() << endl;
//             if(platform.getInfo<CL_PLATFORM_NAME>().find(preferred_platform)!=-1){
//                 default_platform = platform;
//             }
//         }
//         cerr << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

//         // select device
//         std::vector<Device> all_devices;
//         default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
//         if (all_devices.empty()){
//             errorAndExit("No devices found.");
//         }
//         Device default_device = all_devices[0];
//         for(const auto& device: all_devices){
//             cerr << "Found device: " << device.getInfo<CL_DEVICE_NAME>() << endl;
//         }
//         cerr << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";
//         device = default_device;


//         for(const auto& foundDevice : all_devices) {
//             if(foundDevice.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) {
//                 device = foundDevice;
//                 break;
//             }
//         }

//         // create context
//         context = Context({device});

//         // create command queue with profiling enabled
//         queue = CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

//         // load opencl program and compile it
//         Program::Sources sources;
//         ifstream ifs("image_processing_opencl.cl");
//         if (ifs.fail()){
//             errorAndExit("Failed to open image_processing_opencl.");
//         }
//         string kernel_source { istreambuf_iterator<char>(ifs), istreambuf_iterator<char>() };
//         sources.push_back({kernel_source.c_str(), kernel_source.size()});
//         program = Program(context, sources);
//         if (program.build({device}) != CL_SUCCESS){
//             errorAndExit("Error building program!");
//         }

//         kernelHorizontal = Kernel(program, "kernelHorizontal");
//         kernelVertical = Kernel(program, "kernelVertical");
//     }
//     AccurateImage* blur(AccurateImage* image, int size){

//         std::size_t bufferSize = image->x * image->y * sizeof(AccuratePixel);
//         Buffer buffer1(context, CL_MEM_READ_WRITE|CL_MEM_READ_WRITE, bufferSize);
//         Buffer buffer2(context, CL_MEM_READ_WRITE|CL_MEM_READ_WRITE, bufferSize);

//         events.emplace_back(make_pair("copy buffer to image", Event()));

//         queue.enqueueWriteBuffer(buffer1, false, 0, bufferSize, image->data, nullptr, &events.back().second);

//         blurIteration(image, buffer1, buffer2, size);

//         AccurateImage* result = copyAccurateImage(image, true, false);

//         events.emplace_back(make_pair("map buffer in memory", Event()));

//         result->data = (AccuratePixel*)queue.enqueueMapBuffer(buffer1, CL_FALSE, CL_MAP_READ, 0, bufferSize, nullptr, &events.back().second);
//         return result;
//     }
//     void finish(){
//         // finish execution and print events
//         queue.finish();
//         for(auto [s, e]: events){
//             printEvent(s, e);
//         }
//         events.clear();
//     }

// private:
//     Device device;
//     Context context;
//     CommandQueue queue;
//     Program program;
//     Kernel kernelHorizontal;
//     Kernel kernelVertical;
//     std::vector<pair<string, Event>> events;
//     void printEvent(string s, Event& evt){

//         // ensure the event has completed
//         assert(evt.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>()==CL_COMPLETE);
//         cl_ulong queued = evt.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
//         cl_ulong submit = evt.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
//         cl_ulong start = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
//         cl_ulong end = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
//         cerr << "event: " << s << endl;
//         cerr << "queue time: " << submit-queued << "ns" << endl;
//         cerr << "run time: " << end-start << "ns" << endl;
//         cerr << "total time: " << end-queued << "ns" << endl;
//     }
//     void blurIteration(AccurateImage* image, Buffer& src, Buffer& dst, cl_int size){

//         // create Event for profiling
//         events.emplace_back(make_pair("kernelHorizontal", Event()));
//         // set call arguments
//         kernelHorizontal.setArg(0, src);
//         kernelHorizontal.setArg(1, dst);
//         kernelHorizontal.setArg(2, image->x);
//         kernelHorizontal.setArg(3, size);
//         // call 2D kernel
//         queue.enqueueNDRangeKernel(
//                 kernelHorizontal, // kernel to queue
//                 NullRange, // use no offset
//                 NDRange(image->y), // 1D kernel
//                 NullRange, // use no local range
//                 nullptr, // we use the queue in sequential mode so we don't have to specify Events that need to finish before
//                 &events.back().second // Event to use for profiling
//         );

//         events.emplace_back(make_pair("kernelVertical", Event()));
//         // set call arguments
//         kernelVertical.setArg(0, dst);
//         kernelVertical.setArg(1, src);
//         kernelVertical.setArg(2, image->y);
//         kernelVertical.setArg(3, size);
//         // call 2D kernel
//         queue.enqueueNDRangeKernel(
//                 kernelVertical, // kernel to queue
//                 NullRange, // use no offset
//                 NDRange(image->x), // 1D kernel
//                 NullRange, // use no local range
//                 nullptr, // we use the queue in sequential mode so we don't have to specify Events that need to finish before
//                 &events.back().second // Event to use for profiling
//         );
//     }
// };



// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

void blurIterationHorizontalFirst(const PPMPixel*  in, v4Accurate*  out, const int size, const int width, const int height) {
	const v4Accurate divisor = (v4Accurate){1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1)};
	// #pragma omp parallel for schedule(dynamic, 2) num_threads(8)
	for(int y = 0; y < height; ++y) {
		const int yWidth = y * width;

		v4Int sum = {0, 0, 0, 0};

		for(int x = 0; x <= size; ++x) {
			sum += (v4Int){in[yWidth + x].red, in[yWidth + x].green, in[yWidth + x].blue, 0};
		}

		out[yWidth + 0] = (v4Accurate){sum[0], sum[1], sum[2], sum[3]} / (v4Accurate){size + 1, size + 1, size + 1, size + 1};

		for(int x = 1; x <= size; ++x) {
			sum += (v4Int){in[yWidth + x + size].red, in[yWidth + x + size].green, in[yWidth + x + size].blue, 0};
			out[yWidth + x] = (v4Accurate){sum[0], sum[1], sum[2], sum[3]} / (v4Accurate){size + x + 1, size + x + 1, size + x + 1, size + x + 1};
		}

		// #pragma GCC unroll 16
		for(int xx = size + 1; xx < width - size; xx += 16) {
			// __builtin_prefetch(&in[yWidth + xx + size] + PF_OFFSET, 0, 1);
			for(int x = xx; x < xx + 16 && x < width - size; ++x) {
				sum -= (v4Int){in[yWidth + x - size - 1].red, in[yWidth + x - size - 1].green, in[yWidth + x - size - 1].blue, 0};
				sum += (v4Int){in[yWidth + x + size].red, in[yWidth + x + size].green, in[yWidth + x + size].blue, 0};
				out[yWidth + x] = (v4Accurate){sum[0], sum[1], sum[2], sum[3]} * divisor;
			}
		}

		for(int x = width - size; x < width; ++x) {
			sum -= (v4Int){in[yWidth + x - size - 1].red, in[yWidth + x - size - 1].green, in[yWidth + x - size - 1].blue, 0};
			out[yWidth + x] = (v4Accurate){sum[0], sum[1], sum[2], sum[3]} / (v4Accurate){size + width - x, size + width - x, size + width - x, size + width - x};
		}
	}
}

void blurIterationHorizontal(v4Accurate* in, v4Accurate* out, const int size, const int width, const int height) {
	const v4Accurate divisor = (v4Accurate){1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1)};
	// #pragma omp parallel for schedule(dynamic, 2) num_threads(8)
	for(int y = 0; y < height; ++y) {
		const int yWidth = y * width;
		for(int iteration = 0; iteration < 3; ++iteration) {
			
			v4Accurate sum = {0.0, 0.0, 0.0, 0.0};

			for(int x = 0; x <= size; ++x) {
				sum += in[yWidth + x];
			}

			out[yWidth + 0] = sum / (v4Accurate){size + 1, size + 1, size + 1, size + 1};

			for(int x = 1; x <= size; ++x) {
				sum += in[yWidth + x + size];
				out[yWidth + x] = sum / (v4Accurate){size + x + 1, size + x + 1, size + x + 1, size + x + 1};
			}

			// #pragma GCC unroll 16
			for(int xx = size + 1; xx < width - size; xx += 4) {
				// __builtin_prefetch(&in[yWidth + xx + size] + PF_OFFSET, 0, 2);
				for(int x = xx; x < xx + 4 && x < width - size; ++x) {
					sum -= in[yWidth + x - size - 1];
					sum += in[yWidth + x + size];
					out[yWidth + x] = sum * divisor;
				}
			}

			for(int x = width - size; x < width; ++x) {
				sum -= in[yWidth + x - size - 1];
				out[yWidth + x] = sum / (v4Accurate){size + width - x, size + width - x, size + width - x, size + width - x};
			}

			// swap in and out
			v4Accurate* tmp = in;
			in = out;
			out = tmp;
		}
		// swap in and out
		v4Accurate* tmp = in;
		in = out;
		out = tmp;
	}
}

void blurIterationHorizontalTranspose(const v4Accurate*  in, v4Accurate*  out, const int size, const int width, const int height) {
	const v4Accurate divisor = (v4Accurate){1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1)};
	// #pragma omp parallel for schedule(dynamic, 2) num_threads(8)
	for(int y = 0; y < height; ++y) {
		const int yWidth = y * width;

		v4Accurate sum = {0.0, 0.0, 0.0, 0.0};

		for(int x = 0; x <= size; ++x) {
			sum += in[yWidth + x];
		}

		out[0 * height + y] = sum / (v4Accurate){size + 1, size + 1, size + 1, size + 1};

		for(int x = 1; x <= size; ++x) {
			sum += in[yWidth + x + size];
			out[x * height + y] = sum / (v4Accurate){size + x + 1, size + x + 1, size + x + 1, size + x + 1};
		}

		// #pragma GCC unroll 16
		for(int xx = size + 1; xx < width - size; xx += 4) {
			// __builtin_prefetch(&in[yWidth + xx + size] + PF_OFFSET, 0, 2);
			for(int x = xx; x < xx + 4 && x < width - size; ++x) {				
				sum -= in[yWidth + x - size - 1];
				sum += in[yWidth + x + size];
				out[x * height + y] = sum * divisor;
			}
		}

		for(int x = width - size; x < width; ++x) {
			sum -= in[yWidth + x - size - 1];
			out[x * height + y] = sum / (v4Accurate){size + width - x, size + width - x, size + width - x, size + width - x};
		}
	}
}

void blurIterationVertical(v4Accurate* in, v4Accurate* out, const int size, const int width, const int height) {
	const v4Accurate divisor = (v4Accurate){1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1)};
	// #pragma omp parallel for schedule(dynamic, 2) num_threads(8)
	for(int x = 0; x < width; ++x) {
		const int xHeight = x * height;
		for(int iteration = 0; iteration < 5; ++iteration) {
			v4Accurate sum = {0.0, 0.0, 0.0, 0.0};

			for(int y = 0; y <= size; ++y) {
				sum += in[xHeight + y];
			}

			out[xHeight + 0] = sum / (v4Accurate){size + 1, size + 1, size + 1, size + 1};

			for(int y = 1; y <= size; ++y) {
				sum += in[xHeight + y + size];
				out[xHeight + y] = sum / (v4Accurate){y + size + 1, y + size + 1, y + size + 1, y + size + 1};
			}

			// #pragma GCC unroll 16
			for(int yy = size + 1; yy < height - size; yy += 4) {
				// __builtin_prefetch(&in[xHeight + yy + size] + PF_OFFSET, 0, 2);
				for(int y = yy; y < yy + 4 && y < height - size; ++y) {
					sum -= in[xHeight + y - size - 1];
					sum += in[xHeight + y + size];
					out[xHeight + y] = sum * divisor;
				}
			}

			for(int y = height - size; y < height; ++y) {
				sum -= in[xHeight + y - size - 1];
				out[xHeight + y] = sum / (v4Accurate){size + height - y, size + height - y, size + height - y, size + height - y};
			}
			// swap
			v4Accurate* tmp = in;
			in = out;
			out = tmp;
		}
		// swap
		v4Accurate* tmp = in;
		in = out;
		out = tmp;
	}
}

void imageDifference(PPMPixel*  imageOut, const v4Accurate*  small, const v4Accurate*  large, const int width, const int height) {
	
	// #pragma omp parallel for schedule(dynamic, 2) num_threads(8)
	for(int yy = 0; yy < height; yy += BLOCKSIZE) {
		for(int xx = 0; xx < width; xx += BLOCKSIZE) {
			for(int x = xx; x < xx + BLOCKSIZE; ++x) {
				const int xHeight = x * height;
				// __builtin_prefetch(&large[xHeight + height + yy], 0, 3);
				// __builtin_prefetch(&small[xHeight + height + yy], 0, 3);
				// #pragma GGC unroll 16
				for(int y = yy; y < yy + BLOCKSIZE; ++y) {
					v4Accurate diff = large[xHeight + y] - small[xHeight + y];
					imageOut[y * width + x] = (PPMPixel){
						diff[0] = diff[0] < 0.0 ? diff[0] + 257.0 : diff[0],
						diff[1] = diff[1] < 0.0 ? diff[1] + 257.0 : diff[1],
						diff[2] = diff[2] < 0.0 ? diff[2] + 257.0 : diff[2]
					};
				}
			}
		}
	}
}

int main(int argc, char** argv) {

    const PPMImage* image = (argc > 1) ? readPPM("flower.ppm") : readStreamPPM(stdin);

	const int width = image->x;
	const int height = image->y;
	const int size = width * height;

	PPMImage*  result = (PPMImage* )malloc(sizeof(PPMImage*));
	result->x = width;
	result->y = height;
	result->data = result_data;

	blurIterationHorizontalFirst(image->data,  scratch,  2, width, height);
	blurIterationHorizontal( scratch,  one,  2, width, height);
	blurIterationHorizontalTranspose( one,  scratch,  2, width, height);
	blurIterationVertical( scratch,  one,  2, width, height);

	blurIterationHorizontalFirst(image->data,  scratch,  3, width, height);
	blurIterationHorizontal( scratch,  two,  3, width, height);
	blurIterationHorizontalTranspose( two,  scratch,  3, width, height);
	blurIterationVertical( scratch,  two,  3, width, height);

	imageDifference(result_data,  one,  two, width, height);
	(argc > 1) ? writePPM("flower_tiny.ppm", result) : writeStreamPPM(stdout, result);

	blurIterationHorizontalFirst(image->data,  scratch,  5, width, height);
	blurIterationHorizontal( scratch,  one,  5, width, height);
	blurIterationHorizontalTranspose( one,  scratch,  5, width, height);
	blurIterationVertical( scratch,  one,  5, width, height);
	imageDifference(result_data,  two,  one, width, height);
	(argc > 1) ? writePPM("flower_small.ppm", result) : writeStreamPPM(stdout, result);

	blurIterationHorizontalFirst(image->data,  scratch,  8, width, height);
	blurIterationHorizontal( scratch,  two,  8, width, height);
	blurIterationHorizontalTranspose( two,  scratch,  8, width, height);
	blurIterationVertical( scratch,  two,  8, width, height);
	imageDifference(result_data,  one,  two, width, height);
	(argc > 1) ? writePPM("flower_medium.ppm", result) : writeStreamPPM(stdout, result);
}