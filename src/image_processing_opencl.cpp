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


using namespace std;
using namespace cl;

typedef struct {
    float red, green, blue;
} AccuratePixel;

typedef struct {
    int x, y;
    AccuratePixel *data;
} AccurateImage;

void errorAndExit(string error_message) {
    cerr << error_message << endl;
    exit(1);
}
AccurateImage* convertToAccurateImage(PPMImage* image) {
    const int width = image->x;
    const int height = image->y;
    const int size = width * height;
    AccurateImage* imageAccurate = (AccurateImage*)malloc(sizeof(AccurateImage));
    imageAccurate->x = width;
    imageAccurate->y = height;
    imageAccurate->data = (AccuratePixel*)malloc(size * sizeof(AccuratePixel));
    for(int i = 0; i < image->x * image->y; i++) {
        imageAccurate->data[i].red   = (float) image->data[i].red;
        imageAccurate->data[i].green = (float) image->data[i].green;
        imageAccurate->data[i].blue  = (float) image->data[i].blue;
    }
    return imageAccurate;
}

AccurateImage* copyAccurateImage(AccurateImage* image, bool allocate_data, bool copy_pixels) {
    const int width = image->x;
    const int height = image->y;
    const int size = width * height;
    AccurateImage *imageAccurate = (AccurateImage*)malloc(sizeof(AccurateImage));
    imageAccurate->x = width;
    imageAccurate->y = height;
    if(allocate_data){
        imageAccurate->data = (AccuratePixel*)malloc(size * sizeof(AccuratePixel));
        if(copy_pixels){
            memcpy(imageAccurate->data, image->data, size * sizeof(AccuratePixel));
        }
    }
    return imageAccurate;
}

PPMImage* convertToPPPMImage(AccurateImage *imageIn) {
    PPMImage *imageOut;
    imageOut = (PPMImage *)malloc(sizeof(PPMImage));
    imageOut->data = (PPMPixel*)malloc(imageIn->x * imageIn->y * sizeof(PPMPixel));

    imageOut->x = imageIn->x;
    imageOut->y = imageIn->y;

    for(int i = 0; i < imageIn->x * imageIn->y; i++) {
        imageOut->data[i].red = imageIn->data[i].red;
        imageOut->data[i].green = imageIn->data[i].green;
        imageOut->data[i].blue = imageIn->data[i].blue;
    }
    return imageOut;
}


// Perform the final step, and return it as ppm.
PPMImage* imageDifference(AccurateImage *imageInSmall, AccurateImage *imageInLarge) {	
    const int width = imageInSmall->x;
	const int height = imageInSmall->y;
	const int size = width * height;

	PPMImage* imageOut = (PPMImage*)malloc(sizeof(PPMImage));
	imageOut->data = (PPMPixel*)malloc(size * sizeof(PPMPixel));

	imageOut->x = width;
	imageOut->y = height;
	for(int i = 0; i < size; i++) {
		float red = imageInLarge->data[i].red - imageInSmall->data[i].red;
		float green = imageInLarge->data[i].green - imageInSmall->data[i].green;
		float blue = imageInLarge->data[i].blue - imageInSmall->data[i].blue;
		red += 257.0 * (red < 0.0);
		green += 257.0 * (green < 0.0);
		blue += 257.0 * (blue < 0.0);
		imageOut->data[i].red = red;
		imageOut->data[i].green = green;
		imageOut->data[i].blue = blue;
	}
	
	return imageOut;
}


class OpenClBlur{
public:
    OpenClBlur() {
        
        // choose a platform containing this string if available
        string preferred_platform = "Intel";
        // select platform
        std::vector<Platform> all_platforms;
        Platform::get(&all_platforms);
        if (all_platforms.empty()){
            errorAndExit("No platforms found.\n");
        }
        Platform default_platform = all_platforms[0];
        for(const auto& platform: all_platforms){
            cerr << "Found platform: " << platform.getInfo<CL_PLATFORM_NAME>() << endl;
            if(platform.getInfo<CL_PLATFORM_NAME>().find(preferred_platform)!=-1){
                default_platform = platform;
            }
        }
        cerr << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

        // select device
        std::vector<Device> all_devices;
        default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
        if (all_devices.empty()){
            errorAndExit("No devices found.");
        }
        Device default_device = all_devices[0];
        for(const auto& device: all_devices){
            cerr << "Found device: " << device.getInfo<CL_DEVICE_NAME>() << endl;
        }
        cerr << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";
        device = default_device;


        for(const auto& foundDevice : all_devices) {
            if(foundDevice.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) {
                device = foundDevice;
                break;
            }
        }

        // create context
        context = Context({device});

        // create command queue with profiling enabled
        queue = CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

        // load opencl program and compile it
        Program::Sources sources;
        ifstream ifs("image_processing_opencl.cl");
        if (ifs.fail()){
            errorAndExit("Failed to open image_processing_opencl.");
        }
        string kernel_source { istreambuf_iterator<char>(ifs), istreambuf_iterator<char>() };
        sources.push_back({kernel_source.c_str(), kernel_source.size()});
        program = Program(context, sources);
        if (program.build({device}) != CL_SUCCESS){
            errorAndExit("Error building program!");
        }

        kernelHorizontal = Kernel(program, "kernelHorizontal");
        kernelVertical = Kernel(program, "kernelVertical");
    }
    AccurateImage* blur(AccurateImage* image, int size){

        std::size_t bufferSize = image->x * image->y * sizeof(AccuratePixel);
        Buffer buffer1(context, CL_MEM_READ_WRITE|CL_MEM_READ_WRITE, bufferSize);
        Buffer buffer2(context, CL_MEM_READ_WRITE|CL_MEM_READ_WRITE, bufferSize);

        events.emplace_back(make_pair("copy buffer to image", Event()));

        queue.enqueueWriteBuffer(buffer1, false, 0, bufferSize, image->data, nullptr, &events.back().second);

        blurIteration(image, buffer1, buffer2, size);

        AccurateImage* result = copyAccurateImage(image, true, false);

        events.emplace_back(make_pair("map buffer in memory", Event()));

        result->data = (AccuratePixel*)queue.enqueueMapBuffer(buffer1, CL_FALSE, CL_MAP_READ, 0, bufferSize, nullptr, &events.back().second);
        return result;
    }
    void finish(){
        // finish execution and print events
        queue.finish();
        for(auto [s, e]: events){
            printEvent(s, e);
        }
        events.clear();
    }

private:
    Device device;
    Context context;
    CommandQueue queue;
    Program program;
    Kernel kernelHorizontal;
    Kernel kernelVertical;
    std::vector<pair<string, Event>> events;
    void printEvent(string s, Event& evt){

        // ensure the event has completed
        assert(evt.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>()==CL_COMPLETE);
        cl_ulong queued = evt.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
        cl_ulong submit = evt.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
        cl_ulong start = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        cerr << "event: " << s << endl;
        cerr << "queue time: " << submit-queued << "ns" << endl;
        cerr << "run time: " << end-start << "ns" << endl;
        cerr << "total time: " << end-queued << "ns" << endl;
    }
    void blurIteration(AccurateImage* image, Buffer& src, Buffer& dst, cl_int size){

        // create Event for profiling
        events.emplace_back(make_pair("kernelHorizontal", Event()));
        // set call arguments
        kernelHorizontal.setArg(0, src);
        kernelHorizontal.setArg(1, dst);
        kernelHorizontal.setArg(2, image->x);
        kernelHorizontal.setArg(3, size);
        // call 2D kernel
        queue.enqueueNDRangeKernel(
                kernelHorizontal, // kernel to queue
                NullRange, // use no offset
                NDRange(image->y), // 1D kernel
                NullRange, // use no local range
                nullptr, // we use the queue in sequential mode so we don't have to specify Events that need to finish before
                &events.back().second // Event to use for profiling
        );

        events.emplace_back(make_pair("kernelVertical", Event()));
        // set call arguments
        kernelVertical.setArg(0, dst);
        kernelVertical.setArg(1, src);
        kernelVertical.setArg(2, image->y);
        kernelVertical.setArg(3, size);
        // call 2D kernel
        queue.enqueueNDRangeKernel(
                kernelVertical, // kernel to queue
                NullRange, // use no offset
                NDRange(image->x), // 1D kernel
                NullRange, // use no local range
                nullptr, // we use the queue in sequential mode so we don't have to specify Events that need to finish before
                &events.back().second // Event to use for profiling
        );
    }
};


int main(int argc, char** argv){
    PPMImage *image;
    if(argc > 1) {
        image = readPPM("flower.ppm");
    } else {
        image = readStreamPPM(stdin);
    }

    AccurateImage* imageAccurate = convertToAccurateImage(image);

    OpenClBlur blur;
    const int sizes[] = {2, 3, 5, 8};
    AccurateImage* images[4];
    for(int i = 0; i < 4; i++){
        images[i] = blur.blur(imageAccurate, sizes[i]);
    }
    blur.finish();

    PPMImage* final_images[3];
    #pragma omp parallel for num_threads(3)
    for(int i = 0; i < 3; i++){
        final_images[i] = imageDifference(images[i], images[i+1]);
    }

    if(argc > 1) {
        writePPM("flower_tiny.ppm", final_images[0]);
        writePPM("flower_small.ppm", final_images[1]);
        writePPM("flower_medium.ppm", final_images[2]);
    } else {
        writeStreamPPM(stdout, final_images[0]);
        writeStreamPPM(stdout, final_images[1]);
        writeStreamPPM(stdout, final_images[2]);
    }
}
