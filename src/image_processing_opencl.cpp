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
    float red,green,blue;
} AccuratePixel;

typedef struct {
    int x, y;
    AccuratePixel *data;
} AccurateImage;

void errorAndExit(string error_message) {
    cerr << error_message << endl;
    exit(1);
}
AccurateImage *convertToAccurateImage(PPMImage *image) {
    AccurateImage *imageAccurate;
    imageAccurate = (AccurateImage *)malloc(sizeof(AccurateImage));
    imageAccurate->x = image->x;
    imageAccurate->y = image->y;
    std::size_t size = image->x * image->y * sizeof(AccuratePixel);
    imageAccurate->data = (AccuratePixel *)malloc(size);
    for(int i = 0; i < image->x * image->y; i++) {
        imageAccurate->data[i].red   = (float) image->data[i].red;
        imageAccurate->data[i].green = (float) image->data[i].green;
        imageAccurate->data[i].blue  = (float) image->data[i].blue;
    }
    return imageAccurate;
}

AccurateImage *copyAccurateImage(AccurateImage *image, bool allocate_data, bool copy_pixels) {
    // Make a copy
    AccurateImage *imageAccurate;
    imageAccurate = (AccurateImage *)malloc(sizeof(AccurateImage));
    imageAccurate->x = image->x;
    imageAccurate->y = image->y;
    std::size_t size = image->x * image->y * sizeof(AccuratePixel);
    if(allocate_data){
        imageAccurate->data = (AccuratePixel *)malloc(size);
        if(copy_pixels){
            memcpy(imageAccurate->data, image->data, size);
        }
    }
    return imageAccurate;
}

PPMImage * convertToPPPMImage(AccurateImage *imageIn) {
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
PPMImage * imageDifference(AccurateImage *imageInSmall, AccurateImage *imageInLarge) {
    PPMImage *imageOut;
    imageOut = (PPMImage *)malloc(sizeof(PPMImage));
    imageOut->data = (PPMPixel*)malloc(imageInSmall->x * imageInSmall->y * sizeof(PPMPixel));

    imageOut->x = imageInSmall->x;
    imageOut->y = imageInSmall->y;

    for(int i = 0; i < imageInSmall->x * imageInSmall->y; i++) {
        float value = (imageInLarge->data[i].red - imageInSmall->data[i].red);
        if(value > 255)
            imageOut->data[i].red = 255;
        else if (value < -1.0) {
            value = 257.0+value;
            if(value > 255)
                imageOut->data[i].red = 255;
            else
                imageOut->data[i].red = floor(value);
        } else if (value > -1.0 && value < 0.0) {
            imageOut->data[i].red = 0;
        } else {
            imageOut->data[i].red = floor(value);
        }

        value = (imageInLarge->data[i].green - imageInSmall->data[i].green);
        if(value > 255)
            imageOut->data[i].green = 255;
        else if (value < -1.0) {
            value = 257.0+value;
            if(value > 255)
                imageOut->data[i].green = 255;
            else
                imageOut->data[i].green = floor(value);
        } else if (value > -1.0 && value < 0.0) {
            imageOut->data[i].green = 0;
        } else {
            imageOut->data[i].green = floor(value);
        }

        value = (imageInLarge->data[i].blue - imageInSmall->data[i].blue);
        if(value > 255)
            imageOut->data[i].blue = 255;
        else if (value < -1.0) {
            value = 257.0+value;
            if(value > 255)
                imageOut->data[i].blue = 255;
            else
                imageOut->data[i].blue = floor(value);
        } else if (value > -1.0 && value < 0.0) {
            imageOut->data[i].blue = 0;
        } else {
            imageOut->data[i].blue = floor(value);
        }
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

        // create kernel
        naive_kernel = Kernel(program, "naive_kernel");
    }
    AccurateImage * blur(AccurateImage *image, int size){
        // perform blur operation

        // allocate two buffers:
        // use sizeof(AccuratePixel) instead of sizeof(float3) because sizeof(float3) = sizeof(float4)
        // even though vload3 and vstore3 can operate on packed arrays
        std::size_t bufferSize = image->x * image->y * sizeof(AccuratePixel);
        Buffer buffer1(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, bufferSize);
        Buffer buffer2(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, bufferSize);
        // create a new Event so we know how long it took
        events.emplace_back(make_pair("copy buffer to image", Event()));
        // copy image to buffer
        queue.enqueueWriteBuffer(buffer1, false, 0, bufferSize, image->data, nullptr, &events.back().second);
        // 5 blur iterations
        blurIteration(image, buffer1, buffer2, size);
        blurIteration(image, buffer2, buffer1, size);
        blurIteration(image, buffer1, buffer2, size);
        blurIteration(image, buffer2, buffer1, size);
        blurIteration(image, buffer1, buffer2, size);
        // create new empty image
        AccurateImage * result = copyAccurateImage(image, false, false);
        // push back new event
        events.emplace_back(make_pair("map buffer in memory", Event()));
        // map buffer in memory - avoids having to copy again
        result->data = (AccuratePixel *)queue.enqueueMapBuffer(buffer2,CL_FALSE,CL_MAP_READ, 0, bufferSize, nullptr, &events.back().second);
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
    Kernel naive_kernel;
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
    void blurIteration(AccurateImage *image, Buffer& src, Buffer& dst, cl_int size){
        // enqueue the OpenCL kernel naive_kernel

        // create Event for profiling
        events.emplace_back(make_pair("naive_kernel", Event()));
        // set call arguments
        naive_kernel.setArg(0, src);
        naive_kernel.setArg(1, dst);
        naive_kernel.setArg(2, size);
        // call 2D kernel
        queue.enqueueNDRangeKernel(
                naive_kernel, // kernel to queue
                NullRange, // use no offset
                NDRange(image->x, image->y), // 2D kernel
                NullRange, // use no local range
                nullptr, // we use the queue in sequential mode so we don't have to specify Events that need to finish before
                &events.back().second // Event to use for profiling
        );
    }
};


int main(int argc, char** argv){
    // read image
    PPMImage *image;
    // select where to read the image from
    if(argc > 1) {
        // from file for debugging (with argument)
        image = readPPM("flower.ppm");
    } else {
        // from stdin for cmb
        image = readStreamPPM(stdin);
    }

    AccurateImage *imageAccurate = convertToAccurateImage(image);

    OpenClBlur blur;
    // apply blur
    AccurateImage  * imageAccurate2_tiny = blur.blur(imageAccurate, 2);
    AccurateImage  * imageAccurate2_small = blur.blur(imageAccurate, 3);
    AccurateImage  * imageAccurate2_medium = blur.blur(imageAccurate, 5);
    AccurateImage  * imageAccurate2_large = blur.blur(imageAccurate, 8);

    // an intermediate step can be saved for debugging like this
//    writePPM("imageAccurate2_tiny.ppm", convertToPPPMImage(imageAccurate2_tiny));

    // finish OpenCl execution and print events
    // we do this only once at the end in order to keep the OpenCL queue filled
    blur.finish();
    // calculate difference
    PPMImage *final_tiny = imageDifference(imageAccurate2_tiny, imageAccurate2_small);
    PPMImage *final_small = imageDifference(imageAccurate2_small, imageAccurate2_medium);
    PPMImage *final_medium = imageDifference(imageAccurate2_medium, imageAccurate2_large);

    // Save the images.
    if(argc > 1) {
        writePPM("flower_tiny.ppm", final_tiny);
        writePPM("flower_small.ppm", final_small);
        writePPM("flower_medium.ppm", final_medium);
    } else {
        writeStreamPPM(stdout, final_tiny);
        writeStreamPPM(stdout, final_small);
        writeStreamPPM(stdout, final_medium);
    }
}
