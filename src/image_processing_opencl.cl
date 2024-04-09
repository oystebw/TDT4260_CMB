__kernel void kernelHorizontal(__global const float* restrict test, __global float* restrict out_image, const int width, const int size){
    const int height = get_global_size(0);
    const int y = get_global_id(0);
    float3 sum = {0.0, 0.0, 0.0};

    __global const float* restrict in_image = test + y * width;

    for(int x = 0; x <= size; x++) {
        sum += vload3(x, in_image);
    }

    vstore3(sum / (size + 1), y + 0 * height, out_image);

    for(int x = 1; x <= size; x++) {
        sum += vload3(x + size, in_image);
        vstore3(sum / (size + x + 1), y + x * height, out_image);
    }

    for(int x = size + 1; x < width - size; x++) {
        sum -= vload3(x - size - 1, in_image);
        sum += vload3(x + size, in_image);
        vstore3(sum / ((size << 1) + 1), y + x * height, out_image);
    }

    for(int x = width - size; x < width; x++) {
        sum -= vload3(x - size - 1, in_image);
        vstore3(sum / (size + width - x), y + x * height, out_image);
    }
}

__kernel void kernelVertical(__global const float* restrict test, __global float* restrict out_image, const int height, const int size){
    const int width = get_global_size(0);
    const int x = get_global_id(0);
    float3 sum = {0.0, 0.0, 0.0};

    __global const float* restrict in_image = test + x * height;

    for(int y = 0; y <= size; y++) {
        sum += vload3(y, in_image);
    }

    vstore3(sum / (size + 1), 0 * width + x, out_image);

    for(int y = 1; y <= size; y++) {
        sum += vload3((y + size), in_image);
        vstore3(sum / (y + size + 1), y * width + x, out_image);
    }

    for(int y = size + 1; y < height - size; y++) {
        sum -= vload3((y - size - 1), in_image);
        sum += vload3((y + size), in_image);
        vstore3(sum / ((size << 1) + 1), y * width + x, out_image);
    }

    for(int y = height - size; y < height; y++) {
        sum -= vload3((y - size - 1), in_image);
        vstore3(sum / (size + height - y), y * width + x, out_image);
    }
}