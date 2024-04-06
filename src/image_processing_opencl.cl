__kernel void kernelHorizontal(__global const float* restrict in_image, __global float* restrict out_image, const int width, const int size){
    const int height = get_global_size(0);
    const int y = get_global_id(0);
    float3 sum = {0.0, 0.0, 0.0};

    for(int x = 0; x < size; x++) {
        sum += vload3(y * width + x, in_image);
    }

    vstore3(sum / (size + 1), y * width + 0, out_image);

    for(int x = 1; x <= size; x++) {
        sum += vload3((y + size) * width + x, in_image);
        vstore3(sum / (size + x + 1), y * width + x, out_image);
    }

    for(int x = size + 1; x < height - size; x++) {
        sum -= vload3(y * width + x - size - 1, in_image);
        sum += vload3(y * width + x + size, in_image);
        vstore3(sum / (2 * size + 1), y * width + x, out_image);
    }

    for(int x = height - size; x < height; x++) {
        sum -= vload3((x - size - 1) * width + x, in_image);
        vstore3(sum / (size + height - x), y * width + x, out_image);
    }
}

__kernel void kernelVertical(__global const float* restrict in_image, __global float* restrict out_image, const int height, const int size){
    const int width = get_global_size(0);
    const int x = get_global_id(0);
    float3 sum = {0.0, 0.0, 0.0};

    for(int y = 0; y < size; y++) {
        sum += vload3(y * width + x, in_image);
    }

    vstore3(sum / (size + 1), 0 * width + x, out_image);

    for(int y = 1; y <= size; y++) {
        sum += vload3((y + size) * width + x, in_image);
        vstore3(sum / (size + y + 1), y * width + x, out_image);
    }

    for(int y = size + 1; y < height - size; y++) {
        sum -= vload3((y - size - 1) * width + x, in_image);
        sum += vload3((y + size) * width + x, in_image);
        vstore3(sum / (2 * size + 1), y * width + x, out_image);
    }

    for(int y = height - size; y < height; y++) {
        sum -= vload3((y - size - 1) * width + x, in_image);
        vstore3(sum / (size + height - y), y * width + x, out_image);
    }
}