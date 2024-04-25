__kernel void kernelHorizontal(__global float* restrict in_image, __global float* restrict out_image, const int width, const int size){
    const int y = get_global_id(0);
    const float divisor = 1.0 / (2.0 * size + 1.0);
    
    for(int i = 0; i < 5; i++) {
        float3 sum = {0.0, 0.0, 0.0};

        for(int x = 0; x <= size; x++) {
            sum += vload3(y * width + x, in_image);
        }

        vstore3(sum / (size + 1), y * width + 0, out_image);

        for(int x = 1; x <= size; x++) {
            sum += vload3(y * width + x + size, in_image);
            vstore3(sum / (size + x + 1), y * width + x, out_image);
        }

        for(int x = size + 1; x < width - size; x++) {
            sum -= vload3(y * width + x - size - 1, in_image);
            sum += vload3(y * width + x + size, in_image);
            vstore3(sum * divisor, y * width + x, out_image);
        }

        for(int x = width - size; x < width; x++) {
            sum -= vload3(y * width + x - size - 1, in_image);
            vstore3(sum / (size + width - x), y * width + x, out_image);
        }
        // swap in and out
        __global float* tmp = in_image;
        in_image = out_image;
        out_image = tmp;
    }
    // swap in and out
    __global float* tmp = in_image;
    in_image = out_image;
    out_image = tmp;
}

__kernel void kernelVertical(__global float* restrict in_image, __global float* restrict out_image, const int height, const int size){
    const int width = get_global_size(0);
    const int x = get_global_id(0);
    const float divisor = 1.0 / (2.0 * size + 1.0);
    
    for(int i = 0; i < 5; i++) {
        float3 sum = {0.0, 0.0, 0.0};

        for(int y = 0; y <= size; y++) {
            sum += vload3(y * width + x, in_image);
        }

        vstore3(sum / (size + 1), 0 * width + x, out_image);

        for(int y = 1; y <= size; y++) {
            sum += vload3((y + size) * width + x, in_image);
            vstore3(sum / (y + size + 1), y * width + x, out_image);
        }

        for(int y = size + 1; y < height - size; y++) {
            sum -= vload3((y - size - 1) * width + x, in_image);
            sum += vload3((y + size) * width + x, in_image);
            vstore3(sum * divisor, y * width + x, out_image);
        }

        for(int y = height - size; y < height; y++) {
            sum -= vload3((y - size - 1) * width + x, in_image);
            vstore3(sum / (size + height - y), y * width + x, out_image);
        }
        // swap in and out
        __global float* tmp = in_image;
        in_image = out_image;
        out_image = tmp;
    }
    // swap in and out
    __global float* tmp = in_image;
    in_image = out_image;
    out_image = tmp;
}