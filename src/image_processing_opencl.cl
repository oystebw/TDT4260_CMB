__kernel void kernelHorizontal(__global const float* in_image_raw, __global float* out_image_raw, const int width, const int height){
    int y = get_global_id(0);
    float3 sum = {0.0, 0.0, 0.0};

    int size;
    __global const float* in_image;
    __global float* out_image;
    if(y < height) {
        size = 2;
        in_image = in_image_raw;
        out_image = out_image_raw;
    }
    else if(y < height * 2) {
        y -= height;
        size = 3;
        in_image = in_image_raw + width * height * 3;
        out_image = out_image_raw + width * height * 3;
    }
    else if(y < height * 3) {
        y -= height * 2;
        size = 5;
        in_image = in_image_raw + width * height * 2 * 3;
        out_image = out_image_raw + width * height * 2 * 3;
    }
    else {
        y -= height * 3;
        size = 8;
        in_image = in_image_raw + width * height * 3 * 3;
        out_image = out_image_raw + width * height * 3 * 3;
    }

    for(int x = 0; x <= size; x++) {
        sum += vload3(y * width + x, in_image);
    }

    vstore3(sum / (size + 1), y + 0 * height, out_image);

    for(int x = 1; x <= size; x++) {
        sum += vload3(y * width + x + size, in_image);
        vstore3(sum / (size + x + 1), y + x * height, out_image);
    }

    for(int x = size + 1; x < width - size; x++) {
        sum -= vload3(y * width + x - size - 1, in_image);
        sum += vload3(y * width + x + size, in_image);
        vstore3(sum / ((size << 1) + 1), y + x * height, out_image);
    }

    for(int x = width - size; x < width; x++) {
        sum -= vload3(y * width + x - size - 1, in_image);
        vstore3(sum / (size + width - x), y + x * height, out_image);
    }
}

__kernel void kernelVertical(__global const float* in_image_raw, __global float* out_image_raw, const int width, const int height){
    int x = get_global_id(0);
    float3 sum = {0.0, 0.0, 0.0};

    int size;
    __global const float* in_image;
    __global float* out_image;
    if(x < width) {
        size = 2;
        in_image = in_image_raw;
        out_image = out_image_raw;
    }
    else if(x < width * 2) {
        x -= width;
        size = 3;
        in_image = in_image_raw + width * height * 3;
        out_image = out_image_raw + width * height * 3;
    }
    else if(x < width * 3) {
        x -= width * 2;
        size = 5;
        in_image = in_image_raw + width * height * 2 * 3;
        out_image = out_image_raw + width * height * 2 * 3;
    }
    else {
        x -= width * 3;
        size = 8;
        in_image = in_image_raw + width * height * 3 * 3;
        out_image = out_image_raw + width * height * 3 * 3;
    }

    for(int y = 0; y <= size; y++) {
        sum += vload3(y + x * height, in_image);
    }

    vstore3(sum / (size + 1), 0 * width + x, out_image);

    for(int y = 1; y <= size; y++) {
        sum += vload3((y + size) + x * height, in_image);
        vstore3(sum / (y + size + 1), y * width + x, out_image);
    }

    for(int y = size + 1; y < height - size; y++) {
        sum -= vload3((y - size - 1) + x * height, in_image);
        sum += vload3((y + size) + x * height, in_image);
        vstore3(sum / ((size << 1) + 1), y * width + x, out_image);
    }

    for(int y = height - size; y < height; y++) {
        sum -= vload3((y - size - 1) + x * height, in_image);
        vstore3(sum / (size + height - y), y * width + x, out_image);
    }
}