__kernel void kernelHorizontal(__global float* restrict in_image, __global float* restrict out_image, const int width, const int size){
    const int y = get_global_id(0);
    const int yWidth = y * width;
    const float4 multiplier = {size * 2.0f + 1, size * 2.0f + 1, size * 2.0f + 1, 1.0f};

    for(int iteration = 0; iteration < 4; iteration++) {
        float4 sum = {0.0, 0.0, 0.0, 0.0};

        for(int x = 0; x <= size; x++) {
            sum += vload4(yWidth + x, in_image);
        }

        vstore4(sum * multiplier / (size + 1), yWidth + 0, out_image);

        for(int x = 1; x <= size; x++) {
            sum += vload4(yWidth + x + size, in_image);
            vstore4(sum * multiplier / (size + x + 1), yWidth + x, out_image);
        }

        for(int x = size + 1; x < width - size; x++) {
            sum -= vload4(yWidth + x - size - 1, in_image);
            sum += vload4(yWidth + x + size, in_image);
            vstore4(sum, yWidth + x, out_image);
        }

        for(int x = width - size; x < width; x++) {
            sum -= vload4(yWidth + x - size - 1, in_image);
            vstore4(sum * multiplier / (size + width - x), yWidth + x, out_image);
        }
        // swap in_image and out_image 
        __global float* tmp = in_image;
        in_image = out_image;
        out_image = tmp;
    }
}

__kernel void kernelHorizontalTranspose(__global const float* restrict in_image, __global float* restrict out_image, const int width, const int size){
    const int y = get_global_id(0);
    const int height = get_global_size(0);
    const int yWidth = y * width;
    const float4 multiplier = {size * 2.0f + 1, size * 2.0f + 1, size * 2.0f + 1, 1.0f};

    float4 sum = {0.0, 0.0, 0.0, 0.0};

    for(int x = 0; x <= size; x++) {
        sum += vload4(yWidth + x, in_image);
    }

    vstore4(sum * multiplier / (size + 1), 0 * height + y, out_image);

    for(int x = 1; x <= size; x++) {
        sum += vload4(yWidth + x + size, in_image);
        vstore4(sum * multiplier / (size + x + 1), x * height + y, out_image);
    }

    for(int x = size + 1; x < width - size; x++) {
        sum -= vload4(yWidth + x - size - 1, in_image);
        sum += vload4(yWidth + x + size, in_image);
        vstore4(sum, x * height + y, out_image);
    }

    for(int x = width - size; x < width; x++) {
        sum -= vload4(yWidth + x - size - 1, in_image);
        vstore4(sum * multiplier / (size + width - x), x * height + y, out_image);
    }
}

__kernel void kernelVertical(__global float* restrict in_image, __global float* restrict out_image, const int height, const int size){
    const int width = get_global_size(0);
    const int x = get_global_id(0);
    const int xHeight = x * height;
    const float4 multiplier = {size * 2.0f + 1, size * 2.0f + 1, size * 2.0f + 1, 1.0f};

    for(int iteration = 0; iteration < 5; iteration++) {
        float4 sum = {0.0, 0.0, 0.0, 0.0};

        for(int y = 0; y <= size; y++) {
            sum += vload4(xHeight + y, in_image);
        }

        vstore4(sum * multiplier / (size + 1), xHeight + 0, out_image);

        for(int y = 1; y <= size; y++) {
            sum += vload4(xHeight + y + size, in_image);
            vstore4(sum * multiplier / (size + y + 1), xHeight + y, out_image);
        }

        for(int y = size + 1; y < height - size; y++) {
            sum -= vload4(xHeight + y - size - 1, in_image);
            sum += vload4(xHeight + y + size, in_image);
            vstore4(sum, xHeight + y, out_image);
        }

        for(int y = height - size; y < height; y++) {
            sum -= vload4(xHeight + y - size - 1, in_image);
            vstore4(sum * multiplier / (size + height - y), xHeight + y, out_image);
        }

        // swap in_image and out_image 
        __global float* tmp = in_image;
        in_image = out_image;
        out_image = tmp;
    }
}