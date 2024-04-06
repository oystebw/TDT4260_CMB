__kernel void naive_kernel(__global const float* restrict in_image, __global float* restrict out_image, const int size){

    
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    
    int senterX = get_global_id(0);
    int senterY = get_global_id(1);
    int countIncluded = 0;

    float3 sum = (float3) (0.0f, 0.0f, 0.0f);

    for(int x = -size; x <= size; x++) {
        int currentX = senterX + x;
        if(currentX < 0 || currentX >= width) {
            continue;
        }

        for(int y = -size; y <= size; y++) {
            int currentY = senterY + y;
            if(currentY < 0 || currentY >= height) {
                continue;
            }
            int offsetOfThePixel = (width * currentY + currentX);
            sum += vload3(offsetOfThePixel, in_image);

            countIncluded++;
        }
    }

    // Now we compute the final value
    float3 value = sum / countIncluded;

    // Update the output image
    int offsetOfThePixel = (width * senterY + senterX);
    vstore3(value, offsetOfThePixel, out_image);
}

__kernel void kernelHorizontal(__global const float* restrict in_image, __global float* restrict out_image, const int width){
    const int y = get_global_id(0);
    float3 sum;
    
    sum =  vload3(y * width + 0, in_image);
    sum += vload3(y * width + 1, in_image);
    sum += vload3(y * width + 2, in_image);
    vstore3(sum / 3, y * width + 0, out_image);
    sum += vload3(y * width + 3, in_image);
    vstore3(sum / 4, y * width + 1, out_image);
    sum += vload3(y * width + 4, in_image);
    vstore3(sum / 5, y * width + 2, out_image);

    for(int x = 3; x < width - 2; x++) {
        sum -= vload3(y * width + x - 3, in_image);
        sum += vload3(y * width + x + 2, in_image);
        vstore3(sum / 5, y * width + x, out_image);
    }
    sum -= vload3(y * width + width - 5, in_image);
    vstore3(sum / 4, y * width + width - 2, out_image);
    sum -= vload3(y * width + width - 4, in_image);
    vstore3(sum / 3, y * width + width - 1, out_image);
}

__kernel void kernelVertical(__global const float* restrict in_image, __global float* restrict out_image, const int height){
    const int width = get_global_size(0);
    const int x = get_global_id(0);
    float3 sum;
    
    sum =  vload3(0 * width + x, in_image);
    sum += vload3(1 * width + x, in_image);
    sum += vload3(2 * width + x, in_image);
    vstore3(sum / 3, 0 * width + x, out_image);
    sum += vload3(3 * width + x, in_image);
    vstore3(sum / 4, 1 * width + x, out_image);
    sum += vload3(4 * width + x, in_image);
    vstore3(sum / 5, 2 * width + x, out_image);

    for(int y = 3; y < height - 2; y++) {
        sum -= vload3((y - 3) * width + x, in_image);
        sum += vload3((y + 2) * width + x, in_image);
        vstore3(sum / 5, y * width + x, out_image);
    }
    sum -= vload3((height - 5) * width + x, in_image);
    vstore3(sum / 4, (height - 2) * width + x, out_image);
    sum -= vload3((height - 4) * width + x, in_image);
    vstore3(sum / 3, (height - 1) * width + x, out_image);
}