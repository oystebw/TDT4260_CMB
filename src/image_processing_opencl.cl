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
