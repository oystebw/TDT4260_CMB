__kernel void naive_kernel(
        __global const float* restrict in_image,
        __global float* restrict out_image,
        const int size
)
{
    // global 2D NDRange sizes (size of the image)
    int num_cols = get_global_size(0);
    int num_rows = get_global_size(1);
    // point in currently being executed (each pixel)
    int senterX = get_global_id(0);
    int senterY = get_global_id(1);

    // For each pixel we compute a box blur
    float3 sum = (float3) (0.0f, 0.0f, 0.0f);
    int countIncluded = 0;
    for(int x = -size; x <= size; x++) {

        for(int y = -size; y <= size; y++) {
            int currentX = senterX + x;
            int currentY = senterY + y;

            // Check if we are outside the bounds
            if(currentX < 0)
                continue;
            if(currentX >= num_cols)
                continue;
            if(currentY < 0)
                continue;
            if(currentY >= num_rows)
                continue;

            // Now we can begin
            int offsetOfThePixel = (num_cols * currentY + currentX);
            float3 tmp = vload3(offsetOfThePixel, in_image);
            sum += tmp;
            // Keep track of how many values we have included
            countIncluded++;
        }
    }

    // Now we compute the final value
    float3 value = sum / countIncluded;

    // Update the output image
    int offsetOfThePixel = (num_cols * senterY + senterX);
    vstore3(value, offsetOfThePixel, out_image);
}
