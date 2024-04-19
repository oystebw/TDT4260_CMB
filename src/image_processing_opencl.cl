__kernel void kernelDiff(__global uchar* restrict result, __global const float* restrict large, __global const float* restrict small, const int height){
    const int width = get_global_size(0);
    const int x = get_global_id(0);
    const int xHeight = x * height;

    for(int y = 0; y < height; y++) {
        float3 diff = vload3(xHeight + y, large) - vload3(xHeight + y, small);
        unsigned char red = diff[0] < 0.0f ? diff[0] + 257 : diff[0];
        unsigned char green = diff[1] < 0.0f ? diff[1] + 257 : diff[1];
        unsigned char blue = diff[2] < 0.0f ? diff[2] + 257 : diff[2];
        uchar3 rgb = (uchar3){red, green, blue};
        vstore3(rgb, y * width + x, result);
    }
}