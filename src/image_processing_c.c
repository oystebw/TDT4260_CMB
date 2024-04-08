#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <arm_neon.h>

#include <omp.h>

#include "ppm.h"

#define BLUR_ITERATIONS 5

typedef float v4Accurate __attribute__((vector_size(16)));

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

typedef struct {
	unsigned char rgb[3];
} v3Byte;

typedef struct {
	int x, y;
	v3Byte* data;
} SimpleImage;

typedef struct {
     int x, y;
     float32x4_t* data;
} AccurateImage;

AccurateImage* convertToAccurateImage(const PPMImage* image) {
	const int size = image->x * image->y;

	AccurateImage* imageAccurate = (AccurateImage*)malloc(sizeof(AccurateImage));
	imageAccurate->data = (float32x4_t*)malloc(size * sizeof(float32x4_t));
	#pragma GCC unroll 8
	for(int i = 0; i < size; i++) {
		imageAccurate->data[i].val[0] = (float) image->data[i].red;
		imageAccurate->data[i].val[1] = (float) image->data[i].green;
		imageAccurate->data[i].val[2] = (float) image->data[i].blue;
	}
	imageAccurate->x = image->x;
	imageAccurate->y = image->y;
	
	return imageAccurate;
}

AccurateImage* copyAccurateImage(const AccurateImage* imageIn) {
	const int size = imageIn->x * imageIn->y;
	AccurateImage* imageOut = (AccurateImage*)malloc(sizeof(AccurateImage));
	imageOut->data = (float32x4_t*)malloc(size * sizeof(float32x4_t));
	
	memcpy(imageOut->data, imageIn->data, size * sizeof(float32x4_t));

	imageOut->x = imageIn->x;
	imageOut->y = imageIn->y;
	return imageOut;
}

PPMImage* convertToPPPMImage(const AccurateImage* imageIn) {
    const int size = imageIn->x * imageIn->y;
	
	PPMImage* imageOut = (PPMImage*)malloc(sizeof(PPMImage));
    imageOut->data = (PPMPixel*)malloc(size * sizeof(PPMPixel));

    imageOut->x = imageIn->x;
    imageOut->y = imageIn->y;
	#pragma GCC unroll 8
    for(int i = 0; i < size; i++) {
		imageOut->data[i].red = imageIn->data[i].val[0];
		imageOut->data[i].green = imageIn->data[i].val[1];
		imageOut->data[i].blue = imageIn->data[i].val[2];
    }
    return imageOut;
}

void blurIteration(AccurateImage* image, const int size) {
	const int width = image->x;
	const int height = image->y;
	float32x4_t* scratch = (float32x4_t*)malloc(width * height * sizeof(float32x4_t));

	// Transpose to be more cache / access friendly
	float32x4_t (*data)   [width] = (void*) image->data;
  	float32x4_t (*buffer) [height] = (void*) scratch;
	
	#pragma GCC unroll 5
	for(int i = 0; i < BLUR_ITERATIONS; i++) {
		#pragma GCC unroll 8
		for(int y = 0; y < height; y++) {

			float32x4_t sum = {0.0, 0.0, 0.0, 0.0};

			for(int x = 0; x <= size; x++) {
				sum = vaddq_f32(sum, data[y][x]);
			}

			buffer[0][y] = vmulq_f32(sum, vrecpeq_f32((float32x4_t){size + 1, size + 1, size + 1, size + 1}));

			for(int x = 1; x <= size; x++) {
				sum = vaddq_f32(sum, data[y][x + size]);
				buffer[x][y] = vmulq_f32(sum, vrecpeq_f32((float32x4_t){size + x + 1, size + x + 1, size + x + 1, size + x + 1}));
			}

			for(int x = size + 1; x < width - size; x++) {
				sum = vsubq_f32(sum, data[y][x - size - 1]);
				sum = vaddq_f32(sum, data[y][x + size]);
				buffer[x][y] = vmulq_f32(sum, vrecpeq_f32((float32x4_t){2 * size + 1, 2 * size + 1, 2 * size + 1, 2 * size + 1}));
			}

			for(int x = width - size; x < width; x++) {
				sum = vsubq_f32(sum, data[y][x - size - 1]);
				buffer[x][y] = vmulq_f32(sum, vrecpeq_f32((float32x4_t){size + width - x, size + width - x, size + width - x, size + width - x}));
			}
			
		}
		#pragma GCC unroll 8
		for(int x = 0; x < width; x++) {

			float32x4_t sum = {0.0, 0.0, 0.0, 0.0};

			for(int y = 0; y <= size; y++) {
				sum = vaddq_f32(sum, buffer[x][y]);
			}

			data[0][x] = vmulq_f32(sum, vrecpeq_f32(float32x4_t){size + 1, size + 1, size + 1, size + 1}));

			for(int y = 1; y <= size; y++) {
				sum = vaddq_f32(sum, buffer[x][(y + size)]);
				data[y][x] = vmulq_f32(sum, vrecpeq_f32((float32x4_t){y + size + 1, y + size + 1, y + size + 1, y + size + 1}));
			}

			for(int y = size + 1; y < height - size; y++) {
				sum = vsubq_f32(sum, buffer[x][y - size - 1]);
				sum = vaddq_f32(sum, buffer[x][y + size]);
				data[y][x] = vmulq_f32(sum, vrecpeq_f32((float32x4_t){2 * size + 1, 2 * size + 1, 2 * size + 1, 2 * size + 1}));
			}

			for(int y = height - size; y < height; y++) {
				sum = vsubq_f32(sum, buffer[x][y - size - 1]);
				data[y][x] = vmulq_f32(sum, vrecpeq_f32((float32x4_t){size + height - y, size + height - y, size + height - y, size + height - y}));
			}
		}
	}
	free(scratch);
}

PPMImage* imageDifference(const AccurateImage* imageInSmall, const AccurateImage* imageInLarge) {
	const int width = imageInSmall->x;
	const int height = imageInSmall->y;
	const int size = width * height;

	PPMImage* imageOut = (PPMImage*)malloc(sizeof(PPMImage));
	imageOut->data = (PPMPixel*)malloc(size * sizeof(PPMPixel));

	imageOut->x = width;
	imageOut->y = height;
	#pragma GCC unroll 8
	for(int i = 0; i < size; i++) {
		float32x4_t diffvec = vaddq_f32(imageInLarge->data[i], -imageInSmall->data[i]);
		float red = diffvec.val[0];
		float green = diffvec.val[1];
		float blue = diffvec.val[2];
		red += 257.0 * (red < 0.0);
		green += 257.0 * (green < 0.0);
		blue += 257.0 * (blue < 0.0);
		imageOut->data[i].red = red;
		imageOut->data[i].green = green;
		imageOut->data[i].blue = blue;
	}
	
	return imageOut;
}


int main(int argc, char** argv) {

    PPMImage* image;
    if(argc > 1) {
        image = readPPM("flower.ppm");
    } else {
        image = readStreamPPM(stdin);
    }
	
	AccurateImage* imageAccurate1_tiny = convertToAccurateImage(image);
	AccurateImage* imageAccurate1_small = copyAccurateImage(imageAccurate1_tiny);
	AccurateImage* imageAccurate1_medium = copyAccurateImage(imageAccurate1_tiny);
	AccurateImage* imageAccurate1_large = copyAccurateImage(imageAccurate1_tiny);

	AccurateImage* images[4] = {imageAccurate1_tiny, imageAccurate1_small, imageAccurate1_medium, imageAccurate1_large};
	const int sizes[4] = {2, 3, 5, 8};

	#pragma omp parallel for simd num_threads(4)
	for(int i = 0; i < 4; i++) {
		blurIteration(images[i], sizes[i]);
	}

	PPMImage* imagesPPM[3];

	#pragma omp parallel for simd num_threads(3)
	for(int i = 0; i < 3; i++) {
		imagesPPM[i] = imageDifference(images[i], images[i + 1]);
	}

    if(argc > 1) {
        writePPM("flower_tiny.ppm", imagesPPM[0]);
        writePPM("flower_small.ppm", imagesPPM[1]);
        writePPM("flower_medium.ppm", imagesPPM[2]);
    } else {
        writeStreamPPM(stdout, imagesPPM[0]);
        writeStreamPPM(stdout, imagesPPM[1]);
        writeStreamPPM(stdout, imagesPPM[2]);
    }
}

