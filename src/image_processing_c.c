#include <math.h>
#include <string.h>
#include <stdlib.h>

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
     v4Accurate* data;
} AccurateImage;

AccurateImage* convertToAccurateImage(const PPMImage* image) {
	const int width = image->x;
	const int height = image->y;
	const int size = width * height;

	AccurateImage* imageAccurate = (AccurateImage*)malloc(sizeof(AccurateImage));
	imageAccurate->data = (v4Accurate*)malloc(size * sizeof(v4Accurate));
	#pragma GCC unroll 8
	for(int i = 0; i < size; i++) {
		PPMPixel pixel = image->data[i];
		imageAccurate->data[i][0] = (float) pixel.red;
		imageAccurate->data[i][1] = (float) pixel.green;
		imageAccurate->data[i][2] = (float) pixel.blue;
	}
	imageAccurate->x = width;
	imageAccurate->y = height;
	
	return imageAccurate;
}

AccurateImage* copyAccurateImage(const AccurateImage* imageIn) {
	const int size = imageIn->x * imageIn->y;
	AccurateImage* imageOut = (AccurateImage*)malloc(sizeof(AccurateImage));
	imageOut->data = (v4Accurate*)malloc(size * sizeof(v4Accurate));
	
	memcpy(imageOut->data, imageIn->data, size * sizeof(v4Accurate));

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
		imageOut->data[i].red = imageIn->data[i][0];
		imageOut->data[i].green = imageIn->data[i][1];
		imageOut->data[i].blue = imageIn->data[i][2];
    }
    return imageOut;
}

AccurateImage* blurIteration(PPMImage* image, const int size) {
	const int width = image->x;
	const int height = image->y;
	v4Accurate* fake = (v4Accurate*)malloc(width * height * sizeof(v4Accurate));
	AccurateImage* real = convertToAccurateImage(image);
	v4Accurate* scratch = fake;
	v4Accurate* imageOut = real->data;
	
	v4Accurate sum;
	for(int i = 0; i < BLUR_ITERATIONS; i++) {
		// (i % 2) ? (scratch = real->data) : (scratch = fake);
		// (i % 2) ? (imageOut = fake) : (imageOut = real->data);	
		for(int y = 0; y < height; y++) {
			const int yWidth = y * width;

			sum[0] = 0.0;
			sum[1] = 0.0;
			sum[2] = 0.0;

			for(int x = 0; x <= size; x++) {
				sum += imageOut[yWidth + x];
			}

			scratch[yWidth + 0] = sum / (v4Accurate){size + 1, size + 1, size + 1, size + 1};

			for(int x = 1; x <= size; x++) {
				sum += imageOut[yWidth + x + size];
				scratch[yWidth + x] = sum / (v4Accurate){size + x + 1, size + x + 1, size + x + 1, size + x + 1};
			}

			for(int x = size + 1; x < width - size; x++) {
				sum -= imageOut[yWidth + x - size - 1];
				sum += imageOut[yWidth + x + size];
				scratch[yWidth + x] = sum / (v4Accurate){2 * size + 1, 2 * size + 1, 2 * size + 1, 2 * size + 1};
			}

			for(int x = width - size; x < width; x++) {
				sum -= imageOut[yWidth + x - size - 1];
				scratch[yWidth + x] = sum / (v4Accurate){size + width - x, size + width - x, size + width - x, size + width - x};
			}
		}
	//}

	//for(int i = 0; i < BLUR_ITERATIONS; i++) {
		// (i % 2) ? (scratch = real->data) : (scratch = fake);
		// (i % 2) ? (imageOut = fake) : (imageOut = real->data);	
		for(int x = 0; x < width; x++) {
			const int xHeight = x * height;

			sum[0] = 0.0;
			sum[1] = 0.0;
			sum[2] = 0.0;

			for(int y = 0; y <= size; y++) {
				sum += scratch[y * width + x];
			}

			imageOut[0 * width + x] = sum / (v4Accurate){size + 1, size + 1, size + 1, size + 1};

			for(int y = 1; y <= size; y++) {
				sum += scratch[y * width + x + size];
				imageOut[y * width + x] = sum / (v4Accurate){y + size + 1, y + size + 1, y + size + 1, y + size + 1};
			}

			for(int y = size + 1; y < height - size; y++) {
				sum -= scratch[y * width + x - size - 1];
				sum += scratch[y * width + x + size];
				imageOut[y * width + x] = sum / (v4Accurate){2 * size + 1, 2 * size + 1, 2 * size + 1, 2 * size + 1};
			}

			for(int y = height - size; y < height; y++) {
				sum -= scratch[y * width + x - size - 1];
				imageOut[y * width + x] = sum / (v4Accurate){size + height - y, size + height - y, size + height - y, size + height - y};
			}
		}
	}
	//free(scratch);
	real->data = scratch;
	return real;
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
		v4Accurate diffvec = imageInLarge->data[i] - imageInSmall->data[i];
		float red = diffvec[0];
		float green = diffvec[1];
		float blue = diffvec[2];

		red = red < 0.0 ? red + 257.0 : red;
		green = green < 0.0 ? green + 257.0 : green;
		blue = blue < 0.0 ? blue + 257.0 : blue;

		imageOut->data[i] = (PPMPixel){red, green, blue};
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

	AccurateImage** images = (AccurateImage**)malloc(sizeof(AccurateImage*) * 4);
	PPMImage* imagesPPM[3];
	const int sizes[4] = {2, 3, 5, 8};

	//#pragma omp parallel for num_threads(4)
	for(int i = 0; i < 4; i++) {
		images[i] = blurIteration(image, sizes[i]);
	}

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

