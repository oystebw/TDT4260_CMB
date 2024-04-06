#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <omp.h>

#include "ppm.h"

#pragma GCC optimize ("O3")

#define BLUR_ITERATIONS 5

#define DIVISOR2 5
#define DIVISOR3 7
#define DIVISOR5 11
#define DIVISOR8 17

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
     v4Accurate *data;
} AccurateImage;

AccurateImage* convertToAccurateImage(const PPMImage* image) {
	const int size = image->x * image->y;

	AccurateImage* imageAccurate = (AccurateImage*)malloc(sizeof(AccurateImage));
	imageAccurate->data = (v4Accurate*)malloc(size * sizeof(v4Accurate));
	for(int i = 0; i < size; i++) {
		imageAccurate->data[i][0] = (float) image->data[i].red;
		imageAccurate->data[i][1] = (float) image->data[i].green;
		imageAccurate->data[i][2] = (float) image->data[i].blue;
	}
	imageAccurate->x = image->x;
	imageAccurate->y = image->y;
	
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

    for(int i = 0; i < size; i++) {
		imageOut->data[i].red = imageIn->data[i][0];
		imageOut->data[i].green = imageIn->data[i][1];
		imageOut->data[i].blue = imageIn->data[i][2];
    }
    return imageOut;
}

void blurIteration2(AccurateImage* image, v4Accurate* scratch) {
	
	const int width = image->x;
	const int height = image->y;

	v4Accurate sum;

	for(int i = 0; i < BLUR_ITERATIONS; i++) {
		for(int y = 0; y < height; y++) {
			sum = image->data[y * width + 0];
			sum += image->data[y * width + 1];
			sum += image->data[y * width + 2];
			scratch[y * width + 0] = sum / 3;
			sum += image->data[y * width + 3];
			scratch[y * width + 1] = sum / 4;
			sum += image->data[y * width + 4];
			scratch[y * width + 2] = sum / DIVISOR2;

			for(int x = 3; x < width - 2; x++) {
				sum -= image->data[y * width + x - 3];
				sum += image->data[y * width + x + 2];
				scratch[y * width + x] = sum / DIVISOR2;
			}

			sum -= image->data[y * width + width - 5];
			scratch[y * width + width - 2] = sum / 4;
			sum -= image->data[y * width + width - 4];
			scratch[y * width + width - 1] = sum / 3;
		}
		for(int x = 0; x < width; x++) {
			sum = scratch[0 * width + x];
			sum += scratch[1 * width + x];
			sum += scratch[2 * width + x];
			image->data[0 * width + x] = sum / 3;
			sum += scratch[3 * width + x];
			image->data[1 * width + x] = sum / 4;
			sum += scratch[4 * width + x];
			image->data[2 * width + x] = sum / DIVISOR2;

			for(int y = 3; y < height - 2; y++) {
				sum -= scratch[(y - 3) * width + x];
				sum += scratch[(y + 2) * width + x];
				image->data[y * width + x] = sum / DIVISOR2;
			}

			sum -= scratch[(height - 5) * width + x];
			image->data[(height - 2) * width + x] = sum / 4;
			sum -= scratch[(height - 4) * width + x];
			image->data[(height - 1) * width + x] = sum / 3;
		}
	}
}

void blurIteration3(AccurateImage* image, v4Accurate* scratch) {
	
	const int width = image->x;
	const int height = image->y;

	v4Accurate sum;

	for(int i = 0; i < BLUR_ITERATIONS; i++) {
		for(int y = 0; y < height; y++) {
			sum = image->data[y * width + 0];
			sum += image->data[y * width + 1];
			sum += image->data[y * width + 2];
			sum += image->data[y * width + 3];
			scratch[y * width + 0] = sum / 4;
			sum += image->data[y * width + 4];
			scratch[y * width + 1] = sum / 5;
			sum += image->data[y * width + 5];
			scratch[y * width + 2] = sum / 6;
			sum += image->data[y * width + 6];
			scratch[y * width + 3] = sum / DIVISOR3;

			for(int x = 4; x < width - 3; x++) {
				sum -= image->data[y * width + x - 4];
				sum += image->data[y * width + x + 3];
				scratch[y * width + x] = sum / DIVISOR3;
			}

			sum -= image->data[y * width + width - 7];
			scratch[y * width + width - 3] = sum / 6;
			sum -= image->data[y * width + width - 6];
			scratch[y * width + width - 2] = sum / 5;
			sum -= image->data[y * width + width - 5];
			scratch[y * width + width - 1] = sum / 4;
		}
		for(int x = 0; x < width; x++) {
			sum = scratch[0 * width + x];
			sum += scratch[1 * width + x];
			sum += scratch[2 * width + x];
			sum += scratch[3 * width + x];
			image->data[0 * width + x] = sum / 4;
			sum += scratch[4 * width + x];
			image->data[1 * width + x] = sum / 5;
			sum += scratch[5 * width + x];
			image->data[2 * width + x] = sum / 6;
			sum += scratch[6 * width + x];
			image->data[3 * width + x] = sum / DIVISOR3;

			for(int y = 4; y < height - 3; y++) {
				sum -= scratch[(y - 4) * width + x];
				sum += scratch[(y + 3) * width + x];
				image->data[y * width + x] = sum / DIVISOR3;
			}

			sum -= scratch[(height - 7) * width + x];
			image->data[(height - 3) * width + x] = sum / 6;
			sum -= scratch[(height - 6) * width + x];
			image->data[(height - 2) * width + x] = sum / 5;
			sum -= scratch[(height - 5) * width + x];
			image->data[(height - 1) * width + x] = sum / 4;
		}
	}
}

void blurIteration5(AccurateImage* image, v4Accurate* scratch) {
	
	const int width = image->x;
	const int height = image->y;

	v4Accurate sum;

	for(int i = 0; i < BLUR_ITERATIONS; i++) {
		for(int y = 0; y < height; y++) {
			sum = image->data[y * width + 0];
			sum += image->data[y * width + 1];
			sum += image->data[y * width + 2];
			sum += image->data[y * width + 3];
			sum += image->data[y * width + 4];
			sum += image->data[y * width + 5];
			scratch[y * width + 0] = sum / 6;
			sum += image->data[y * width + 6];
			scratch[y * width + 1] = sum / 7;
			sum += image->data[y * width + 7];
			scratch[y * width + 2] = sum / 8;
			sum += image->data[y * width + 8];
			scratch[y * width + 3] = sum / 9;
			sum += image->data[y * width + 9];
			scratch[y * width + 4] = sum / 10;
			sum += image->data[y * width + 10];
			scratch[y * width + 5] = sum / DIVISOR5;

			for(int x = 6; x < width - 5; x++) {
				sum -= image->data[y * width + x - 6];
				sum += image->data[y * width + x + 5];
				scratch[y * width + x] = sum / DIVISOR5;
			}

			sum -= image->data[y * width + width - 11];
			scratch[y * width + width - 5] = sum / 10;
			sum -= image->data[y * width + width - 10];
			scratch[y * width + width - 4] = sum / 9;
			sum -= image->data[y * width + width - 9];
			scratch[y * width + width - 3] = sum / 8;
			sum -= image->data[y * width + width - 8];
			scratch[y * width + width - 2] = sum / 7;
			sum -= image->data[y * width + width - 7];
			scratch[y * width + width - 1] = sum / 6;
		}
		for(int x = 0; x < width; x++) {
			sum = scratch[0 * width + x];
			sum += scratch[1 * width + x];
			sum += scratch[2 * width + x];
			sum += scratch[3 * width + x];
			sum += scratch[4 * width + x];
			sum += scratch[5 * width + x];
			image->data[0 * width + x] = sum / 6;
			sum += scratch[6 * width + x];
			image->data[1 * width + x] = sum / 7;
			sum += scratch[7 * width + x];
			image->data[2 * width + x] = sum / 8;
			sum += scratch[8 * width + x];
			image->data[3 * width + x] = sum / 9;
			sum += scratch[9 * width + x];
			image->data[4 * width + x] = sum / 10;
			sum += scratch[10 * width + x];
			image->data[5 * width + x] = sum / DIVISOR5;

			for(int y = 6; y < height - 5; y++) {
				sum -= scratch[(y - 6) * width + x];
				sum += scratch[(y + 5) * width + x];
				image->data[y * width + x] = sum / DIVISOR5;
			}

			sum -= scratch[(height - 11) * width + x];
			image->data[(height - 5) * width + x] = sum / 10;
			sum -= scratch[(height - 10) * width + x];
			image->data[(height - 4) * width + x] = sum / 9;
			sum -= scratch[(height - 9) * width + x];
			image->data[(height - 3) * width + x] = sum / 8;
			sum -= scratch[(height - 8) * width + x];
			image->data[(height - 2) * width + x] = sum / 7;
			sum -= scratch[(height - 7) * width + x];
			image->data[(height - 1) * width + x] = sum / 6;
		}
	}
}

void blurIteration8(AccurateImage* image, v4Accurate* scratch) {
	
	const int width = image->x;
	const int height = image->y;

	v4Accurate sum;
	
	for(int i = 0; i < BLUR_ITERATIONS; i++) {
		for(int y = 0; y < height; y++) {
			sum = image->data[y * width + 0];
			sum += image->data[y * width + 1];
			sum += image->data[y * width + 2];
			sum += image->data[y * width + 3];
			sum += image->data[y * width + 4];
			sum += image->data[y * width + 5];
			sum += image->data[y * width + 6];
			sum += image->data[y * width + 7];
			sum += image->data[y * width + 8];
			scratch[y * width + 0] = sum / 9;
			sum += image->data[y * width + 9];
			scratch[y * width + 1] = sum / 10;
			sum += image->data[y * width + 10];
			scratch[y * width + 2] = sum / 11;
			sum += image->data[y * width + 11];
			scratch[y * width + 3] = sum / 12;
			sum += image->data[y * width + 12];
			scratch[y * width + 4] = sum / 13;
			sum += image->data[y * width + 13];
			scratch[y * width + 5] = sum / 14;
			sum += image->data[y * width + 14];
			scratch[y * width + 6] = sum / 15;
			sum += image->data[y * width + 15];
			scratch[y * width + 7] = sum / 16;
			sum += image->data[y * width + 16];
			scratch[y * width + 8] = sum / DIVISOR8;

			for(int x = 9; x < width - 8; x++) {
				sum -= image->data[y * width + x - 9];
				sum += image->data[y * width + x + 8];
				scratch[y * width + x] = sum / DIVISOR8;
			}

			sum -= image->data[y * width + width - 17];
			scratch[y * width + width - 8] = sum / 16;
			sum -= image->data[y * width + width - 16];
			scratch[y * width + width - 7] = sum / 15;
			sum -= image->data[y * width + width - 15];
			scratch[y * width + width - 6] = sum / 14;
			sum -= image->data[y * width + width - 14];
			scratch[y * width + width - 5] = sum / 13;
			sum -= image->data[y * width + width - 13];
			scratch[y * width + width - 4] = sum / 12;
			sum -= image->data[y * width + width - 12];
			scratch[y * width + width - 3] = sum / 11;
			sum -= image->data[y * width + width - 11];
			scratch[y * width + width - 2] = sum / 10;
			sum -= image->data[y * width + width - 10];
			scratch[y * width + width - 1] = sum / 9;
		}
		for(int x = 0; x < width; x++) {
			sum = scratch[0 * width + x];
			sum += scratch[1 * width + x];
			sum += scratch[2 * width + x];
			sum += scratch[3 * width + x];
			sum += scratch[4 * width + x];
			sum += scratch[5 * width + x];
			sum += scratch[6 * width + x];
			sum += scratch[7 * width + x];
			sum += scratch[8 * width + x];
			image->data[0 * width + x] = sum / 9;
			sum += scratch[9 * width + x];
			image->data[1 * width + x] = sum / 10;
			sum += scratch[10 * width + x];
			image->data[2 * width + x] = sum / 11;
			sum += scratch[11 * width + x];
			image->data[3 * width + x] = sum / 12;
			sum += scratch[12 * width + x];
			image->data[4 * width + x] = sum / 13;
			sum += scratch[13 * width + x];
			image->data[5 * width + x] = sum / 14;
			sum += scratch[14 * width + x];
			image->data[6 * width + x] = sum / 15;
			sum += scratch[15 * width + x];
			image->data[7 * width + x] = sum / 16;
			sum += scratch[16 * width + x];
			image->data[8 * width + x] = sum / DIVISOR8;

			for(int y = 9; y < height - 8; y++) {
				sum -= scratch[(y - 9) * width + x];
				sum += scratch[(y + 8) * width + x];
				image->data[y * width + x] = sum / DIVISOR8;
			}

			sum -= scratch[(height - 17) * width + x];
			image->data[(height - 8) * width + x] = sum / 16;
			sum -= scratch[(height - 16) * width + x];
			image->data[(height - 7) * width + x] = sum / 15;
			sum -= scratch[(height - 15) * width + x];
			image->data[(height - 6) * width + x] = sum / 14;
			sum -= scratch[(height - 14) * width + x];
			image->data[(height - 5) * width + x] = sum / 13;
			sum -= scratch[(height - 13) * width + x];
			image->data[(height - 4) * width + x] = sum / 12;
			sum -= scratch[(height - 12) * width + x];
			image->data[(height - 3) * width + x] = sum / 11;
			sum -= scratch[(height - 11) * width + x];
			image->data[(height - 2) * width + x] = sum / 10;
			sum -= scratch[(height - 10) * width + x];
			image->data[(height - 1) * width + x] = sum / 9;
		}
	}
}

PPMImage* imageDifference(const AccurateImage* imageInSmall, const AccurateImage* imageInLarge) {
	const int width = imageInSmall->x;
	const int height = imageInSmall->y;
	const int size = width * height;

	PPMImage* imageOut = (PPMImage*)malloc(sizeof(PPMImage));
	imageOut->data = (PPMPixel*)malloc(size * sizeof(PPMPixel));

	imageOut->x = width;
	imageOut->y = height;

	for(int i = 0; i < size; i++) {
		v4Accurate diffvec = imageInLarge->data[i] - imageInSmall->data[i];
		float red = diffvec[0];
		float green = diffvec[1];
		float blue = diffvec[2];
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
	v4Accurate* scratch[4] = {(v4Accurate*)malloc(image->x * image->y * sizeof(v4Accurate)),
							  (v4Accurate*)malloc(image->x * image->y * sizeof(v4Accurate)),
							  (v4Accurate*)malloc(image->x * image->y * sizeof(v4Accurate)),
							  (v4Accurate*)malloc(image->x * image->y * sizeof(v4Accurate))};

	AccurateImage* images[4] = {imageAccurate1_tiny, imageAccurate1_small, imageAccurate1_medium, imageAccurate1_large};
	void (*funcs[4])(AccurateImage*, v4Accurate*) = {&blurIteration2, &blurIteration3, &blurIteration5, &blurIteration8};
	
	#pragma omp parallel for num_threads(4)
	for(int variant = 0; variant < 4; variant++) {
		(*funcs[variant])(images[variant], scratch[variant]);
	}

	PPMImage* imagesPPM[3];

	#pragma omp parallel for num_threads(3)
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

	// free(imageAccurate1_tiny->data);
	// free(imageAccurate1_tiny);
	// free(imageAccurate1_small->data);
	// free(imageAccurate1_small);
	// free(imageAccurate1_medium->data);
	// free(imageAccurate1_medium);
	// free(imageAccurate1_large->data);
	// free(imageAccurate1_large);
	// free(scratch->data);
	// free(scratch);
	// free(image->data);
	// free(image);
	// free(final_tiny->data);
	// free(final_tiny);
	// free(final_small->data);
	// free(final_small);
	// free(final_medium->data);
	// free(final_medium);
	return 0;
}

