#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <omp.h>

#include "ppm.h"

#define BLUR_ITERATIONS 5

#define DIVISOR2 5
#define DIVISOR3 7
#define DIVISOR5 11
#define DIVISOR8 17

//typedef float v4 __attribute__((vector_size(16)));

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

typedef struct {
	unsigned char rgb[3];
} SimplePixel;

typedef struct {
	int x, y;
	SimplePixel *data;
} SimpleImage;

typedef struct {
     float rgb[3];
} AccuratePixel;

typedef struct {
     int x, y;
     AccuratePixel *data;
} AccurateImage;

AccurateImage* convertToAccurateImage(const PPMImage* image) {
	const int size = image->x * image->y;

	AccurateImage* imageAccurate = (AccurateImage*)malloc(sizeof(AccurateImage));
	imageAccurate->data = (AccuratePixel*)malloc(size * sizeof(AccuratePixel));
	for(int i = 0; i < size; i++) {
		imageAccurate->data[i].rgb[0] = (float) image->data[i].red;
		imageAccurate->data[i].rgb[1] = (float) image->data[i].green;
		imageAccurate->data[i].rgb[2] = (float) image->data[i].blue;
	}
	imageAccurate->x = image->x;
	imageAccurate->y = image->y;
	
	return imageAccurate;
}

AccurateImage* copyAccurateImage(const AccurateImage* imageIn) {
	const int size = imageIn->x * imageIn->y;
	AccurateImage* imageOut = (AccurateImage*)malloc(sizeof(AccurateImage));
	imageOut->data = (AccuratePixel*)malloc(size * sizeof(AccuratePixel));
	
	memcpy(imageOut->data, imageIn->data, size * sizeof(AccuratePixel));

	imageOut->x = imageIn->x;
	imageOut->y = imageIn->y;
	return imageOut;
}

PPMImage* convertToPPPMImage(const AccurateImage* imageIn) {
    const int size = imageIn->x * imageIn->y;
	
	PPMImage *imageOut;
    imageOut = (PPMImage*)malloc(sizeof(PPMImage));
    imageOut->data = (PPMPixel*)malloc(size * sizeof(PPMPixel));

    imageOut->x = imageIn->x;
    imageOut->y = imageIn->y;

    for(int i = 0; i < size; i++) {
        imageOut->data[i].red = imageIn->data[i].rgb[0];
        imageOut->data[i].green = imageIn->data[i].rgb[1];
        imageOut->data[i].blue = imageIn->data[i].rgb[2];
    }
    return imageOut;
}

void blurIteration2(AccurateImage* image, AccuratePixel* scratch, const int colourType) {
	
	const int width = image->x;
	const int height = image->y;

	float sum;

	for(int i = 0; i < BLUR_ITERATIONS; i++) {
		#pragma omp parallel for
		for(int y = 0; y < height; y++) {
			sum = image->data[y * width + 0].rgb[colourType];
			sum += image->data[y * width + 1].rgb[colourType];
			sum += image->data[y * width + 2].rgb[colourType];
			scratch[y * width + 0].rgb[colourType] = sum / 3;
			sum += image->data[y * width + 3].rgb[colourType];
			scratch[y * width + 1].rgb[colourType] = sum / 4;
			sum += image->data[y * width + 4].rgb[colourType];
			scratch[y * width + 2].rgb[colourType] = sum / DIVISOR2;

			for(int x = 3; x < width - 2; x++) {
				sum -= image->data[y * width + x - 3].rgb[colourType];
				sum += image->data[y * width + x + 2].rgb[colourType];
				scratch[y * width + x].rgb[colourType] = sum / DIVISOR2;
			}

			sum -= image->data[y * width + width - 5].rgb[colourType];
			scratch[y * width + width - 2].rgb[colourType] = sum / 4;
			sum -= image->data[y * width + width - 4].rgb[colourType];
			scratch[y * width + width - 1].rgb[colourType] = sum / 3;
		}
		#pragma omp parallel for
		for(int x = 0; x < width; x++) {
			sum = scratch[0 * width + x].rgb[colourType];
			sum += scratch[1 * width + x].rgb[colourType];
			sum += scratch[2 * width + x].rgb[colourType];
			image->data[0 * width + x].rgb[colourType] = sum / 3;
			sum += scratch[3 * width + x].rgb[colourType];
			image->data[1 * width + x].rgb[colourType] = sum / 4;
			sum += scratch[4 * width + x].rgb[colourType];
			image->data[2 * width + x].rgb[colourType] = sum / DIVISOR2;

			for(int y = 3; y < height - 2; y++) {
				sum -= scratch[(y - 3) * width + x].rgb[colourType];
				sum += scratch[(y + 2) * width + x].rgb[colourType];
				image->data[y * width + x].rgb[colourType] = sum / DIVISOR2;
			}

			sum -= scratch[(height - 5) * width + x].rgb[colourType];
			image->data[(height - 2) * width + x].rgb[colourType] = sum / 4;
			sum -= scratch[(height - 4) * width + x].rgb[colourType];
			image->data[(height - 1) * width + x].rgb[colourType] = sum / 3;
		}
	}
}

void blurIteration3(AccurateImage* image, AccuratePixel* scratch, const int colourType) {
	
	const int width = image->x;
	const int height = image->y;

	float sum;

	for(int i = 0; i < BLUR_ITERATIONS; i++) {
		#pragma omp parallel for
		for(int y = 0; y < height; y++) {
			sum = image->data[y * width + 0].rgb[colourType];
			sum += image->data[y * width + 1].rgb[colourType];
			sum += image->data[y * width + 2].rgb[colourType];
			sum += image->data[y * width + 3].rgb[colourType];
			scratch[y * width + 0].rgb[colourType] = sum / 4;
			sum += image->data[y * width + 4].rgb[colourType];
			scratch[y * width + 1].rgb[colourType] = sum / 5;
			sum += image->data[y * width + 5].rgb[colourType];
			scratch[y * width + 2].rgb[colourType] = sum / 6;
			sum += image->data[y * width + 6].rgb[colourType];
			scratch[y * width + 3].rgb[colourType] = sum / DIVISOR3;

			for(int x = 4; x < width - 3; x++) {
				sum -= image->data[y * width + x - 4].rgb[colourType];
				sum += image->data[y * width + x + 3].rgb[colourType];
				scratch[y * width + x].rgb[colourType] = sum / DIVISOR3;
			}

			sum -= image->data[y * width + width - 7].rgb[colourType];
			scratch[y * width + width - 3].rgb[colourType] = sum / 6;
			sum -= image->data[y * width + width - 6].rgb[colourType];
			scratch[y * width + width - 2].rgb[colourType] = sum / 5;
			sum -= image->data[y * width + width - 5].rgb[colourType];
			scratch[y * width + width - 1].rgb[colourType] = sum / 4;
		}
		#pragma omp parallel for
		for(int x = 0; x < width; x++) {
			sum = scratch[0 * width + x].rgb[colourType];
			sum += scratch[1 * width + x].rgb[colourType];
			sum += scratch[2 * width + x].rgb[colourType];
			sum += scratch[3 * width + x].rgb[colourType];
			image->data[0 * width + x].rgb[colourType] = sum / 4;
			sum += scratch[4 * width + x].rgb[colourType];
			image->data[1 * width + x].rgb[colourType] = sum / 5;
			sum += scratch[5 * width + x].rgb[colourType];
			image->data[2 * width + x].rgb[colourType] = sum / 6;
			sum += scratch[6 * width + x].rgb[colourType];
			image->data[3 * width + x].rgb[colourType] = sum / DIVISOR3;

			for(int y = 4; y < height - 3; y++) {
				sum -= scratch[(y - 4) * width + x].rgb[colourType];
				sum += scratch[(y + 3) * width + x].rgb[colourType];
				image->data[y * width + x].rgb[colourType] = sum / DIVISOR3;
			}

			sum -= scratch[(height - 7) * width + x].rgb[colourType];
			image->data[(height - 3) * width + x].rgb[colourType] = sum / 6;
			sum -= scratch[(height - 6) * width + x].rgb[colourType];
			image->data[(height - 2) * width + x].rgb[colourType] = sum / 5;
			sum -= scratch[(height - 5) * width + x].rgb[colourType];
			image->data[(height - 1) * width + x].rgb[colourType] = sum / 4;
		}
	}
}

void blurIteration5(AccurateImage* image, AccuratePixel* scratch, const int colourType) {
	
	const int width = image->x;
	const int height = image->y;

	float sum;

	for(int i = 0; i < BLUR_ITERATIONS; i++) {
		#pragma omp parallel for
		for(int y = 0; y < height; y++) {
			sum = image->data[y * width + 0].rgb[colourType];
			sum += image->data[y * width + 1].rgb[colourType];
			sum += image->data[y * width + 2].rgb[colourType];
			sum += image->data[y * width + 3].rgb[colourType];
			sum += image->data[y * width + 4].rgb[colourType];
			sum += image->data[y * width + 5].rgb[colourType];
			scratch[y * width + 0].rgb[colourType] = sum / 6;
			sum += image->data[y * width + 6].rgb[colourType];
			scratch[y * width + 1].rgb[colourType] = sum / 7;
			sum += image->data[y * width + 7].rgb[colourType];
			scratch[y * width + 2].rgb[colourType] = sum / 8;
			sum += image->data[y * width + 8].rgb[colourType];
			scratch[y * width + 3].rgb[colourType] = sum / 9;
			sum += image->data[y * width + 9].rgb[colourType];
			scratch[y * width + 4].rgb[colourType] = sum / 10;
			sum += image->data[y * width + 10].rgb[colourType];
			scratch[y * width + 5].rgb[colourType] = sum / DIVISOR5;

			for(int x = 6; x < width - 5; x++) {
				sum -= image->data[y * width + x - 6].rgb[colourType];
				sum += image->data[y * width + x + 5].rgb[colourType];
				scratch[y * width + x].rgb[colourType] = sum / DIVISOR5;
			}

			sum -= image->data[y * width + width - 11].rgb[colourType];
			scratch[y * width + width - 5].rgb[colourType] = sum / 10;
			sum -= image->data[y * width + width - 10].rgb[colourType];
			scratch[y * width + width - 4].rgb[colourType] = sum / 9;
			sum -= image->data[y * width + width - 9].rgb[colourType];
			scratch[y * width + width - 3].rgb[colourType] = sum / 8;
			sum -= image->data[y * width + width - 8].rgb[colourType];
			scratch[y * width + width - 2].rgb[colourType] = sum / 7;
			sum -= image->data[y * width + width - 7].rgb[colourType];
			scratch[y * width + width - 1].rgb[colourType] = sum / 6;
		}
		#pragma omp parallel for
		for(int x = 0; x < width; x++) {
			sum = scratch[0 * width + x].rgb[colourType];
			sum += scratch[1 * width + x].rgb[colourType];
			sum += scratch[2 * width + x].rgb[colourType];
			sum += scratch[3 * width + x].rgb[colourType];
			sum += scratch[4 * width + x].rgb[colourType];
			sum += scratch[5 * width + x].rgb[colourType];
			image->data[0 * width + x].rgb[colourType] = sum / 6;
			sum += scratch[6 * width + x].rgb[colourType];
			image->data[1 * width + x].rgb[colourType] = sum / 7;
			sum += scratch[7 * width + x].rgb[colourType];
			image->data[2 * width + x].rgb[colourType] = sum / 8;
			sum += scratch[8 * width + x].rgb[colourType];
			image->data[3 * width + x].rgb[colourType] = sum / 9;
			sum += scratch[9 * width + x].rgb[colourType];
			image->data[4 * width + x].rgb[colourType] = sum / 10;
			sum += scratch[10 * width + x].rgb[colourType];
			image->data[5 * width + x].rgb[colourType] = sum / DIVISOR5;

			for(int y = 6; y < height - 5; y++) {
				sum -= scratch[(y - 6) * width + x].rgb[colourType];
				sum += scratch[(y + 5) * width + x].rgb[colourType];
				image->data[y * width + x].rgb[colourType] = sum / DIVISOR5;
			}

			sum -= scratch[(height - 11) * width + x].rgb[colourType];
			image->data[(height - 5) * width + x].rgb[colourType] = sum / 10;
			sum -= scratch[(height - 10) * width + x].rgb[colourType];
			image->data[(height - 4) * width + x].rgb[colourType] = sum / 9;
			sum -= scratch[(height - 9) * width + x].rgb[colourType];
			image->data[(height - 3) * width + x].rgb[colourType] = sum / 8;
			sum -= scratch[(height - 8) * width + x].rgb[colourType];
			image->data[(height - 2) * width + x].rgb[colourType] = sum / 7;
			sum -= scratch[(height - 7) * width + x].rgb[colourType];
			image->data[(height - 1) * width + x].rgb[colourType] = sum / 6;
		}
	}
}

void blurIteration8(AccurateImage* image, AccuratePixel* scratch, const int colourType) {
	
	const int width = image->x;
	const int height = image->y;

	float sum;
	
	for(int i = 0; i < BLUR_ITERATIONS; i++) {
		#pragma omp parallel for
		for(int y = 0; y < height; y++) {
			sum = image->data[y * width + 0].rgb[colourType];
			sum += image->data[y * width + 1].rgb[colourType];
			sum += image->data[y * width + 2].rgb[colourType];
			sum += image->data[y * width + 3].rgb[colourType];
			sum += image->data[y * width + 4].rgb[colourType];
			sum += image->data[y * width + 5].rgb[colourType];
			sum += image->data[y * width + 6].rgb[colourType];
			sum += image->data[y * width + 7].rgb[colourType];
			sum += image->data[y * width + 8].rgb[colourType];
			scratch[y * width + 0].rgb[colourType] = sum / 9;
			sum += image->data[y * width + 9].rgb[colourType];
			scratch[y * width + 1].rgb[colourType] = sum / 10;
			sum += image->data[y * width + 10].rgb[colourType];
			scratch[y * width + 2].rgb[colourType] = sum / 11;
			sum += image->data[y * width + 11].rgb[colourType];
			scratch[y * width + 3].rgb[colourType] = sum / 12;
			sum += image->data[y * width + 12].rgb[colourType];
			scratch[y * width + 4].rgb[colourType] = sum / 13;
			sum += image->data[y * width + 13].rgb[colourType];
			scratch[y * width + 5].rgb[colourType] = sum / 14;
			sum += image->data[y * width + 14].rgb[colourType];
			scratch[y * width + 6].rgb[colourType] = sum / 15;
			sum += image->data[y * width + 15].rgb[colourType];
			scratch[y * width + 7].rgb[colourType] = sum / 16;
			sum += image->data[y * width + 16].rgb[colourType];
			scratch[y * width + 8].rgb[colourType] = sum / DIVISOR8;

			for(int x = 9; x < width - 8; x++) {
				sum -= image->data[y * width + x - 9].rgb[colourType];
				sum += image->data[y * width + x + 8].rgb[colourType];
				scratch[y * width + x].rgb[colourType] = sum / DIVISOR8;
			}

			sum -= image->data[y * width + width - 17].rgb[colourType];
			scratch[y * width + width - 8].rgb[colourType] = sum / 16;
			sum -= image->data[y * width + width - 16].rgb[colourType];
			scratch[y * width + width - 7].rgb[colourType] = sum / 15;
			sum -= image->data[y * width + width - 15].rgb[colourType];
			scratch[y * width + width - 6].rgb[colourType] = sum / 14;
			sum -= image->data[y * width + width - 14].rgb[colourType];
			scratch[y * width + width - 5].rgb[colourType] = sum / 13;
			sum -= image->data[y * width + width - 13].rgb[colourType];
			scratch[y * width + width - 4].rgb[colourType] = sum / 12;
			sum -= image->data[y * width + width - 12].rgb[colourType];
			scratch[y * width + width - 3].rgb[colourType] = sum / 11;
			sum -= image->data[y * width + width - 11].rgb[colourType];
			scratch[y * width + width - 2].rgb[colourType] = sum / 10;
			sum -= image->data[y * width + width - 10].rgb[colourType];
			scratch[y * width + width - 1].rgb[colourType] = sum / 9;
		}
		#pragma omp parallel for
		for(int x = 0; x < width; x++) {
			sum = scratch[0 * width + x].rgb[colourType];
			sum += scratch[1 * width + x].rgb[colourType];
			sum += scratch[2 * width + x].rgb[colourType];
			sum += scratch[3 * width + x].rgb[colourType];
			sum += scratch[4 * width + x].rgb[colourType];
			sum += scratch[5 * width + x].rgb[colourType];
			sum += scratch[6 * width + x].rgb[colourType];
			sum += scratch[7 * width + x].rgb[colourType];
			sum += scratch[8 * width + x].rgb[colourType];
			image->data[0 * width + x].rgb[colourType] = sum / 9;
			sum += scratch[9 * width + x].rgb[colourType];
			image->data[1 * width + x].rgb[colourType] = sum / 10;
			sum += scratch[10 * width + x].rgb[colourType];
			image->data[2 * width + x].rgb[colourType] = sum / 11;
			sum += scratch[11 * width + x].rgb[colourType];
			image->data[3 * width + x].rgb[colourType] = sum / 12;
			sum += scratch[12 * width + x].rgb[colourType];
			image->data[4 * width + x].rgb[colourType] = sum / 13;
			sum += scratch[13 * width + x].rgb[colourType];
			image->data[5 * width + x].rgb[colourType] = sum / 14;
			sum += scratch[14 * width + x].rgb[colourType];
			image->data[6 * width + x].rgb[colourType] = sum / 15;
			sum += scratch[15 * width + x].rgb[colourType];
			image->data[7 * width + x].rgb[colourType] = sum / 16;
			sum += scratch[16 * width + x].rgb[colourType];
			image->data[8 * width + x].rgb[colourType] = sum / DIVISOR8;

			for(int y = 9; y < height - 8; y++) {
				sum -= scratch[(y - 9) * width + x].rgb[colourType];
				sum += scratch[(y + 8) * width + x].rgb[colourType];
				image->data[y * width + x].rgb[colourType] = sum / DIVISOR8;
			}

			sum -= scratch[(height - 17) * width + x].rgb[colourType];
			image->data[(height - 8) * width + x].rgb[colourType] = sum / 16;
			sum -= scratch[(height - 16) * width + x].rgb[colourType];
			image->data[(height - 7) * width + x].rgb[colourType] = sum / 15;
			sum -= scratch[(height - 15) * width + x].rgb[colourType];
			image->data[(height - 6) * width + x].rgb[colourType] = sum / 14;
			sum -= scratch[(height - 14) * width + x].rgb[colourType];
			image->data[(height - 5) * width + x].rgb[colourType] = sum / 13;
			sum -= scratch[(height - 13) * width + x].rgb[colourType];
			image->data[(height - 4) * width + x].rgb[colourType] = sum / 12;
			sum -= scratch[(height - 12) * width + x].rgb[colourType];
			image->data[(height - 3) * width + x].rgb[colourType] = sum / 11;
			sum -= scratch[(height - 11) * width + x].rgb[colourType];
			image->data[(height - 2) * width + x].rgb[colourType] = sum / 10;
			sum -= scratch[(height - 10) * width + x].rgb[colourType];
			image->data[(height - 1) * width + x].rgb[colourType] = sum / 9;
		}
	}
}

PPMImage* imageDifference(const AccurateImage* imageInSmall, const AccurateImage* imageInLarge) {
	const int width = imageInSmall->x;
	const int height = imageInSmall->y;
	const int size = width * height;

	PPMImage* imageOut = (PPMImage*)malloc(sizeof(PPMImage));
	SimplePixel* scratch = (SimplePixel*)malloc(sizeof(SimplePixel) * size);

	imageOut->x = width;
	imageOut->y = height;

	for(int color = 0; color < 3; color++) {
		float diff;
		for(int i = 0; i < size; i++) {
			diff = imageInLarge->data[i].rgb[color] - imageInSmall->data[i].rgb[color];
			diff += 257.0 * (diff < 0.0);
			scratch[i].rgb[color] = truncf(diff);
		}	
	}
	imageOut->data = scratch;
	
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
	AccuratePixel* scratch[4] = {(AccuratePixel*)malloc(image->x * image->y * sizeof(AccuratePixel)),
								(AccuratePixel*)malloc(image->x * image->y * sizeof(AccuratePixel)),
								(AccuratePixel*)malloc(image->x * image->y * sizeof(AccuratePixel)),
								(AccuratePixel*)malloc(image->x * image->y * sizeof(AccuratePixel))};

	AccurateImage* images[4] = {imageAccurate1_tiny, imageAccurate1_small, imageAccurate1_medium, imageAccurate1_large};
	void (*funcs[4])(AccurateImage*, AccuratePixel*, const int) = {&blurIteration2, &blurIteration3, &blurIteration5, &blurIteration8};
	
	#pragma omp parallel for
	for(int variant = 0; variant < 12; variant++) {
		(*funcs[variant / 3])(images[variant / 3], scratch[variant / 3], variant % 3);
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

