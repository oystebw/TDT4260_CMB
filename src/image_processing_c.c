#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <emmintrin.h>

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
	const int size = image->x * image->y;

	AccurateImage* imageAccurate = (AccurateImage*)malloc(sizeof(AccurateImage));
	imageAccurate->data = (v4Accurate*)malloc(size * sizeof(v4Accurate));
	#pragma GCC unroll 8
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
	#pragma GCC unroll 8
    for(int i = 0; i < size; i++) {
		imageOut->data[i].red = imageIn->data[i][0];
		imageOut->data[i].green = imageIn->data[i][1];
		imageOut->data[i].blue = imageIn->data[i][2];
    }
    return imageOut;
}

void blurIteration(AccurateImage* image, const int size) {
	const int width = image->x;
	const int height = image->y;
	v4Accurate* scratch = (v4Accurate*)malloc(width * height * sizeof(v4Accurate));

	// Transpose to be more cache / access friendly
	v4Accurate (*data)   [width] = (void*) image->data;
  	v4Accurate (*buffer) [height] = (void*) scratch;
	

	for(int i = 0; i < BLUR_ITERATIONS; i++) {
		#pragma GCC unroll 8
		for(int y = 0; y < height; y++) {

			v4Accurate sum = {0.0, 0.0, 0.0, 0.0};

			for(int x = 0; x <= size; x++) {
				sum += data[y][x];
			}

			buffer[0][y] = sum / (v4Accurate){size + 1, size + 1, size + 1, size + 1};

			for(int x = 1; x <= size; x++) {
				sum += data[y][x + size];
				buffer[x][y] = sum / (v4Accurate){size + x + 1, size + x + 1, size + x + 1, size + x + 1};
			}

			for(int x = size + 1; x < width - size; x++) {
				sum -= data[y][x - size - 1];
				sum += data[y][x + size];
				buffer[x][y] = sum / (v4Accurate){2 * size + 1, 2 * size + 1, 2 * size + 1, 2 * size + 1};
			}

			for(int x = width - size; x < width; x++) {
				sum -= data[y][x - size - 1];
				buffer[x][y] = sum / (v4Accurate){size + width - x, size + width - x, size + width - x, size + width - x};
			}
			
		}
		#pragma GCC unroll 8
		for(int x = 0; x < width; x++) {

			v4Accurate sum = {0.0, 0.0, 0.0, 0.0};

			for(int y = 0; y <= size; y++) {
				sum += buffer[x][y];
			}

			data[0][x] = sum / (v4Accurate){size + 1, size + 1, size + 1, size + 1};

			for(int y = 1; y <= size; y++) {
				sum += buffer[x][(y + size)];
				data[y][x] = sum / (v4Accurate){y + size + 1, y + size + 1, y + size + 1, y + size + 1};
			}

			for(int y = size + 1; y < height - size; y++) {
				sum -= buffer[x][y - size - 1];
				sum += buffer[x][y + size];
				data[y][x] = sum / (v4Accurate){2 * size + 1, 2 * size + 1, 2 * size + 1, 2 * size + 1};
			}

			for(int y = height - size; y < height; y++) {
				sum -= buffer[x][y - size - 1];
				data[y][x] = sum / (v4Accurate){size + height - y, size + height - y, size + height - y, size + height - y};
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

