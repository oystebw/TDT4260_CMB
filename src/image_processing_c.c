#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <omp.h>

#include "ppm.h"

typedef float v4Accurate __attribute__((vector_size(16)));
typedef __uint32_t v4Int __attribute__((vector_size(16)));

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

typedef struct {
     int x, y;
     v4Accurate* data;
} AccurateImage;


void blurIterationHorizontalFirst(const PPMPixel* restrict in, v4Accurate* restrict out, const int size, const int width, const int height) {
	const register v4Accurate divisor = (v4Accurate){1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1)};
	#pragma omp parallel for schedule(dynamic, 32) num_threads(8)
	for(int y = 0; y < height; y++) {
		const int yWidth = y * width;

		v4Int sum = {0, 0, 0, 0};

		for(int x = 0; x <= size; x++) {
			PPMPixel pixel = in[yWidth + x];
			sum += (v4Int){pixel.red, pixel.green, pixel.blue, 0.0};
		}

		out[yWidth + 0] = (v4Accurate){sum[0], sum[1], sum[2], sum[3]} / (v4Accurate){size + 1, size + 1, size + 1, size + 1};

		for(int x = 1; x <= size; x++) {
			PPMPixel pixel = in[yWidth + x + size];
			sum += (v4Int){pixel.red, pixel.green, pixel.blue, 0.0};
			out[yWidth + x] = (v4Accurate){sum[0], sum[1], sum[2], sum[3]} / (v4Accurate){size + x + 1, size + x + 1, size + x + 1, size + x + 1};
		}

		for(int x = size + 1; x < width - size; x++) {
			PPMPixel pixelMinus = in[yWidth + x - size - 1];
			PPMPixel pixelPlus = in[yWidth + x + size];
			sum -= (v4Int){pixelMinus.red, pixelMinus.green, pixelMinus.blue, 0.0};
			sum += (v4Int){pixelPlus.red, pixelPlus.green, pixelPlus.blue, 0.0};
			out[yWidth + x] = (v4Accurate){sum[0], sum[1], sum[2], sum[3]} * divisor;
		}

		for(int x = width - size; x < width; x++) {
			PPMPixel pixel = in[yWidth + x - size - 1];
			sum -= (v4Int){pixel.red, pixel.green, pixel.blue, 0.0};
			out[yWidth + x] = (v4Accurate){sum[0], sum[1], sum[2], sum[3]} / (v4Accurate){size + width - x, size + width - x, size + width - x, size + width - x};
		}
	}
}

void blurIterationHorizontal(v4Accurate* restrict in, v4Accurate* restrict out, const int size, const int width, const int height) {
	const register v4Accurate divisor = (v4Accurate){1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1)};
	#pragma omp parallel for schedule(dynamic, 32) num_threads(8)
	for(int y = 0; y < height; y++) {
		const int yWidth = y * width;
		for(int iteration = 0; iteration < 3; iteration++) {
			
			v4Accurate sum = {0.0, 0.0, 0.0, 0.0};

			for(int x = 0; x <= size; x++) {
				sum += in[yWidth + x];
			}

			out[yWidth + 0] = sum / (v4Accurate){size + 1, size + 1, size + 1, size + 1};

			for(int x = 1; x <= size; x++) {
				sum += in[yWidth + x + size];
				out[yWidth + x] = sum / (v4Accurate){size + x + 1, size + x + 1, size + x + 1, size + x + 1};
			}

			for(int x = size + 1; x < width - size; x++) {
				
				sum -= in[yWidth + x - size - 1];
				sum += in[yWidth + x + size];
				out[yWidth + x] = sum * divisor;
			}

			for(int x = width - size; x < width; x++) {
				sum -= in[yWidth + x - size - 1];
				out[yWidth + x] = sum / (v4Accurate){size + width - x, size + width - x, size + width - x, size + width - x};
			}

			// swap in and out
			v4Accurate* tmp = in;
			in = out;
			out = tmp;
		}
		// swap in and out
		v4Accurate* tmp = in;
		in = out;
		out = tmp;
	}
}

void blurIterationHorizontalTranspose(const v4Accurate* restrict in, v4Accurate* restrict out, const int size, const int width, const int height) {
	const register v4Accurate divisor = (v4Accurate){1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1)};
	#pragma omp parallel for schedule(dynamic, 32) num_threads(8)
	for(int y = 0; y < height; y++) {
		const int yWidth = y * width;

		v4Accurate sum = {0.0, 0.0, 0.0, 0.0};

		for(int x = 0; x <= size; x++) {
			sum += in[yWidth + x];
		}

		out[0 * height + y] = sum / (v4Accurate){size + 1, size + 1, size + 1, size + 1};

		for(int x = 1; x <= size; x++) {
			sum += in[yWidth + x + size];
			out[x * height + y] = sum / (v4Accurate){size + x + 1, size + x + 1, size + x + 1, size + x + 1};
		}

		for(int x = size + 1; x < width - size; x++) {
			sum -= in[yWidth + x - size - 1];
			sum += in[yWidth + x + size];
			out[x * height + y] = sum * divisor;
		}

		for(int x = width - size; x < width; x++) {
			sum -= in[yWidth + x - size - 1];
			out[x * height + y] = sum / (v4Accurate){size + width - x, size + width - x, size + width - x, size + width - x};
		}
	}
}

void blurIterationVertical(v4Accurate* restrict in, v4Accurate* restrict out, const int size, const int width, const int height) {
	const register v4Accurate divisor = (v4Accurate){1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1)};
	#pragma omp parallel for schedule(dynamic, 32) num_threads(8)
	for(int x = 0; x < width; x++) {
		const int xHeight = x * height;
		for(int iteration = 0; iteration < 5; iteration++) {
			
			v4Accurate sum = {0.0, 0.0, 0.0, 0.0};

			for(int y = 0; y <= size; y++) {
				sum += in[xHeight + y];
			}

			out[xHeight + 0] = sum / (v4Accurate){size + 1, size + 1, size + 1, size + 1};

			for(int y = 1; y <= size; y++) {
				sum += in[xHeight + y + size];
				out[xHeight + y] = sum / (v4Accurate){y + size + 1, y + size + 1, y + size + 1, y + size + 1};
			}

			for(int y = size + 1; y < height - size; y++) {
				sum -= in[xHeight + y - size - 1];
				sum += in[xHeight + y + size];
				out[xHeight + y] = sum * divisor;
			}

			for(int y = height - size; y < height; y++) {
				sum -= in[xHeight + y - size - 1];
				out[xHeight + y] = sum / (v4Accurate){size + height - y, size + height - y, size + height - y, size + height - y};
			}
			// swap
			v4Accurate* tmp = in;
			in = out;
			out = tmp;
		}
		// swap
		v4Accurate* tmp = in;
		in = out;
		out = tmp;
	}
}


PPMImage* imageDifference(const AccurateImage* restrict imageInSmall, const AccurateImage* restrict imageInLarge) {
	const int width = imageInSmall->x;
	const int height = imageInSmall->y;
	const int size = width * height;

	PPMImage* restrict imageOut = (PPMImage*)malloc(sizeof(PPMImage));
	imageOut->data = (PPMPixel* restrict)malloc(sizeof(PPMPixel) * size);

	imageOut->x = width;
	imageOut->y = height;
	#pragma omp parallel for schedule(dynamic, 32) num_threads(8)
	for(int x = 0; x < width; x++) {
		const int xHeight = x * height;
		for(int y = 0; y < height; y++) {
			v4Accurate diffvec = imageInLarge->data[xHeight + y] - imageInSmall->data[xHeight + y];
			float red = diffvec[0];
			float green = diffvec[1];
			float blue = diffvec[2];

			red = red < 0.0 ? red + 257.0 : red;
			green = green < 0.0 ? green + 257.0 : green;
			blue = blue < 0.0 ? blue + 257.0 : blue;

			imageOut->data[y * width + x] = (PPMPixel){red, green, blue};
		}
	}
	return imageOut;
}

int main(int argc, char** argv) {

    const PPMImage* restrict image = (argc > 1) ? readPPM("flower.ppm") : readStreamPPM(stdin);

	const int width = image->x;
	const int height = image->y;
	const int size = width * height;

	const int sizes[4] = {2, 3, 5, 8};

	AccurateImage** restrict images = (AccurateImage** restrict)malloc(sizeof(AccurateImage*) * 4);
	v4Accurate* restrict scratches = (v4Accurate* restrict)malloc(sizeof(v4Accurate) * size * 4);

	for(int i = 0; i < 4; i++) {
		images[i] = (AccurateImage* restrict)malloc(sizeof(AccurateImage));
		images[i]->x = width;
		images[i]->y = height;
		images[i]->data = (v4Accurate* restrict)malloc(sizeof(v4Accurate) * size);
	}

	PPMImage* imagesPPM[3];

	for(int i = 0; i < 4; i++) {
		blurIterationHorizontalFirst(image->data, scratches + i * size, sizes[i], width, height);
		blurIterationHorizontal(scratches + i * size, images[i]->data, sizes[i], width, height);
		blurIterationHorizontalTranspose(images[i]->data, scratches + i * size, sizes[i], width, height);
		blurIterationVertical(scratches + i * size, images[i]->data, sizes[i], width, height);
	}

	for(int i = 0; i < 3; i++) {
		imagesPPM[i] = imageDifference(images[i], images[i + 1]);
	}

	(argc > 1) ? writePPM("flower_tiny.ppm", imagesPPM[0]) : writeStreamPPM(stdout, imagesPPM[0]);
	(argc > 1) ? writePPM("flower_small.ppm", imagesPPM[1]) : writeStreamPPM(stdout, imagesPPM[1]);
	(argc > 1) ? writePPM("flower_medium.ppm", imagesPPM[2]) : writeStreamPPM(stdout, imagesPPM[2]);
}

