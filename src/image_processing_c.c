#pragma GCC optimize ("Ofast")

#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <omp.h>
#include "ppm.h"

#define CACHELINESIZE 16
#define BLOCKSIZE 32

typedef float v4Accurate __attribute__((vector_size(16)));
typedef __uint32_t v4Int __attribute__((vector_size(16)));

PPMPixel result_data[1920 * 1200];
v4Accurate one[1920 * 1200];
v4Accurate two[1920 * 1200];
v4Accurate scratch[1920 * 1200];

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html


void blurIterationHorizontalFirst(const PPMPixel* restrict in, v4Accurate* restrict out, const int size, const int width, const int height) {
	const v4Accurate divisor = (v4Accurate){1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1)};
	#pragma omp parallel for schedule(dynamic, 2) num_threads(8)
	for(int y = 0; y < height; ++y) {
		const int yWidth = y * width;

		v4Int sum = {0, 0, 0, 0};

		for(int x = 0; x <= size; ++x) {
			PPMPixel pixel = in[yWidth + x];
			sum += (v4Int){pixel.red, pixel.green, pixel.blue, 0.0};
		}

		out[yWidth + 0] = (v4Accurate){sum[0], sum[1], sum[2], sum[3]} / (v4Accurate){size + 1, size + 1, size + 1, size + 1};

		for(int x = 1; x <= size; ++x) {
			PPMPixel pixel = in[yWidth + x + size];
			sum += (v4Int){pixel.red, pixel.green, pixel.blue, 0.0};
			out[yWidth + x] = (v4Accurate){sum[0], sum[1], sum[2], sum[3]} / (v4Accurate){size + x + 1, size + x + 1, size + x + 1, size + x + 1};
		}

		#pragma GCC unroll 16
		for(int x = size + 1; x < width - size; ++x) {
			PPMPixel pixelMinus = in[yWidth + x - size - 1];
			PPMPixel pixelPlus = in[yWidth + x + size];
			sum -= (v4Int){pixelMinus.red, pixelMinus.green, pixelMinus.blue, 0.0};
			sum += (v4Int){pixelPlus.red, pixelPlus.green, pixelPlus.blue, 0.0};
			out[yWidth + x] = (v4Accurate){sum[0], sum[1], sum[2], sum[3]} * divisor;
		}

		for(int x = width - size; x < width; ++x) {
			PPMPixel pixel = in[yWidth + x - size - 1];
			sum -= (v4Int){pixel.red, pixel.green, pixel.blue, 0.0};
			out[yWidth + x] = (v4Accurate){sum[0], sum[1], sum[2], sum[3]} / (v4Accurate){size + width - x, size + width - x, size + width - x, size + width - x};
		}
	}
}

void blurIterationHorizontal(v4Accurate* restrict in, v4Accurate* restrict out, const int size, const int width, const int height) {
	const v4Accurate divisor = (v4Accurate){1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1)};
	#pragma omp parallel for schedule(dynamic, 2) num_threads(8)
	for(int y = 0; y < height; ++y) {
		const int yWidth = y * width;
		for(int iteration = 0; iteration < 3; ++iteration) {
			
			v4Accurate sum = {0.0, 0.0, 0.0, 0.0};

			for(int x = 0; x <= size; ++x) {
				sum += in[yWidth + x];
			}

			out[yWidth + 0] = sum / (v4Accurate){size + 1, size + 1, size + 1, size + 1};

			for(int x = 1; x <= size; ++x) {
				sum += in[yWidth + x + size];
				out[yWidth + x] = sum / (v4Accurate){size + x + 1, size + x + 1, size + x + 1, size + x + 1};
			}

			#pragma GCC unroll 16
			for(int x = size + 1; x < width - size; ++x) {
				sum -= in[yWidth + x - size - 1];
				sum += in[yWidth + x + size];
				out[yWidth + x] = sum * divisor;
			}

			for(int x = width - size; x < width; ++x) {
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
	const v4Accurate divisor = (v4Accurate){1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1)};
	#pragma omp parallel for schedule(dynamic, 2) num_threads(8)
	for(int y = 0; y < height; ++y) {
		const int yWidth = y * width;

		v4Accurate sum = {0.0, 0.0, 0.0, 0.0};

		for(int x = 0; x <= size; ++x) {
			sum += in[yWidth + x];
		}

		out[0 * height + y] = sum / (v4Accurate){size + 1, size + 1, size + 1, size + 1};

		for(int x = 1; x <= size; ++x) {
			sum += in[yWidth + x + size];
			out[x * height + y] = sum / (v4Accurate){size + x + 1, size + x + 1, size + x + 1, size + x + 1};
		}

		#pragma GCC unroll 16
		for(int x = size + 1; x < width - size; ++x) {
			sum -= in[yWidth + x - size - 1];
			sum += in[yWidth + x + size];
			out[x * height + y] = sum * divisor;
		}

		for(int x = width - size; x < width; ++x) {
			sum -= in[yWidth + x - size - 1];
			out[x * height + y] = sum / (v4Accurate){size + width - x, size + width - x, size + width - x, size + width - x};
		}
	}
}

void blurIterationVertical(v4Accurate* restrict in, v4Accurate* restrict out, const int size, const int width, const int height) {
	const v4Accurate divisor = (v4Accurate){1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1)};
	#pragma omp parallel for schedule(dynamic, 2) num_threads(8)
	for(int x = 0; x < width; ++x) {
		const int xHeight = x * height;
		for(int iteration = 0; iteration < 5; ++iteration) {
			
			v4Accurate sum = {0.0, 0.0, 0.0, 0.0};

			for(int y = 0; y <= size; ++y) {
				sum += in[xHeight + y];
			}

			out[xHeight + 0] = sum / (v4Accurate){size + 1, size + 1, size + 1, size + 1};

			for(int y = 1; y <= size; ++y) {
				sum += in[xHeight + y + size];
				out[xHeight + y] = sum / (v4Accurate){y + size + 1, y + size + 1, y + size + 1, y + size + 1};
			}

			#pragma GCC unroll 16
			for(int y = size + 1; y < height - size; ++y) {
				sum -= in[xHeight + y - size - 1];
				sum += in[xHeight + y + size];
				out[xHeight + y] = sum * divisor;
			}

			for(int y = height - size; y < height; ++y) {
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

void imageDifference(PPMPixel* restrict imageOut, const v4Accurate* restrict small, const v4Accurate* restrict large, const int width, const int height) {
	
	#pragma omp parallel for schedule(dynamic, 2) num_threads(8)
	for(int xx = 0; xx < width; xx += BLOCKSIZE) {
		for(int yy = 0; yy < height; yy += BLOCKSIZE) {
			for(int x = xx; x < xx + BLOCKSIZE; ++x) {
				const int xHeight = x * height;
				for(int y = yy; y < yy + BLOCKSIZE && y < height; ++y) {
					const v4Accurate diff = large[xHeight + y] - small[xHeight + y];

					float red = diff[0];
					float green = diff[1];
					float blue = diff[2];
					red = red < 0.0 ? red + 257.0 : red;
					green = green < 0.0 ? green + 257.0 : green;
					blue = blue < 0.0 ? blue + 257.0 : blue;
					imageOut[y * width + x] = (PPMPixel){red, green, blue};
				}
			}
		}
	}
}

int main(int argc, char** argv) {

    const PPMImage* restrict image = (argc > 1) ? readPPM("flower.ppm") : readStreamPPM(stdin);

	const int width = image->x;
	const int height = image->y;
	const int size = width * height;

	PPMImage* restrict result = (PPMImage* restrict)aligned_alloc(CACHELINESIZE, sizeof(PPMImage*));
	result->x = width;
	result->y = height;
	result->data = result_data;

	blurIterationHorizontalFirst(image->data,  scratch,  2, width, height);
	blurIterationHorizontal( scratch,  one,  2, width, height);
	blurIterationHorizontalTranspose( one,  scratch,  2, width, height);
	blurIterationVertical( scratch,  one,  2, width, height);

	blurIterationHorizontalFirst(image->data,  scratch,  3, width, height);
	blurIterationHorizontal( scratch,  two,  3, width, height);
	blurIterationHorizontalTranspose( two,  scratch,  3, width, height);
	blurIterationVertical( scratch,  two,  3, width, height);

	imageDifference(result_data,  one,  two, width, height);
	(argc > 1) ? writePPM("flower_tiny.ppm", result) : writeStreamPPM(stdout, result);

	blurIterationHorizontalFirst(image->data,  scratch,  5, width, height);
	blurIterationHorizontal( scratch,  one,  5, width, height);
	blurIterationHorizontalTranspose( one,  scratch,  5, width, height);
	blurIterationVertical( scratch,  one,  5, width, height);
	imageDifference(result_data,  two,  one, width, height);
	(argc > 1) ? writePPM("flower_small.ppm", result) : writeStreamPPM(stdout, result);

	blurIterationHorizontalFirst(image->data,  scratch,  8, width, height);
	blurIterationHorizontal( scratch,  two,  8, width, height);
	blurIterationHorizontalTranspose( two,  scratch,  8, width, height);
	blurIterationVertical( scratch,  two,  8, width, height);
	imageDifference(result_data,  one,  two, width, height);
	(argc > 1) ? writePPM("flower_medium.ppm", result) : writeStreamPPM(stdout, result);
}

