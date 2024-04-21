#pragma GCC optimize ("Ofast")
#pragma GCC tune ("cortex-a15")
__attribute__((optimize("prefetch-loop-arrays")))

#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <omp.h>
#include "ppm.h"

#define BLOCKSIZE 8
#define PF_OFFSET 16

typedef float v4Accurate __attribute__((vector_size(16)));
typedef __uint32_t v4Int __attribute__((vector_size(16)));

PPMPixel result_data[1920 * 1200];
v4Accurate accurate[1920 * 1200 * 3];

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

void blurIterationHorizontalFirst(const PPMPixel* restrict in, v4Accurate* restrict out, const int size, const int width, const int height) {
	const v4Accurate divisor = (v4Accurate){1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0f};
	#pragma omp parallel for schedule(dynamic, 2) num_threads(8)
	for(int y = 0; y < height; ++y) {
		const int yWidth = y * width;

		v4Int sum = {0, 0, 0, 0};

		for(int x = 0; x <= size; ++x) {
			sum += (v4Int){in[yWidth + x].red, in[yWidth + x].green, in[yWidth + x].blue, 0.0f};
		}

		out[(yWidth + 0) * 3] = (v4Accurate){sum[0], sum[1], sum[2], sum[3]} / (v4Accurate){size + 1, size + 1, size + 1, 1.0f};

		for(int x = 1; x <= size; ++x) {
			sum += (v4Int){in[yWidth + x + size].red, in[yWidth + x + size].green, in[yWidth + x + size].blue, 0.0};
			out[(yWidth + x) * 3] = (v4Accurate){sum[0], sum[1], sum[2], sum[3]} / (v4Accurate){size + x + 1, size + x + 1, size + x + 1, 1.0f};
		}

		#pragma GCC unroll 16
		for(int x = size + 1; x < width - size; ++x) {
			sum -= (v4Int){in[yWidth + x - size - 1].red, in[yWidth + x - size - 1].green, in[yWidth + x - size - 1].blue, 0.0};
			sum += (v4Int){in[yWidth + x + size].red, in[yWidth + x + size].green, in[yWidth + x + size].blue, 0.0};
			out[(yWidth + x) * 3] = (v4Accurate){sum[0], sum[1], sum[2], sum[3]} * divisor;
		}

		for(int x = width - size; x < width; ++x) {
			sum -= (v4Int){in[yWidth + x - size - 1].red, in[yWidth + x - size - 1].green, in[yWidth + x - size - 1].blue, 0.0};
			out[(yWidth + x) * 3] = (v4Accurate){sum[0], sum[1], sum[2], sum[3]} / (v4Accurate){size + width - x, size + width - x, size + width - x, 1.0f};
		}
	}
}

void blurIterationHorizontal(v4Accurate* in, v4Accurate* out, const int size, const int width, const int height) {
	const v4Accurate divisor = (v4Accurate){1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0f};
	#pragma omp parallel for schedule(dynamic, 2) num_threads(8)
	for(int y = 0; y < height; ++y) {
		const int yWidth = y * width;
		for(int iteration = 0; iteration < 3; ++iteration) {
			
			v4Accurate sum = {0.0, 0.0, 0.0, 0.0};

			for(int x = 0; x <= size; ++x) {
				sum += in[(yWidth + x) * 3];
			}

			out[(yWidth + 0) * 3] = sum / (v4Accurate){size + 1, size + 1, size + 1, 1.0f};

			for(int x = 1; x <= size; ++x) {
				sum += in[(yWidth + x + size) * 3];
				out[(yWidth + x) * 3] = sum / (v4Accurate){size + x + 1, size + x + 1, size + x + 1, 1.0f};
			}

			#pragma GCC unroll 16
			for(int x = size + 1; x < width - size; ++x) {
				sum -= in[(yWidth + x - size - 1) * 3];
				sum += in[(yWidth + x + size) * 3];
				out[(yWidth + x) * 3] = sum * divisor;
			}

			for(int x = width - size; x < width; ++x) {
				sum -= in[(yWidth + x - size - 1) * 3];
				out[(yWidth + x) * 3] = sum / (v4Accurate){size + width - x, size + width - x, size + width - x, 1.0f};
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
	const v4Accurate divisor = (v4Accurate){1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0f};
	#pragma omp parallel for schedule(dynamic, 2) num_threads(8)
	for(int y = 0; y < height; ++y) {
		const int yWidth = y * width;

		v4Accurate sum = {0.0, 0.0, 0.0, 0.0};

		for(int x = 0; x <= size; ++x) {
			sum += in[(yWidth + x) * 3];
		}

		out[(0 * height + y) * 3] = sum / (v4Accurate){size + 1, size + 1, size + 1, 1.0f};

		for(int x = 1; x <= size; ++x) {
			sum += in[(yWidth + x + size) * 3];
			out[(x * height + y) * 3] = sum / (v4Accurate){size + x + 1, size + x + 1, size + x + 1, 1.0f};
		}

		#pragma GCC unroll 16
		for(int x = size + 1; x < width - size; ++x) {			
			sum -= in[(yWidth + x - size - 1) * 3];
			sum += in[(yWidth + x + size) * 3];
			out[(x * height + y) * 3] = sum * divisor;
		}

		for(int x = width - size; x < width; ++x) {
			sum -= in[(yWidth + x - size - 1) * 3];
			out[(x * height + y) * 3] = sum / (v4Accurate){size + width - x, size + width - x, size + width - x, 1.0f};
		}
	}
}

void blurIterationVertical(v4Accurate* in, v4Accurate* out, const int size, const int width, const int height) {
	const v4Accurate divisor = (v4Accurate){1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0f};
	#pragma omp parallel for schedule(dynamic, 2) num_threads(8)
	for(int x = 0; x < width; ++x) {
		const int xHeight = x * height;
		for(int iteration = 0; iteration < 5; ++iteration) {
			v4Accurate sum = {0.0, 0.0, 0.0, 0.0};

			for(int y = 0; y <= size; ++y) {
				sum += in[(xHeight + y) * 3];
			}

			out[(xHeight + 0) * 3] = sum / (v4Accurate){size + 1, size + 1, size + 1, 1.0f};

			for(int y = 1; y <= size; ++y) {
				sum += in[(xHeight + y + size) * 3];
				out[(xHeight + y) * 3] = sum / (v4Accurate){y + size + 1, y + size + 1, y + size + 1, 1.0f};
			}

			#pragma GCC unroll 16
			for(int y = size + 1; y < height - size; ++y) {
				sum -= in[(xHeight + y - size - 1) * 3];
				sum += in[(xHeight + y + size) * 3];
				out[(xHeight + y) * 3] = sum * divisor;
			}

			for(int y = height - size; y < height; ++y) {
				sum -= in[(xHeight + y - size - 1) * 3];
				out[(xHeight + y) * 3] = sum / (v4Accurate){size + height - y, size + height - y, size + height - y, 1.0f};
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
	for(int yy = 0; yy < height; yy += BLOCKSIZE) {
		for(int xx = 0; xx < width; xx += BLOCKSIZE) {
			for(int x = xx; x < xx + BLOCKSIZE; ++x) {
				const int xHeight = x * height;
				__builtin_prefetch((float*)&large[(xHeight + height + yy) * 3], 0, 3);
				__builtin_prefetch((float*)&small[(xHeight + height + yy) * 3], 0, 3);
				#pragma GGC unroll 8
				for(int y = yy; y < yy + BLOCKSIZE; ++y) {
					v4Accurate diff = large[(xHeight + y) * 3] - small[(xHeight + y) * 3];
					imageOut[y * width + x] = (PPMPixel){
						diff[0] = diff[0] < 0.0 ? diff[0] + 257.0 : diff[0],
						diff[1] = diff[1] < 0.0 ? diff[1] + 257.0 : diff[1],
						diff[2] = diff[2] < 0.0 ? diff[2] + 257.0 : diff[2]
					};
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

	PPMImage* restrict result = (PPMImage* restrict)malloc(sizeof(PPMImage*));
	result->x = width;
	result->y = height;
	result->data = result_data;

	blurIterationHorizontalFirst(image->data,  accurate,  2, width, height);
	blurIterationHorizontal( accurate,  accurate + 1,  2, width, height);
	blurIterationHorizontalTranspose( accurate + 1,  accurate,  2, width, height);
	blurIterationVertical( accurate,  accurate + 1,  2, width, height);

	blurIterationHorizontalFirst(image->data,  accurate,  3, width, height);
	blurIterationHorizontal( accurate,  accurate + 2,  3, width, height);
	blurIterationHorizontalTranspose( accurate + 2,  accurate,  3, width, height);
	blurIterationVertical( accurate,  accurate + 2,  3, width, height);

	imageDifference(result_data,  accurate + 1,  accurate + 2, width, height);
	(argc > 1) ? writePPM("flower_tiny.ppm", result) : writeStreamPPM(stdout, result);

	blurIterationHorizontalFirst(image->data,  accurate,  5, width, height);
	blurIterationHorizontal( accurate,  accurate + 1,  5, width, height);
	blurIterationHorizontalTranspose( accurate + 1,  accurate,  5, width, height);
	blurIterationVertical( accurate,  accurate + 1,  5, width, height);
	imageDifference(result_data,  accurate + 2,  accurate + 1, width, height);
	(argc > 1) ? writePPM("flower_small.ppm", result) : writeStreamPPM(stdout, result);

	blurIterationHorizontalFirst(image->data,  accurate,  8, width, height);
	blurIterationHorizontal( accurate,  accurate + 2,  8, width, height);
	blurIterationHorizontalTranspose( accurate + 2,  accurate,  8, width, height);
	blurIterationVertical( accurate,  accurate + 2,  8, width, height);
	imageDifference(result_data,  accurate + 1,  accurate + 2, width, height);
	(argc > 1) ? writePPM("flower_medium.ppm", result) : writeStreamPPM(stdout, result);
}

