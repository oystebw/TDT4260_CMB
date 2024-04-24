#pragma GCC optimize ("Ofast")
#pragma GCC tune ("cortex-a15")
#pragma GCC tune ("mfpu=neon-vfpv4")
__attribute__((optimize("prefetch-loop-arrays")))

#define _GNU_SOURCE

#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <omp.h>
#include "ppm.h"

#define BLOCKSIZE 8
#define CACHELINESIZE 64
#define PF_OFFSET 128

typedef float v4Accurate __attribute__((vector_size(16)));

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

__attribute__((hot)) void blurIterationHorizontalFirst(const PPMPixel* restrict in, v4Accurate* restrict out, const int size, const int width, const int height) {
	register const v4Accurate multiplier = (v4Accurate){(2 * size + 1), (2 * size + 1), (2 * size + 1), 1.0f};
	#pragma omp parallel for simd schedule(dynamic, 2) num_threads(8)
	for(int y = 0; y < height; ++y) {

		register const int yWidth = y * width;

		register v4Accurate sum = {0.0f, 0.0f, 0.0f, 0.0f};

		for(int x = 0; x <= size; ++x) {
			sum += (v4Accurate){in[yWidth + x].red, in[yWidth + x].green, in[yWidth + x].blue, 0.0f};
		}

		out[yWidth + 0] = sum * multiplier / (v4Accurate){size + 1, size + 1, size + 1, 1.0f};

		for(int x = 1; x <= size; ++x) {
			sum += (v4Accurate){in[yWidth + x + size].red, in[yWidth + x + size].green, in[yWidth + x + size].blue, 0.0f};
			out[yWidth + x] = sum * multiplier / (v4Accurate){size + x + 1, size + x + 1, size + x + 1, 1.0f};
		}


		register v4Accurate* restrict res = &out[yWidth + size + 1];
		register const PPMPixel* restrict minus = &in[yWidth];
		register const PPMPixel* restrict plus = &in[yWidth + size * 2 + 1];
		#pragma GCC unroll 16
		for(int x = width - 2 * size - 1; x > 0; --x) {
			__builtin_prefetch((void*)plus + PF_OFFSET, 0, 3);
			sum -= (v4Accurate){minus->red, minus->green, minus->blue, 0.0f};
			sum += (v4Accurate){plus->red, plus->green, plus->blue, 0.0f};
			*res++ = sum;
			++minus;
			++plus;
		}

		for(int x = width - size; x < width; ++x) {
			sum -= (v4Accurate){in[yWidth + x - size - 1].red, in[yWidth + x - size - 1].green, in[yWidth + x - size - 1].blue, 0.0};
			out[yWidth + x] = sum * multiplier / (v4Accurate){size + width - x, size + width - x, size + width - x, 1.0f};
		}
	}
}

__attribute__((hot)) void blurIterationHorizontal(v4Accurate* restrict in, v4Accurate* restrict out, const int size, const int width, const int height) {
	register const v4Accurate multiplier = (v4Accurate){(2 * size + 1), (2 * size + 1), (2 * size + 1), 1.0f};
	#pragma omp parallel for simd schedule(dynamic, 2) num_threads(8)
	for(int y = 0; y < height; ++y) {
		register const int yWidth = y * width;

		for(int iteration = 0; iteration < 3; ++iteration) {
			
			register v4Accurate sum = {0.0, 0.0, 0.0, 0.0};

			for(int x = 0; x <= size; ++x) {
				sum += in[yWidth + x];
			}

			out[yWidth + 0] = sum * multiplier / (v4Accurate){size + 1, size + 1, size + 1, 1.0f};

			for(int x = 1; x <= size; ++x) {
				sum += in[yWidth + x + size];
				out[yWidth + x] = sum * multiplier / (v4Accurate){size + x + 1, size + x + 1, size + x + 1, 1.0f};
			}

			register v4Accurate* restrict res = &out[yWidth + size + 1];
			register const v4Accurate* restrict minus = &in[yWidth];
			register const v4Accurate* restrict plus = &in[yWidth + size * 2 + 1];
			#pragma GCC unroll 16
			for(int x = width - 2 * size - 1; x > 0; --x) {
				__builtin_prefetch((void*)plus + PF_OFFSET, 0, 3);
				*res++ = sum += *plus++ - *minus++;
			}

			for(int x = width - size; x < width; ++x) {
				sum -= in[yWidth + x - size - 1];
				out[yWidth + x] = sum * multiplier / (v4Accurate){size + width - x, size + width - x, size + width - x, 1.0f};
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

__attribute__((hot)) void blurIterationHorizontalTranspose(const v4Accurate* restrict in, v4Accurate* restrict out, const int size, const int width, const int height) {
	register const v4Accurate multiplier = (v4Accurate){(2 * size + 1), (2 * size + 1), (2 * size + 1), 1.0f};
	#pragma omp parallel for simd schedule(dynamic, 2) num_threads(8)
	for(int y = 0; y < height; ++y) {
		register const int yWidth = y * width;

		register v4Accurate sum = {0.0f, 0.0f, 0.0f, 0.0f};

		for(int x = 0; x <= size; ++x) {
			sum += in[yWidth + x];
		}

		out[0 * height + y] = sum * multiplier / (v4Accurate){size + 1, size + 1, size + 1, 1.0f};

		for(int x = 1; x <= size; ++x) {
			sum += in[yWidth + x + size];
			out[x * height + y] = sum * multiplier / (v4Accurate){size + x + 1, size + x + 1, size + x + 1, 1.0f};
		}

		register v4Accurate* restrict res = &out[(size + 1) * height + y];
		register const v4Accurate* restrict minus = &in[yWidth];
		register const v4Accurate* restrict plus = &in[yWidth + size * 2 + 1];
		#pragma GCC unroll 16
		for(int x = width - 2 * size - 1; x > 0; --x) {
			__builtin_prefetch((void*)plus + PF_OFFSET, 0, 3);		
			*res = sum += *plus++ - *minus++;
			res += height;
		}

		for(int x = width - size; x < width; ++x) {
			sum -= in[yWidth + x - size - 1];
			out[x * height + y] = sum * multiplier / (v4Accurate){size + width - x, size + width - x, size + width - x, 1.0f};
		}
	}
}

__attribute__((hot)) void blurIterationVertical(v4Accurate* restrict in, v4Accurate* restrict out, const int size, const int width, const int height) {
	register const v4Accurate multiplier = (v4Accurate){(2 * size + 1), (2 * size + 1), (2 * size + 1), 1.0f};
	#pragma omp parallel for simd schedule(dynamic, 2) num_threads(8)
	for(int x = 0; x < width; ++x) {
		register const int xHeight = x * height;

		for(int iteration = 0; iteration < 5; ++iteration) {
			
			register v4Accurate sum = {0.0, 0.0, 0.0, 0.0};

			for(int y = 0; y <= size; ++y) {
				sum += in[xHeight + y];
			}

			out[xHeight + 0] = sum * multiplier / (v4Accurate){size + 1, size + 1, size + 1, 1.0f};

			for(int y = 1; y <= size; ++y) {
				sum += in[xHeight + y + size];
				out[xHeight + y] = sum * multiplier / (v4Accurate){y + size + 1, y + size + 1, y + size + 1, 1.0f};
			}

			register v4Accurate* restrict res = &out[xHeight + size + 1];
			register const v4Accurate* restrict minus = &in[xHeight];
			register const v4Accurate* restrict plus = &in[xHeight + size * 2 + 1];
			#pragma GCC unroll 16
			for(int x = height - 2 * size - 1; x > 0; --x) {
				__builtin_prefetch((void*)plus + PF_OFFSET, 0, 3);
				*res++ = sum += *plus++ - *minus++;
			}

			for(int y = height - size; y < height; ++y) {
				sum -= in[xHeight + y - size - 1];
				out[xHeight + y] = sum * multiplier / (v4Accurate){size + height - y, size + height - y, size + height - y, 1.0f};
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

__attribute__((hot)) void imageDifference(PPMPixel* restrict imageOut, const v4Accurate* restrict small, const v4Accurate* restrict large, const int width, const int height, const float sizeSmall, const float sizeLarge) {
	register const v4Accurate divisorSmall = (v4Accurate){1.0f / pow((2.0f * sizeSmall + 1.0f), 10), 1.0f / pow((2.0f * sizeSmall + 1.0f), 10), 1.0f / pow((2.0f * sizeSmall + 1.0f), 10), 1.0f};
	register const v4Accurate divisorLarge = (v4Accurate){1.0f / pow((2.0f * sizeLarge + 1.0f), 10), 1.0f / pow((2.0f * sizeLarge + 1.0f), 10), 1.0f / pow((2.0f * sizeLarge + 1.0f), 10), 1.0f};
	#pragma omp parallel for simd schedule(dynamic, 2) num_threads(8)
	for(int yy = 0; yy < height; yy += BLOCKSIZE) {
		for(int xx = 0; xx < width; xx += BLOCKSIZE) {
			for(int x = xx; x < xx + BLOCKSIZE; ++x) {
				register const int xHeight = x * height;
				// first four elements in tight loop
				__builtin_prefetch(&large[xHeight + height * 2 + yy], 0, 3);
				__builtin_prefetch(&small[xHeight + height * 2 + yy], 0, 3);
				// last four elements in tight loop
				__builtin_prefetch(&large[xHeight + height * 2 + yy + 4], 0, 3);
				__builtin_prefetch(&small[xHeight + height * 2 + yy + 4], 0, 3);
				#pragma GGC unroll 8
				for(int y = yy; y < yy + BLOCKSIZE; ++y) {
					register const v4Accurate diff = large[xHeight + y] * divisorLarge - small[xHeight + y] * divisorSmall;
					imageOut[y * width + x] = (PPMPixel){
						diff[0] < 0.0 ? diff[0] + 257.0 : diff[0],
						diff[1] < 0.0 ? diff[1] + 257.0 : diff[1],
						diff[2] < 0.0 ? diff[2] + 257.0 : diff[2]
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

	v4Accurate* restrict scratch = (v4Accurate* restrict)aligned_alloc(CACHELINESIZE, sizeof(v4Accurate) * width * height);
	v4Accurate* restrict one = (v4Accurate* restrict)aligned_alloc(CACHELINESIZE, sizeof(v4Accurate) * width * height);
	v4Accurate* restrict two = (v4Accurate* restrict)aligned_alloc(CACHELINESIZE, sizeof(v4Accurate) * width * height);
	PPMImage* restrict result = (PPMImage* restrict)malloc(sizeof(PPMImage*));
	result->x = width;
	result->y = height;
	result->data = (PPMPixel* restrict)aligned_alloc(CACHELINESIZE, sizeof(PPMPixel) * width * height);

	blurIterationHorizontalFirst(image->data,  scratch,  2, width, height);
	blurIterationHorizontal( scratch,  one,  2, width, height);
	blurIterationHorizontalTranspose( one,  scratch,  2, width, height);
	blurIterationVertical( scratch,  one,  2, width, height);

	blurIterationHorizontalFirst(image->data,  scratch,  3, width, height);
	blurIterationHorizontal( scratch,  two,  3, width, height);
	blurIterationHorizontalTranspose( two,  scratch,  3, width, height);
	blurIterationVertical( scratch,  two,  3, width, height);

	imageDifference(result->data,  one,  two, width, height, 2.0f, 3.0f);
	(argc > 1) ? writePPM("flower_tiny.ppm", result) : writeStreamPPM(stdout, result);

	blurIterationHorizontalFirst(image->data,  scratch,  5, width, height);
	blurIterationHorizontal( scratch,  one,  5, width, height);
	blurIterationHorizontalTranspose( one,  scratch,  5, width, height);
	blurIterationVertical( scratch,  one,  5, width, height);

	imageDifference(result->data,  two,  one, width, height, 3.0f, 5.0f);
	(argc > 1) ? writePPM("flower_small.ppm", result) : writeStreamPPM(stdout, result);

	blurIterationHorizontalFirst(image->data,  scratch,  8, width, height);
	blurIterationHorizontal( scratch,  two,  8, width, height);
	blurIterationHorizontalTranspose( two,  scratch,  8, width, height);
	blurIterationVertical( scratch,  two,  8, width, height);

	imageDifference(result->data,  one,  two, width, height, 5.0f, 8.0f);
	(argc > 1) ? writePPM("flower_medium.ppm", result) : writeStreamPPM(stdout, result);
}

