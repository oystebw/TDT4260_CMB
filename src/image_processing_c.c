#pragma GCC optimize ("Ofast")
#pragma GCC tune ("cortex-a15")
__attribute__((optimize("prefetch-loop-arrays")))

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

#include "ppm.h"

#define BLOCKSIZE 8
#define CACHELINESIZE 64
#define PF_OFFSET 8

typedef float v4Accurate __attribute__((vector_size(16)));

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

__attribute__((hot)) void blurIterationHorizontal(const PPMPixel* restrict ppm, v4Accurate* restrict in, v4Accurate* restrict out, const int size, const int width, const int height) {
	
	const float sizef = (float)size;
	const v4Accurate multiplier = (v4Accurate){(2.0f * sizef + 1.0f), (2.0f * sizef + 1.0f), (2.0f * sizef + 1.0f), 1.0f};
	
	#pragma omp parallel for simd schedule(dynamic, 2) num_threads(8)
	for(int y = 0; y < height; ++y) {
		register const int yWidth = y * width;
			
		register v4Accurate sum1 = {0.0f, 0.0f, 0.0f, 0.0f};
		register v4Accurate sum2 = {0.0f, 0.0f, 0.0f, 0.0f};
		register v4Accurate sum3 = {0.0f, 0.0f, 0.0f, 0.0f};
		register v4Accurate sum4 = {0.0f, 0.0f, 0.0f, 0.0f};

		// setup first iteration
		for(int x1 = 0; x1 <= size; ++x1) {
			sum1 += (v4Accurate){ppm[yWidth + x1].red, ppm[yWidth + x1].green, ppm[yWidth + x1].blue, 0.0f};
		}
		out[yWidth + 0] = sum1 * multiplier / (v4Accurate){sizef + 1.0f, sizef + 1.0f, sizef + 1.0f, 1.0f};
		for(int x1 = 1; x1 < size + 1; ++x1) {
			sum1 += (v4Accurate){ppm[yWidth + x1 + size].red, ppm[yWidth + x1 + size].green, ppm[yWidth + x1 + size].blue, 0.0f};
			out[yWidth + x1] = sum1 * multiplier / (v4Accurate){sizef + x1 + 1.0f, sizef + x1 + 1.0f, sizef + x1 + 1.0f, 1.0f};
		}
		for(int x1 = size + 1; x1 < 4 * size + 4; ++x1){
			sum1 -= (v4Accurate){ppm[yWidth + x1 - size - 1].red, ppm[yWidth + x1 - size - 1].green, ppm[yWidth + x1 - size - 1].blue, 0.0f};
			sum1 += (v4Accurate){ppm[yWidth + x1 + size].red, ppm[yWidth + x1 + size].green, ppm[yWidth + x1 + size].blue, 0.0f};
			out[yWidth + x1] = sum1;
		}
		// setup first iteration

		// setup second iteration
		for(int x2 = 0; x2 < size + 1; ++x2) {
			sum2 += out[yWidth + x2];
		}
		in[yWidth + 0] = sum2 * multiplier / (v4Accurate){sizef + 1, sizef + 1, sizef + 1, 1.0f};
		for(int x2 = 1; x2 < size + 1; ++x2) {
			sum2 += out[yWidth + x2 + size];
			in[yWidth + x2] = sum2 * multiplier / (v4Accurate){sizef + x2 + 1, sizef + x2 + 1, sizef + x2 + 1, 1.0f};
		}
		for(int x2 = size + 1; x2 < 3 * size + 3; ++x2){
			in[yWidth + x2] = sum2 += out[yWidth + x2 + size] - out[yWidth + x2 - size - 1];
		}
		// setup second iteration

		// setup third iteration
		for(int x3 = 0; x3 < size + 1; ++x3) {
			sum3 += in[yWidth + x3];
		}
		out[yWidth + 0] = sum3 * multiplier / (v4Accurate){sizef + 1, sizef + 1, sizef + 1, 1.0f};
		for(int x3 = 1; x3 < size + 1; ++x3) {
			sum3 += in[yWidth + x3 + size];
			out[yWidth + x3] = sum3 * multiplier / (v4Accurate){sizef + x3 + 1, sizef + x3 + 1, sizef + x3 + 1, 1.0f};
		}
		for(int x3 = size + 1; x3 < 2 * size + 2; ++x3){
			out[yWidth + x3] = sum3 += in[yWidth + x3 + size] - in[yWidth + x3 - size - 1];
		}
		// setup third iteration

		// setup fourth iteration
		for(int x4 = 0; x4 < size + 1; ++x4) {
			sum4 += out[yWidth + x4];
		}
		in[yWidth + 0] = sum4 * multiplier / (v4Accurate){sizef + 1, sizef + 1, sizef + 1, 1.0f};
		for(int x4 = 1; x4 < size + 1; ++x4) {
			sum4 += out[yWidth + x4 + size];
			in[yWidth + x4] = sum4 * multiplier / (v4Accurate){sizef + x4 + 1, sizef + x4 + 1, sizef + x4 + 1, 1.0f};
		}
		// setup fourth iteration

		#pragma GCC unroll 16
		#pragma GCC ivdep
		for(int x = 4 * size + 4; x < width - size; ++x) {
			__builtin_prefetch(&ppm[yWidth + x + size + 42], 0, 3); // two cachelines ahead
			__builtin_prefetch(&out[yWidth + x - 1 + PF_OFFSET], 0, 3); // two cachelines ahead
			__builtin_prefetch(&in[yWidth + x - size - 2 + PF_OFFSET], 0, 3); // two cachelines ahead
			sum1 -= (v4Accurate){ppm[yWidth + x - size - 1].red, ppm[yWidth + x - size - 1].green, ppm[yWidth + x - size - 1].blue, 0.0f};
			sum1 += (v4Accurate){ppm[yWidth + x + size].red, ppm[yWidth + x + size].green, ppm[yWidth + x + size].blue, 0.0f};
			out[yWidth + x] = sum1;
			in[yWidth + x - size - 1] = sum2 += out[yWidth + x - 1] - out[yWidth + x - 2 * size - 2];
			out[yWidth + x - 2 * size - 2] = sum3 += in[yWidth + x - size - 2] - in[yWidth + x - 3 * size - 3];
			in[yWidth + x - 3 * size - 3] = sum4 += out[yWidth + x - 2 * size - 3] - out[yWidth + x - 4 * size - 4];
		}

		// finishing first iteration
		for(int x1 = width - size; x1 < width; ++x1) {
			sum1 -= (v4Accurate){ppm[yWidth + x1 - size - 1].red, ppm[yWidth + x1 - size - 1].green, ppm[yWidth + x1 - size - 1].blue, 0.0f};
			out[yWidth + x1] = sum1 * multiplier / (v4Accurate){sizef + width - x1, sizef + width - x1, sizef + width - x1, 1.0f};
		}
		// finishing first iteration

		// finishing second iteration
		for(int x2 = width - 2 * size - 1; x2 < width - size; ++x2) {
			in[yWidth + x2] = sum2 += out[yWidth + x2 + size] -= out[yWidth + x2 - size - 1];
		}
		for(int x2 = width - size; x2 < width; ++x2) {
			sum2 -= out[yWidth + x2 - size - 1];
			in[yWidth + x2] = sum2 * multiplier / (v4Accurate){sizef + width - x2, sizef + width - x2, sizef + width - x2, 1.0f};
		}
		// finishing second iteration

		// finishing third iteration
		for(int x3 = width - 3 * size - 2; x3 < width - size; ++x3) {
			out[yWidth + x3] = sum3 += in[yWidth + x3 + size] - in[yWidth + x3 - size - 1];
		}
		for(int x3 = width - size; x3 < width; ++x3) {
			sum3 -= in[yWidth + x3 - size - 1];
			out[yWidth + x3] = sum3 * multiplier / (v4Accurate){sizef + width - x3, sizef + width - x3, sizef + width - x3, 1.0f};
		}
		// finishing third iteration

		// finishing fourth iteration
		for(int x4 = width - 4 * size - 3; x4 < width - size; ++x4) {
			in[yWidth + x4] = sum4 += out[yWidth + x4 + size] - out[yWidth + x4 - size - 1];
		}
		for(int x4 = width - size; x4 < width; ++x4) {
			sum4 -= out[yWidth + x4 - size - 1];
			in[yWidth + x4] = sum4 * multiplier / (v4Accurate){sizef + width - x4, sizef + width - x4, sizef + width - x4, 1.0f};
		}
		// finishing fourth iteration
	}
}

/*
We "cheat" by transposing directly during the fifth horizontal iteration. This is much faster than doing it as a separate routine.
In addition, we do the non cache-friendly operation as a store, which is better than doing it as a load.
*/
__attribute__((hot)) void blurIterationHorizontalTranspose(const v4Accurate* restrict in, v4Accurate* restrict out, const int size, const int width, const int height) {
	
	const float sizef = (float)size;
	const v4Accurate multiplier = (v4Accurate){(2.0f * sizef + 1.0f), (2.0f * sizef + 1.0f), (2.0f * sizef + 1.0f), 1.0f};
	
	#pragma omp parallel for simd schedule(dynamic, 2) num_threads(8)
	for(int y = 0; y < height; ++y) {
		register const int yWidth = y * width;

		register v4Accurate sum = {0.0f, 0.0f, 0.0f, 0.0f};

		for(int x = 0; x < size + 1; ++x) {
			sum += in[yWidth + x];
		}

		out[0 * height + y] = sum * multiplier / (v4Accurate){sizef + 1.0f, sizef + 1.0f, sizef + 1.0f, 1.0f};

		for(int x = 1; x < size + 1; ++x) {
			sum += in[yWidth + x + size];
			out[x * height + y] = sum * multiplier / (v4Accurate){sizef + x + 1.0f, sizef + x + 1.0f, sizef + x + 1.0f, 1.0f};
		}

		#pragma GCC unroll 16
		#pragma GCC ivdep
		for(int x = size + 1; x < width - size; ++x) {
			__builtin_prefetch(&in[yWidth + x + size + PF_OFFSET], 0, 3); // two cachelines ahead
			out[x * height + y] = sum += in[yWidth + x + size] - in[yWidth + x - size - 1];
		}

		for(int x = width - size; x < width; ++x) {
			sum -= in[yWidth + x - size - 1];
			out[x * height + y] = sum * multiplier / (v4Accurate){sizef + width - x, sizef + width - x, sizef + width - x, 1.0f};
		}
	}
}

__attribute__((hot)) void blurIterationVertical(v4Accurate* restrict in, v4Accurate* restrict out, const int size, const int width, const int height) {
	
	const float sizef = (float)size;
	const v4Accurate multiplier = (v4Accurate){(2.0f * sizef + 1.0f), (2.0f * sizef + 1.0f), (2.0f * sizef + 1.0f), 1.0f};
	const v4Accurate divisor9 = (v4Accurate){1.0 / pow((2.0f * sizef + 1.0f), 9), 1.0 / pow((2.0f * sizef + 1.0f), 9), 1.0 / pow((2.0f * sizef + 1.0f), 9), 1.0f};
	register const v4Accurate divisor10 = (v4Accurate){1.0 / pow((2.0f * sizef + 1.0f), 10), 1.0 / pow((2.0f * sizef + 1.0f), 10), 1.0 / pow((2.0f * sizef + 1.0f), 10), 1.0f};
	
	#pragma omp parallel for simd schedule(dynamic, 2) num_threads(8)
	for(int x = 0; x < width; ++x) {
		register const int xHeight = x * height;
			
		register v4Accurate sum1 = {0.0f, 0.0f, 0.0f, 0.0f};
		register v4Accurate sum2 = {0.0f, 0.0f, 0.0f, 0.0f};
		register v4Accurate sum3 = {0.0f, 0.0f, 0.0f, 0.0f};
		register v4Accurate sum4 = {0.0f, 0.0f, 0.0f, 0.0f};
		register v4Accurate sum5 = {0.0f, 0.0f, 0.0f, 0.0f};

		// setup first iteration
		for(int y1 = 0; y1 < size + 1; ++y1) {
			sum1 += in[xHeight + y1];
		}
		out[xHeight + 0] = sum1 * multiplier / (v4Accurate){sizef + 1, sizef + 1, sizef + 1, 1.0f};
		for(int y1 = 1; y1 < size + 1; ++y1) {
			sum1 += in[xHeight + y1 + size];
			out[xHeight + y1] = sum1 * multiplier / (v4Accurate){sizef + y1 + 1, sizef + y1 + 1, sizef + y1 + 1, 1.0f};
		}
		for(int y1 = size + 1; y1 < 5 * size + 5; ++y1) {
			out[xHeight + y1] = sum1 += in[xHeight + y1 + size] - in[xHeight + y1 - size - 1];
		}
		// setup first iteration

		// setup second iteration
		for(int y2 = 0; y2 < size + 1; ++y2) {
			sum2 += out[xHeight + y2];
		}
		in[xHeight + 0] = sum2 * multiplier / (v4Accurate){sizef + 1, sizef + 1, sizef + 1, 1.0f};
		for(int y2 = 1; y2 < size + 1; ++y2) {
			sum2 += out[xHeight + y2 + size];
			in[xHeight + y2] = sum2 * multiplier / (v4Accurate){sizef + y2 + 1, sizef + y2 + 1, sizef + y2 + 1, 1.0f};
		}
		for(int y2 = size + 1; y2 < 4 * size + 4; ++y2) {
			in[xHeight + y2] = sum2 += out[xHeight + y2 + size] - out[xHeight + y2 - size - 1];
		}
		// setup second iteration

		// setup third iteration
		for(int y3 = 0; y3 < size + 1; ++y3) {
			sum3 += in[xHeight + y3];
		}
		out[xHeight + 0] = sum3 * multiplier / (v4Accurate){sizef + 1, sizef + 1, sizef + 1, 1.0f};
		for(int y3 = 1; y3 < size + 1; ++y3) {
			sum3 += in[xHeight + y3 + size];
			out[xHeight + y3] = sum3 * multiplier / (v4Accurate){sizef + y3 + 1, sizef + y3 + 1, sizef + y3 + 1, 1.0f};
		}
		for(int y3 = size + 1; y3 < 3 * size + 3; ++y3) {
			out[xHeight + y3] = sum3 += in[xHeight + y3 + size] - in[xHeight + y3 - size - 1];
		}
		// setup third iteration

		// setup fourth iteration
		for(int y4 = 0; y4 < size + 1; ++y4) {
			sum4 += out[xHeight + y4];
		}
		in[xHeight + 0] = sum4 * multiplier / (v4Accurate){sizef + 1, sizef + 1, sizef + 1, 1.0f};
		for(int y4 = 1; y4 < size + 1; ++y4) {
			sum4 += out[xHeight + y4 + size];
			in[xHeight + y4] = sum4 * multiplier / (v4Accurate){sizef + y4 + 1, sizef + y4 + 1, sizef + y4 + 1, 1.0f};
		}
		for(int y4 = size + 1; y4 < 2 * size + 2; ++y4) {
			in[xHeight + y4] = sum4 += out[xHeight + y4 + size] - out[xHeight + y4 - size - 1];
		}
		// setup fourth iteration

		// setup fifth iteration
		for(int y5 = 0; y5 < size + 1; ++y5) {
			sum5 += in[xHeight + y5];
		}
		out[xHeight + 0] = sum5 * divisor9 / (v4Accurate){sizef + 1, sizef + 1, sizef + 1, 1.0f};
		for(int y5 = 1; y5 < size + 1; ++y5) {
			sum5 += in[xHeight + y5 + size];
			out[xHeight + y5] = sum5 * divisor9 / (v4Accurate){sizef + y5 + 1, sizef + y5 + 1, sizef + y5 + 1, 1.0f};
		}
		// setup last iteration

		#pragma GCC unroll 16
		#pragma GCC ivdep
		for(int y = 5 * size + 5; y < height - size; ++y) {
			__builtin_prefetch(&in[xHeight + y + size + PF_OFFSET], 0, 3); // two cachelines ahead
			__builtin_prefetch(&out[xHeight + y - 1 + PF_OFFSET], 0, 3);
			out[xHeight + y] = sum1 += in[xHeight + y + size] - in[xHeight + y - size - 1];
			in[xHeight + y - size - 1] = sum2 += out[xHeight + y - 1] - out[xHeight + y - 2 * size - 2];
			out[xHeight + y - 2 * size - 2] = sum3 += in[xHeight + y - size - 2] - in[xHeight + y - 3 * size - 3];
			in[xHeight + y - 3 * size - 3] = sum4 += out[xHeight + y - 2 * size - 3] - out[xHeight + y - 4 * size - 4];
			sum5 += in[xHeight + y - 3 * size - 4] - in[xHeight + y - 5 * size - 5];
			out[xHeight + y - 4 * size - 4] = sum5 * divisor10;
		}

		// finishing first iteration
		for(int y1 = height - size; y1 < height; ++y1) {
			sum1 -= in[xHeight + y1 - size - 1];
			out[xHeight + y1] = sum1 * multiplier / (v4Accurate){sizef + height - y1, sizef + height - y1, sizef + height - y1, 1.0f};
		}
		// finishing first iteration

		// finishing second iteration
		for(int y2 = height - 2 * size - 1; y2 < height - size; ++y2) {
			in[xHeight + y2] = sum2 += out[xHeight + y2 + size] -= out[xHeight + y2 - size - 1];
		}
		for(int y2 = height - size; y2 < height; ++y2) {
			sum2 -= out[xHeight + y2 - size - 1];
			in[xHeight + y2] = sum2 * multiplier / (v4Accurate){sizef + height - y2, sizef + height - y2, sizef + height - y2, 1.0f};
		}
		// finishing second iteration

		// finishing third iteration
		for(int y3 = height - 3 * size - 2; y3 < height - size; ++y3) {
			out[xHeight + y3] = sum3 += in[xHeight + y3 + size] - in[xHeight + y3 - size - 1];
		}
		for(int y3 = height - size; y3 < height; ++y3) {
			sum3 -= in[xHeight + y3 - size - 1];
			out[xHeight + y3] = sum3 * multiplier / (v4Accurate){sizef + height - y3, sizef + height - y3, sizef + height - y3, 1.0f};
		}
		// finishing third iteration

		// finishing fourth iteration
		for(int y4 = height - 4 * size - 3; y4 < height - size; ++y4) {
			in[xHeight + y4] = sum4 += out[xHeight + y4 + size] - out[xHeight + y4 - size - 1];
		}
		for(int y4 = height - size; y4 < height; ++y4) {
			sum4 -= out[xHeight + y4 - size - 1];
			in[xHeight + y4] = sum4 * multiplier / (v4Accurate){sizef + height - y4, sizef + height - y4, sizef + height - y4, 1.0f};
		}
		// finishing fourth iteration

		// finishing fifth iteration
		for(int y5 = height - 5 * size - 4; y5 < height - size; ++y5) {
			sum5 += in[xHeight + y5 + size] - in[xHeight + y5 - size - 1];
			out[xHeight + y5] = sum5 * divisor10;
		}
		for(int y5 = height - size; y5 < height; ++y5) {
			sum5 -= in[xHeight + y5 - size - 1];
			out[xHeight + y5] = sum5 * divisor9 / (v4Accurate){sizef + height - y5, sizef + height - y5, sizef + height - y5, 1.0f};
		}
		// finishing fifth iteration
	}
}

/*
This function serves three purposes:
1. Divides the pixels by (2 * kernelSize + 1)^10 to normalize.
2. Computes the difference between two Gaussian blurred images.
3. Stores the resulting image AND transposes it. Notice that we transpose twice in this program, and both are "for free".
*/
__attribute__((hot)) void imageDifference(PPMPixel* restrict imageOut, const v4Accurate* restrict small, const v4Accurate* restrict large, const int width, const int height, const float sizeSmall, const float sizeLarge) {
	// do all 10 divisions at the end of the pipeline
	// register const v4Accurate divisorSmall = (v4Accurate){1.0f / pow((2.0f * sizeSmall + 1.0f), 10), 1.0f / pow((2.0f * sizeSmall + 1.0f), 10), 1.0f / pow((2.0f * sizeSmall + 1.0f), 10), 1.0f};
	// register const v4Accurate divisorLarge = (v4Accurate){1.0f / pow((2.0f * sizeLarge + 1.0f), 10), 1.0f / pow((2.0f * sizeLarge + 1.0f), 10), 1.0f / pow((2.0f * sizeLarge + 1.0f), 10), 1.0f};
	
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
				#pragma GCC ivdep
				for(int y = yy; y < yy + BLOCKSIZE; ++y) {
					register const v4Accurate diff = large[xHeight + y] - small[xHeight + y];
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

/*
Just to clean up and explicitly show what happens to the image.
*/
void blurImage(const PPMPixel* restrict imageIn, v4Accurate* restrict scratch, v4Accurate* restrict result, const int kernelSize, const int width, const int height) {
	blurIterationHorizontal(imageIn, result, scratch, kernelSize, width, height);
	blurIterationHorizontalTranspose(result, scratch, kernelSize, width, height);
	blurIterationVertical(scratch, result, kernelSize, width, height);
}

int main(int argc, char** argv) {

    const PPMImage* restrict image = (argc > 1) ? readPPM("flower.ppm") : readStreamPPM(stdin);

	const int width = image->x;
	const int height = image->y;
	const int size = width * height;

	/*
	We only need to allocate memory for three images. This reduces runtime since the allocation takes some time.
	In addition, there are fewer page misses and page walks when we only work within three images and not eight.
	
	aligned_alloc made an improvement due to the alignment with cachelines.
	*/
	v4Accurate* restrict scratch = (v4Accurate* restrict)aligned_alloc(CACHELINESIZE, sizeof(v4Accurate) * width * height);
	v4Accurate* restrict one = (v4Accurate* restrict)aligned_alloc(CACHELINESIZE, sizeof(v4Accurate) * width * height);
	v4Accurate* restrict two = (v4Accurate* restrict)aligned_alloc(CACHELINESIZE, sizeof(v4Accurate) * width * height);

	PPMImage* restrict result = (PPMImage* restrict)malloc(sizeof(PPMImage*));
	result->x = width;
	result->y = height;
	result->data = (PPMPixel* restrict)aligned_alloc(CACHELINESIZE, sizeof(PPMPixel) * width * height);

	// tiny image
	blurImage(image->data, scratch, one, 2, width, height);

	// small image
	blurImage(image->data, scratch, two, 3, width, height);

	// tinyPPM
	imageDifference(result->data,  one,  two, width, height, 2.0f, 3.0f);
	(argc > 1) ? writePPM("flower_tiny.ppm", result) : writeStreamPPM(stdout, result);

	// medium image
	blurImage(image->data, scratch, one, 5, width, height);

	// smallPPM
	imageDifference(result->data,  two,  one, width, height, 3.0f, 5.0f);
	(argc > 1) ? writePPM("flower_small.ppm", result) : writeStreamPPM(stdout, result);

	// large image
	blurImage(image->data, scratch, two, 8, width, height);

	// mediumPPM
	imageDifference(result->data,  one,  two, width, height, 5.0f, 8.0f);
	(argc > 1) ? writePPM("flower_medium.ppm", result) : writeStreamPPM(stdout, result);
}

