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

/*
First horizontal blur. No need to convert PPMImage to AccurateImage before starting the Gaussian blur.
Notice the running sum used. This is super efficient, and makes the run time independent of kernel size.
We only do, on average, two loads, two summation, and one store per pixel, since the three channels are SIMD'ed.
*/
__attribute__((hot)) void blurIterationHorizontalFirst(const PPMPixel* restrict in, v4Accurate* restrict out, const int size, const int width, const int height) {
	
	const float sizef = (float)size;
	/*
	Notice 'out[yWidth + x] = sum;' in the most important loop. We don't divide by the number of elements,
	but rather divide by the number of elements^10 after all blurring (in imageDifference), so we only perform 1/10 of the
	divisions as a naive approach. Notice that store operations outside of this loop are multiplied
	by 'multiplier' and divided by another number. This is because they are summed by fewer numbers,
	so it is done to scale up the edge pixels by the same amount as the rest of the image.
	This makes it possible to divide all pixels by (2 * kernelSize + 1)^10 at the end.
	It is also necessary to compute the correct result.
	*/
	const v4Accurate multiplier = (v4Accurate){(2.0f * sizef + 1.0f), (2.0f * sizef + 1.0f), (2.0f * sizef + 1.0f), 1.0f};
	
	/*
	We schedule dynamic due to the nature of the CPU, which consists of four A15 (performance) cores
	and four A7 (energy) cores. The P cores are much faster, and the dynamic scheduling mitigates
	load unbalancing. Going from static to dynamic improved the runtime by approx. 20-30%
	*/
	#pragma omp parallel for simd schedule(dynamic, 2) num_threads(8)
	for(int y = 0; y < height; ++y) {
		register const int yWidth = y * width;

		register v4Accurate sum = {0.0f, 0.0f, 0.0f, 0.0f};

		for(int x = 0; x <= size; ++x) {
			sum += (v4Accurate){in[yWidth + x].red, in[yWidth + x].green, in[yWidth + x].blue, 0.0f};
		}

		out[yWidth + 0] = sum * multiplier / (v4Accurate){sizef + 1.0f, sizef + 1.0f, sizef + 1.0f, 1.0f};

		for(int x = 1; x <= size; ++x) {
			sum += (v4Accurate){in[yWidth + x + size].red, in[yWidth + x + size].green, in[yWidth + x + size].blue, 0.0f};
			out[yWidth + x] = sum * multiplier / (v4Accurate){sizef + x + 1.0f, sizef + x + 1.0f, sizef + x + 1.0f, 1.0f};
		}

		// this is the 'important loop', and consists of over 99% of the runtime
		#pragma GCC unroll 16
		#pragma GCC ivdep
		for(int x = size + 1; x < width - size; ++x) {
			__builtin_prefetch(&in[yWidth + x + size + 42], 0, 3); // two cachelines ahead
			sum -= (v4Accurate){in[yWidth + x - size - 1].red, in[yWidth + x - size - 1].green, in[yWidth + x - size - 1].blue, 0.0f};
			sum += (v4Accurate){in[yWidth + x + size].red, in[yWidth + x + size].green, in[yWidth + x + size].blue, 0.0f};
			out[yWidth + x] = sum;
		}

		for(int x = width - size; x < width; ++x) {
			sum -= (v4Accurate){in[yWidth + x - size - 1].red, in[yWidth + x - size - 1].green, in[yWidth + x - size - 1].blue, 0.0f};
			out[yWidth + x] = sum * multiplier / (v4Accurate){sizef + width - x, sizef + width - x, sizef + width - x, 1.0f};
		}
	}
}


/*
The next three horizontal blurs. We do each row three times before going one row down, to increase cache locality.
In addition, this makes it possible to have the #pragma omp in the outer loop, since each row is independent.
This reduces multiprocessing overhead.
*/
__attribute__((hot)) void blurIterationHorizontal(v4Accurate* restrict in, v4Accurate* restrict out, const int size, const int width, const int height) {
	
	const float sizef = (float)size;
	const v4Accurate multiplier = (v4Accurate){(2.0f * sizef + 1.0f), (2.0f * sizef + 1.0f), (2.0f * sizef + 1.0f), 1.0f};
	
	#pragma omp parallel for simd schedule(dynamic, 2) num_threads(8)
	for(int y = 0; y < height; ++y) {
		register const int yWidth = y * width;

		for(int iteration = 0; iteration < 3; ++iteration) {
			
			register v4Accurate sum = {0.0f, 0.0f, 0.0f, 0.0f};

			for(int x = 0; x <= size; ++x) {
				sum += in[yWidth + x];
			}

			out[yWidth + 0] = sum * multiplier / (v4Accurate){sizef + 1, sizef + 1, sizef + 1, 1.0f};

			for(int x = 1; x <= size; ++x) {
				sum += in[yWidth + x + size];
				out[yWidth + x] = sum * multiplier / (v4Accurate){sizef + x + 1, sizef + x + 1, sizef + x + 1, 1.0f};
			}

			#pragma GCC unroll 16
			#pragma GCC ivdep
			for(int x = size + 1; x < width - size; ++x) {
				__builtin_prefetch(&in[yWidth + x + size + PF_OFFSET], 0, 3); // two cachelines ahead
				out[yWidth + x] = sum += in[yWidth + x + size] - in[yWidth + x - size - 1];
			}

			for(int x = width - size; x < width; ++x) {
				sum -= in[yWidth + x - size - 1];
				out[yWidth + x] = sum * multiplier / (v4Accurate){sizef + width - x, sizef + width - x, sizef + width - x, 1.0f};
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

		for(int x = 0; x <= size; ++x) {
			sum += in[yWidth + x];
		}

		out[0 * height + y] = sum * multiplier / (v4Accurate){sizef + 1.0f, sizef + 1.0f, sizef + 1.0f, 1.0f};

		for(int x = 1; x <= size; ++x) {
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

/*
Five vertical blur iteration with a transposed image. Super cache friendly.
*/
__attribute__((hot)) void blurIterationVertical(v4Accurate* restrict in, v4Accurate* restrict out, const int size, const int width, const int height) {
	
	const float sizef = (float)size;
	const v4Accurate multiplier = (v4Accurate){(2.0f * sizef + 1.0f), (2.0f * sizef + 1.0f), (2.0f * sizef + 1.0f), 1.0f};
	
	#pragma omp parallel for simd schedule(dynamic, 2) num_threads(8)
	for(int x = 0; x < width; ++x) {
		register const int xHeight = x * height;

		for(int iteration = 0; iteration < 5; ++iteration) {
			
			register v4Accurate sum = {0.0f, 0.0f, 0.0f, 0.0f};

			for(int y = 0; y <= size; ++y) {
				sum += in[xHeight + y];
			}

			out[xHeight + 0] = sum * multiplier / (v4Accurate){sizef + 1.0f, sizef + 1.0f, sizef + 1.0f, 1.0f};

			for(int y = 1; y <= size; ++y) {
				sum += in[xHeight + y + size];
				out[xHeight + y] = sum * multiplier / (v4Accurate){y + sizef + 1.0f, y + sizef + 1.0f, y + sizef + 1.0f, 1.0f};
			}

			#pragma GCC unroll 16
			#pragma GCC ivdep
			for(int y = size + 1; y < height - size; ++y) {
				__builtin_prefetch(&in[xHeight + y + size + PF_OFFSET], 0, 3); // two cachelines ahead
				out[xHeight + y] = sum += in[xHeight + y + size] - in[xHeight + y - size - 1];
			}

			for(int y = height - size; y < height; ++y) {
				sum -= in[xHeight + y - size - 1];
				out[xHeight + y] = sum * multiplier / (v4Accurate){sizef + height - y, sizef + height - y, sizef + height - y, 1.0f};
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
	// do all 10 divisions at the end of the pipeline
	register const v4Accurate divisorSmall = (v4Accurate){1.0f / pow((2.0f * sizeSmall + 1.0f), 10), 1.0f / pow((2.0f * sizeSmall + 1.0f), 10), 1.0f / pow((2.0f * sizeSmall + 1.0f), 10), 1.0f};
	register const v4Accurate divisorLarge = (v4Accurate){1.0f / pow((2.0f * sizeLarge + 1.0f), 10), 1.0f / pow((2.0f * sizeLarge + 1.0f), 10), 1.0f / pow((2.0f * sizeLarge + 1.0f), 10), 1.0f};
	
	#pragma omp parallel for simd schedule(dynamic, 2) num_threads(8)
	for(int yy = 0; yy < height; yy += BLOCKSIZE) {
		for(int xx = 0; xx < width; xx += BLOCKSIZE) {
			for(int x = xx; x < xx + BLOCKSIZE; ++x) {
				register const int xHeight = x * height;
				// first four elements in tight loop
				__builtin_prefetch(&large[xHeight + height + yy], 0, 3);
				__builtin_prefetch(&small[xHeight + height + yy], 0, 3);
				// last four elements in tight loop
				__builtin_prefetch(&large[xHeight + height + yy + 4], 0, 3);
				__builtin_prefetch(&small[xHeight + height + yy + 4], 0, 3);
				#pragma GGC unroll 8
				#pragma GCC ivdep
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

	// tiny image
	blurIterationHorizontalFirst(image->data, scratch, 2, width, height);
	blurIterationHorizontal(scratch, one, 2, width, height);
	blurIterationHorizontalTranspose( one, scratch, 2, width, height);
	blurIterationVertical(scratch, one, 2, width, height);

	// small image
	blurIterationHorizontalFirst(image->data, scratch, 3, width, height);
	blurIterationHorizontal(scratch, two, 3, width, height);
	blurIterationHorizontalTranspose(two, scratch,  3, width, height);
	blurIterationVertical(scratch, two, 3, width, height);

	// tinyPPM
	imageDifference(result->data,  one,  two, width, height, 2.0f, 3.0f);
	(argc > 1) ? writePPM("flower_tiny.ppm", result) : writeStreamPPM(stdout, result);

	// medium image
	blurIterationHorizontalFirst(image->data, scratch,  5, width, height);
	blurIterationHorizontal(scratch, one, 5, width, height);
	blurIterationHorizontalTranspose(one, scratch,  5, width, height);
	blurIterationVertical(scratch, one, 5, width, height);

	// smallPPM
	imageDifference(result->data,  two,  one, width, height, 3.0f, 5.0f);
	(argc > 1) ? writePPM("flower_small.ppm", result) : writeStreamPPM(stdout, result);

	// large image
	blurIterationHorizontalFirst(image->data, scratch,  8, width, height);
	blurIterationHorizontal(scratch, two, 8, width, height);
	blurIterationHorizontalTranspose(two, scratch,  8, width, height);
	blurIterationVertical(scratch, two, 8, width, height);

	// mediumPPM
	imageDifference(result->data,  one,  two, width, height, 5.0f, 8.0f);
	(argc > 1) ? writePPM("flower_medium.ppm", result) : writeStreamPPM(stdout, result);
}

