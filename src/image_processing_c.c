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

typedef struct {
	v4Accurate data[3];
} Data;

PPMPixel result_data[1920 * 1200];
Data data[1920 * 1200];

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

void blurIterationHorizontalFirst(const PPMPixel* restrict in, Data* restrict data, const int size, const int width, const int height) {
	const v4Accurate divisor = (v4Accurate){1.0f / (2 * size + 1), 1.0f / (2 * size + 1), 1.0f / (2 * size + 1), 1.0f};
	#pragma omp parallel for schedule(dynamic, 2) num_threads(8)
	for(int y = 0; y < height; ++y) {
		const int yWidth = y * width;

		v4Int sum = {0, 0, 0, 0};

		for(int x = 0; x <= size; ++x) {
			sum += (v4Int){in[yWidth + x].red, in[yWidth + x].green, in[yWidth + x].blue, 0.0f};
		}

		data[yWidth + 0].data[0] = (v4Accurate){sum[0], sum[1], sum[2], 0.0f} / (v4Accurate){size + 1, size + 1, size + 1, 1.0f};

		for(int x = 1; x <= size; ++x) {
			sum += (v4Int){in[yWidth + x + size].red, in[yWidth + x + size].green, in[yWidth + x + size].blue, 0.0};
			data[yWidth + x].data[0] = (v4Accurate){sum[0], sum[1], sum[2], 0.0f} / (v4Accurate){size + x + 1, size + x + 1, size + x + 1, 1.0f};
		}

		#pragma GCC unroll 16
		for(int x = size + 1; x < width - size; ++x) {
			sum -= (v4Int){in[yWidth + x - size - 1].red, in[yWidth + x - size - 1].green, in[yWidth + x - size - 1].blue, 0.0};
			sum += (v4Int){in[yWidth + x + size].red, in[yWidth + x + size].green, in[yWidth + x + size].blue, 0.0};
			data[yWidth + x].data[0] = (v4Accurate){sum[0], sum[1], sum[2], 0.0f} * divisor;
		}

		for(int x = width - size; x < width; ++x) {
			sum -= (v4Int){in[yWidth + x - size - 1].red, in[yWidth + x - size - 1].green, in[yWidth + x - size - 1].blue, 0.0};
			data[yWidth + x].data[0] = (v4Accurate){sum[0], sum[1], sum[2], 0.0f} / (v4Accurate){size + width - x, size + width - x, size + width - x, 1.0f};
		}
	}
}

void blurIterationHorizontal(Data* data, const int size, const int width, const int height, int index) {
	int scratchIndex = 0;
	const v4Accurate divisor = (v4Accurate){1.0f / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0f};
	#pragma omp parallel for schedule(dynamic, 2) num_threads(8)
	for(int y = 0; y < height; ++y) {
		const int yWidth = y * width;
		for(int iteration = 0; iteration < 3; ++iteration) {
			
			v4Accurate sum = {0.0, 0.0, 0.0, 0.0};

			for(int x = 0; x <= size; ++x) {
				sum += data[yWidth + x].data[scratchIndex];
			}

			data[yWidth + 0].data[index] = sum / (v4Accurate){size + 1, size + 1, size + 1, 1.0f};

			for(int x = 1; x <= size; ++x) {
				sum += data[yWidth + x + size].data[scratchIndex];
				data[yWidth + x].data[index] = sum / (v4Accurate){size + x + 1, size + x + 1, size + x + 1, 1.0f};
			}

			#pragma GCC unroll 16
			for(int x = size + 1; x < width - size; ++x) {
				sum -= data[yWidth + x - size - 1].data[scratchIndex];
				sum += data[yWidth + x + size].data[scratchIndex];
				data[yWidth + x].data[index] = sum * divisor;
			}

			for(int x = width - size; x < width; ++x) {
				sum -= data[yWidth + x - size - 1].data[scratchIndex];
				data[yWidth + x].data[index] = sum / (v4Accurate){size + width - x, size + width - x, size + width - x, 1.0f};
			}
			// swap scratchIndex and index
			int temp = index;
			index = scratchIndex;
			scratchIndex = temp;
		}
		// swap scratchIndex and index
		int temp = index;
		index = scratchIndex;
		scratchIndex = temp;
	}
}

void blurIterationHorizontalTranspose(Data* data, const int size, const int width, const int height, int index) {
	const v4Accurate divisor = (v4Accurate){1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0f};
	#pragma omp parallel for schedule(dynamic, 2) num_threads(8)
	for(int y = 0; y < height; ++y) {
		const int yWidth = y * width;

		v4Accurate sum = {0.0, 0.0, 0.0, 0.0};

		for(int x = 0; x <= size; ++x) {
			sum += data[yWidth + x].data[index];
		}

		data[0 * height + y].data[0] = sum / (v4Accurate){size + 1, size + 1, size + 1, 1.0f};

		for(int x = 1; x <= size; ++x) {
			sum += data[yWidth + x + size].data[index];
			data[x * height + y].data[0] = sum / (v4Accurate){size + x + 1, size + x + 1, size + x + 1, 1.0f};
		}

		#pragma GCC unroll 16
		for(int x = size + 1; x < width - size; ++x) {			
			sum -= data[yWidth + x - size - 1].data[index];
			sum += data[yWidth + x + size].data[index];
			data[x * height + y].data[0] = sum * divisor;
		}

		for(int x = width - size; x < width; ++x) {
			sum -= data[yWidth + x - size - 1].data[index];
			data[x * height + y].data[0] = sum / (v4Accurate){size + width - x, size + width - x, size + width - x, 1.0f};
		}
	}
}

void blurIterationVertical(Data* data, const int size, const int width, const int height, int index) {
	int scratchIndex = 0;
	const v4Accurate divisor = (v4Accurate){1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0 / (2 * size + 1), 1.0f};
	#pragma omp parallel for schedule(dynamic, 2) num_threads(8)
	for(int x = 0; x < width; ++x) {
		const int xHeight = x * height;
		for(int iteration = 0; iteration < 5; ++iteration) {
			v4Accurate sum = {0.0, 0.0, 0.0, 0.0};

			for(int y = 0; y <= size; ++y) {
				sum += data[xHeight + y].data[scratchIndex];
			}

			data[xHeight + 0].data[index] = sum / (v4Accurate){size + 1, size + 1, size + 1, 1.0f};

			for(int y = 1; y <= size; ++y) {
				sum += data[xHeight + y + size].data[scratchIndex];
				data[xHeight + y].data[index] = sum / (v4Accurate){y + size + 1, y + size + 1, y + size + 1, 1.0f};
			}

			#pragma GCC unroll 16
			for(int y = size + 1; y < height - size; ++y) {
				sum -= data[xHeight + y - size - 1].data[scratchIndex];
				sum += data[xHeight + y + size].data[scratchIndex];
				data[xHeight + y].data[index] = sum * divisor;
			}

			for(int y = height - size; y < height; ++y) {
				sum -= data[xHeight + y - size - 1].data[scratchIndex];
				data[xHeight + y].data[index] = sum / (v4Accurate){size + height - y, size + height - y, size + height - y, 1.0f};
			}
			// swap scratchIndex and index
			int temp = index;
			index = scratchIndex;
			scratchIndex = temp;
		}
		// swap scratchIndex and index
		int temp = index;
		index = scratchIndex;
		scratchIndex = temp;
	}
}

void imageDifference(PPMPixel* restrict imageOut, const Data* restrict data, const int width, const int height, const int indexLarge, const int indexSmall) {
	
	#pragma omp parallel for schedule(dynamic, 2) num_threads(8)
	for(int yy = 0; yy < height; yy += BLOCKSIZE) {
		for(int xx = 0; xx < width; xx += BLOCKSIZE) {
			for(int x = xx; x < xx + BLOCKSIZE; ++x) {
				const int xHeight = x * height;
				__builtin_prefetch((float*)&data[xHeight + height + yy].data[0], 0, 3);
				#pragma GGC unroll 8
				for(int y = yy; y < yy + BLOCKSIZE; ++y) {
					v4Accurate diff = data[xHeight + y].data[indexLarge] - data[xHeight + y].data[indexSmall];
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

	blurIterationHorizontalFirst(image->data,  data,  2, width, height);
	blurIterationHorizontal(data,  2, width, height, 1);
	blurIterationHorizontalTranspose(data,  2, width, height, 1);
	blurIterationVertical(data,  2, width, height, 1);

	blurIterationHorizontalFirst(image->data,  data,  3, width, height);
	blurIterationHorizontal(data, 3, width, height, 2);
	blurIterationHorizontalTranspose(data, 3, width, height, 2);
	blurIterationVertical(data, 3, width, height, 2);

	imageDifference(result_data, data, width, height, 2, 1);
	(argc > 1) ? writePPM("flower_tiny.ppm", result) : writeStreamPPM(stdout, result);

	blurIterationHorizontalFirst(image->data,  data,  5, width, height);
	blurIterationHorizontal(data,  5, width, height, 1);
	blurIterationHorizontalTranspose(data,  5, width, height, 1);
	blurIterationVertical(data,  5, width, height, 1);
	imageDifference(result_data, data, width, height, 1, 2);
	(argc > 1) ? writePPM("flower_small.ppm", result) : writeStreamPPM(stdout, result);

	blurIterationHorizontalFirst(image->data,  data,  8, width, height);
	blurIterationHorizontal(data,  8, width, height, 2);
	blurIterationHorizontalTranspose(data,  8, width, height, 2);
	blurIterationVertical(data,  8, width, height, 2);
	imageDifference(result_data, data, width, height, 2, 1);
	(argc > 1) ? writePPM("flower_medium.ppm", result) : writeStreamPPM(stdout, result);
}

