#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <omp.h>
#include <pthread.h>

#include "ppm.h"
#define THREAD_NUMS 8

typedef float v4Accurate __attribute__((vector_size(16)));
typedef __uint32_t v4Int __attribute__((vector_size(16)));

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

typedef struct {
     int x, y;
     v4Accurate* data;
} AccurateImage;


void blurIterationHorizontalFirst(PPMPixel* in, v4Accurate* out, const int size, const int width, const int height, const int offset) {
	
	for(int y = offset; y < height; y += 2) {
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
			out[yWidth + x] = (v4Accurate){sum[0], sum[1], sum[2], sum[3]} / (v4Accurate){2 * size + 1, 2 * size + 1, 2 * size + 1, 2 * size + 1};
		}

		for(int x = width - size; x < width; x++) {
			PPMPixel pixel = in[yWidth + x - size - 1];
			sum -= (v4Int){pixel.red, pixel.green, pixel.blue, 0.0};
			out[yWidth + x] = (v4Accurate){sum[0], sum[1], sum[2], sum[3]} / (v4Accurate){size + width - x, size + width - x, size + width - x, size + width - x};
		}
	}
}

void blurIterationHorizontal(v4Accurate* in, v4Accurate* out, const int size, const int width, const int height, const int offset) {
	
	for(int y = offset; y < height; y += 2) {
		const int yWidth = y * width;

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
			out[yWidth + x] = sum / (v4Accurate){2 * size + 1, 2 * size + 1, 2 * size + 1, 2 * size + 1};
		}

		for(int x = width - size; x < width; x++) {
			sum -= in[yWidth + x - size - 1];
			out[yWidth + x] = sum / (v4Accurate){size + width - x, size + width - x, size + width - x, size + width - x};
		}
	}
}

void blurIterationHorizontalTranspose(v4Accurate* in, v4Accurate* out, const int size, const int width, const int height, const int offset) {
	
	for(int y = offset; y < height; y += 2) {
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
			out[x * height + y] = sum / (v4Accurate){2 * size + 1, 2 * size + 1, 2 * size + 1, 2 * size + 1};
		}

		for(int x = width - size; x < width; x++) {
			sum -= in[yWidth + x - size - 1];
			out[x * height + y] = sum / (v4Accurate){size + width - x, size + width - x, size + width - x, size + width - x};
		}
	}
}

void blurIterationVertical(v4Accurate* in, v4Accurate* out, const int size, const int width, const int height, const int offset) {

	for(int x = offset; x < width; x += 2) {
		const int xHeight = x * height;

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
			out[xHeight + y] = sum / (v4Accurate){2 * size + 1, 2 * size + 1, 2 * size + 1, 2 * size + 1};
		}

		for(int y = height - size; y < height; y++) {
			sum -= in[xHeight + y - size - 1];
			out[xHeight + y] = sum / (v4Accurate){size + height - y, size + height - y, size + height - y, size + height - y};
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

struct args{
    int offset;
	int width;
	int height;
	int size;
    PPMPixel* ppmImage;
	v4Accurate* result;
	v4Accurate* scratch;
};

void* threadFuncHorizontal(void* arg) {
	struct args* args = (struct args*)arg;
	const int offset = args->offset;
	const int width = args->width;
	const int height = args->height;
	const int size = args->size;
	PPMPixel* ppmImage = args->ppmImage;
	v4Accurate* result = args->result;
	v4Accurate* scratch = args->scratch;

	blurIterationHorizontalFirst(ppmImage, scratch, size, width, height, offset);
	blurIterationHorizontal(scratch, result, size, width, height, offset);
	blurIterationHorizontal(result, scratch, size, width, height, offset);
	blurIterationHorizontal(scratch, result, size, width, height, offset);
}


void* threadFuncHorizontalTranspose(void* arg) {
	struct args* args = (struct args*)arg;
	const int offset = args->offset;
	const int width = args->width;
	const int height = args->height;
	const int size = args->size;
	PPMPixel* ppmImage = args->ppmImage;
	v4Accurate* result = args->result;
	v4Accurate* scratch = args->scratch;

	blurIterationHorizontalTranspose(result, scratch, size, width, height, offset);
}

void* threadFuncVertical(void* arg) {
	struct args* args = (struct args*)arg;
	const int offset = args->offset;
	const int width = args->width;
	const int height = args->height;
	const int size = args->size;
	PPMPixel* ppmImage = args->ppmImage;
	v4Accurate* result = args->result;
	v4Accurate* scratch = args->scratch;

	blurIterationVertical(scratch, result, size, width, height, offset);
	blurIterationVertical(result, scratch, size, width, height, offset);
	blurIterationVertical(scratch, result, size, width, height, offset);
	blurIterationVertical(result, scratch, size, width, height, offset);
	blurIterationVertical(scratch, result, size, width, height, offset);
}


int main(int argc, char** argv) {

    PPMImage* image;
    if(argc > 1) {
        image = readPPM("flower.ppm");
    } else {
        image = readStreamPPM(stdin);
    }

	const int width = image->x;
	const int height = image->y;
	const int size = width * height;

	const int sizes[4] = {2, 3, 5, 8};

	AccurateImage** images = (AccurateImage**)malloc(sizeof(AccurateImage*) * 4);
	v4Accurate* scratches = (v4Accurate*)malloc(sizeof(v4Accurate) * size * 4);

	for(int i = 0; i < 4; i++) {
		images[i] = (AccurateImage*)malloc(sizeof(AccurateImage));
		images[i]->x = width;
		images[i]->y = height;
		images[i]->data = (v4Accurate*)malloc(size * sizeof(v4Accurate));
	}

	pthread_t threads[THREAD_NUMS];

	for(int i = 0; i < THREAD_NUMS; i++) {
		struct args* args = (struct args*)malloc(sizeof(struct args));
		args->offset = i / 4;
		args->width = width;
		args->height = height;
		args->size = sizes[i % 4];
		args->ppmImage = image->data;
		args->result = images[i % 4]->data;
		args->scratch = scratches + (i % 4) * size;
		pthread_create(&threads[i], NULL, threadFuncHorizontal, (void*)args);
	}

	for(int i = 0; i < THREAD_NUMS; i++) {
		pthread_join(threads[i], NULL);
	}

	for(int i = 0; i < THREAD_NUMS; i++) {
		struct args* args = (struct args*)malloc(sizeof(struct args));
		args->offset = i / 4;
		args->width = width;
		args->height = height;
		args->size = sizes[i % 4];
		args->ppmImage = image->data;
		args->result = images[i % 4]->data;
		args->scratch = scratches + (i % 4) * size;
		pthread_create(&threads[i], NULL, threadFuncHorizontalTranspose, (void*)args);
	}

	for(int i = 0; i < THREAD_NUMS; i++) {
		pthread_join(threads[i], NULL);
	}

	for(int i = 0; i < THREAD_NUMS; i++) {
		struct args* args = (struct args*)malloc(sizeof(struct args));
		args->offset = i / 4;
		args->width = width;
		args->height = height;
		args->size = sizes[i % 4];
		args->ppmImage = image->data;
		args->result = images[i % 4]->data;
		args->scratch = scratches + (i % 4) * size;
		pthread_create(&threads[i], NULL, threadFuncVertical, (void*)args);
	}

	for(int i = 0; i < THREAD_NUMS; i++) {
		pthread_join(threads[i], NULL);
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

