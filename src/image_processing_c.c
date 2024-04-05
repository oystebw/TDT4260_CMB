#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <omp.h>

#include "ppm.h"

#define CREATOR "RPFELGUEIRAS"
#define RGB_COMPONENT_COLOR 255

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

typedef struct {
     float red,green,blue;
} AccuratePixel;

typedef struct {
     int x, y;
     AccuratePixel *data;
} AccurateImage;


PPMPixel* readImage(const char* path, int* width, int* height) {
  FILE* file = path ? fopen(path, "rb") : stdin;
  assert(file);
  char buffer[16];
  assert(fgets(buffer, sizeof(buffer), file));
  assert(buffer[0] == 'P' && buffer[1] == '6');
  
  while (getc(file) == '#')
    while (getc(file) != '\n');
  fseek(file, -1, SEEK_CUR);

  int d;
  assert(fscanf(file, "%d %d\n%d\n", width, height, &d) == 3);
  assert(d == 255);

  PPMPixel* data = malloc(*width * *height * sizeof(PPMPixel));
  assert(fread(data, *width * *height * sizeof(PPMPixel), 1, file) == 1);
  if (path) fclose(file);

  return data;
}

void writeImage(const char* path, PPMPixel* data, const int width, const int height) {
  FILE* file = path ? fopen(path, "wb") : stdout;
  assert(file);
  fprintf(file, "P6\n# Created by RPFELGUEIRAS\n%d %d\n255\n", width, height);
  fwrite(data, width * height * sizeof(PPMPixel), 1, file);
  if (path) fclose(file);
}

// Convert ppm to high precision format.
void convertToAccurateImage(AccuratePixel* imageAccurate, PPMPixel* image, const int width, const int height) {
	const int size = width * height;
	
	for(int i = 0; i < size; i++) {
		imageAccurate[i].red   = (float) image[i].red;
		imageAccurate[i].green = (float) image[i].green;
		imageAccurate[i].blue  = (float) image[i].blue;
	}
}

PPMPixel* convertToPPPMImage(AccuratePixel* imageIn, const int width, const int height) {
    const int size = width * height;
	
	PPMPixel* imageOut;
    imageOut = (PPMPixel*)malloc(size * sizeof(PPMPixel));

    for(int i = 0; i < size; i++) {
        imageOut[i].red = imageIn[i].red;
        imageOut[i].green = imageIn[i].green;
        imageOut[i].blue = imageIn[i].blue;
    }
    return imageOut;
}

// blur one color channel
void blurIteration(AccuratePixel* imageOut, AccuratePixel* imageIn, const int colourType, const int size, const int width, const int height) {
	
	// Iterate over each pixel
	for(int senterX = 0; senterX < width; senterX++) {

		for(int senterY = 0; senterY < height; senterY++) {

			// For each pixel we compute the magic number
			float sum = 0;
			int countIncluded = 0;
			for(int x = -size; x <= size; x++) {
				const int currentX = senterX + x;
				// Check if we are outside the bounds
				if(currentX < 0 || currentX >= width)
					continue;
				for(int y = -size; y <= size; y++) {
					const int currentY = senterY + y;
					// Check if we are outside the bounds
					if(currentY < 0 || currentY >= height)
						continue;

					float* colors = &(imageIn[width * currentY + currentX]);
					sum += colors[colourType];

					// Keep track of how many values we have included
					countIncluded++;
				}

			}

			// Now we compute the final value
			float* colors = &(imageOut[width * senterY + senterX]);
			colors[colourType] = sum / countIncluded;
		}

	}
	
}


// Perform the final step, and return it as ppm.
PPMPixel* imageDifference(AccuratePixel* imageInSmall, AccuratePixel* imageInLarge, const int width, const int height) {
	const int size = width * height;

	PPMPixel* imageOut;
	imageOut = (PPMPixel*)malloc(size * sizeof(PPMPixel));

	for(int i = 0; i < size; i++) {
		float value = (imageInLarge[i].red - imageInSmall[i].red);
		if(value > 255)
			imageOut[i].red = 255;
		else if (value < -1.0) {
			value = 257.0+value;
			if(value > 255)
				imageOut[i].red = 255;
			else
				imageOut[i].red = floor(value);
		} else if (value > -1.0 && value < 0.0) {
			imageOut[i].red = 0;
		} else {
			imageOut[i].red = floor(value);
		}

		value = (imageInLarge[i].green - imageInSmall[i].green);
		if(value > 255)
			imageOut[i].green = 255;
		else if (value < -1.0) {
			value = 257.0+value;
			if(value > 255)
				imageOut[i].green = 255;
			else
				imageOut[i].green = floor(value);
		} else if (value > -1.0 && value < 0.0) {
			imageOut[i].green = 0;
		} else {
			imageOut[i].green = floor(value);
		}

		value = (imageInLarge[i].blue - imageInSmall[i].blue);
		if(value > 255)
			imageOut[i].blue = 255;
		else if (value < -1.0) {
			value = 257.0+value;
			if(value > 255)
				imageOut[i].blue = 255;
			else
				imageOut[i].blue = floor(value);
		} else if (value > -1.0 && value < 0.0) {
			imageOut[i].blue = 0;
		} else {
			imageOut[i].blue = floor(value);
		}
	}
	return imageOut;
}


int main(int argc, char** argv) {
	int file_input = argc > 0;
	int width;
	int height;
    
	PPMPixel* image = readImage(file_input ? "flower.ppm" : 0, &width, &height);
	const int size = width * height;
	
	AccuratePixel* imageAccurate1_tiny = (AccuratePixel*)malloc(size * sizeof(AccuratePixel));
	AccuratePixel* imageAccurate2_tiny = (AccuratePixel*)malloc(size * sizeof(AccuratePixel));
	AccuratePixel* imageAccurate1_small = (AccuratePixel*)malloc(size * sizeof(AccuratePixel));
	AccuratePixel* imageAccurate2_small = (AccuratePixel*)malloc(size * sizeof(AccuratePixel));
	AccuratePixel* imageAccurate1_medium = (AccuratePixel*)malloc(size * sizeof(AccuratePixel));
	AccuratePixel* imageAccurate2_medium = (AccuratePixel*)malloc(size * sizeof(AccuratePixel));
	AccuratePixel* imageAccurate1_large = (AccuratePixel*)malloc(size * sizeof(AccuratePixel));
	AccuratePixel* imageAccurate2_large = (AccuratePixel*)malloc(size * sizeof(AccuratePixel));
	
	convertToAccurateImage(imageAccurate1_tiny, image, width, height);
	memcpy(imageAccurate1_small, imageAccurate1_tiny, size * sizeof(AccuratePixel));
	memcpy(imageAccurate1_medium, imageAccurate1_tiny, size * sizeof(AccuratePixel));
	memcpy(imageAccurate1_large, imageAccurate1_tiny, size * sizeof(AccuratePixel));
	
	// Process the tiny case:
	for(int colour = 0; colour < 3; colour++) {
		int size = 2;
        blurIteration(imageAccurate2_tiny, imageAccurate1_tiny, colour, size, width, height);
        blurIteration(imageAccurate1_tiny, imageAccurate2_tiny, colour, size, width, height);
        blurIteration(imageAccurate2_tiny, imageAccurate1_tiny, colour, size, width, height);
        blurIteration(imageAccurate1_tiny, imageAccurate2_tiny, colour, size, width, height);
        blurIteration(imageAccurate2_tiny, imageAccurate1_tiny, colour, size, width, height);
	}
	
	// Process the small case:
	for(int colour = 0; colour < 3; colour++) {
		int size = 3;
        blurIteration(imageAccurate2_small, imageAccurate1_small, colour, size, width, height);
        blurIteration(imageAccurate1_small, imageAccurate2_small, colour, size, width, height);
        blurIteration(imageAccurate2_small, imageAccurate1_small, colour, size, width, height);
        blurIteration(imageAccurate1_small, imageAccurate2_small, colour, size, width, height);
        blurIteration(imageAccurate2_small, imageAccurate1_small, colour, size, width, height);
	}

    // an intermediate step can be saved for debugging like this
//    writePPM("imageAccurate2_tiny.ppm", convertToPPPMImage(imageAccurate2_tiny));

	// Process the medium case:
	for(int colour = 0; colour < 3; colour++) {
		int size = 5;
        blurIteration(imageAccurate2_medium, imageAccurate1_medium, colour, size, width, height);
        blurIteration(imageAccurate1_medium, imageAccurate2_medium, colour, size, width, height);
        blurIteration(imageAccurate2_medium, imageAccurate1_medium, colour, size, width, height);
        blurIteration(imageAccurate1_medium, imageAccurate2_medium, colour, size, width, height);
        blurIteration(imageAccurate2_medium, imageAccurate1_medium, colour, size, width, height);
	}
	
	// Do each color channel
	for(int colour = 0; colour < 3; colour++) {
		int size = 8;
        blurIteration(imageAccurate2_large, imageAccurate1_large, colour, size, width, height);
        blurIteration(imageAccurate1_large, imageAccurate2_large, colour, size, width, height);
        blurIteration(imageAccurate2_large, imageAccurate1_large, colour, size, width, height);
        blurIteration(imageAccurate1_large, imageAccurate2_large, colour, size, width, height);
        blurIteration(imageAccurate2_large, imageAccurate1_large, colour, size, width, height);
	}
	// calculate difference
	PPMPixel* final_tiny = imageDifference(imageAccurate2_tiny, imageAccurate2_small, width, height);
    PPMPixel* final_small = imageDifference(imageAccurate2_small, imageAccurate2_medium, width, height);
    PPMPixel* final_medium = imageDifference(imageAccurate2_medium, imageAccurate2_large, width, height);
	// Save the images.

	writeImage(file_input ? "flower_tiny.ppm"   : 0, final_tiny,   width, height);
	writeImage(file_input ? "flower_small.ppm"  : 0, final_small,  width, height);
	writeImage(file_input ? "flower_medium.ppm" : 0, final_medium, width, height);
}

