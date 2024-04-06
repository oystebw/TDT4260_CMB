#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <omp.h>

#include "ppm.h"

#define DIVISOR2 5
#define DIVISOR3 7
#define DIVISOR5 11
#define DIVISOR8 17

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

typedef struct {
     float rgb[3];
} AccuratePixel;

typedef struct {
     int x, y;
     AccuratePixel *data;
} AccurateImage;

// Convert ppm to high precision format.
AccurateImage* convertToAccurateImage(PPMImage* image) {
	const int size = image->x * image->y;
	// Make a copy
	AccurateImage* imageAccurate;
	imageAccurate = (AccurateImage*)malloc(sizeof(AccurateImage));
	imageAccurate->data = (AccuratePixel*)malloc(size * sizeof(AccuratePixel));
	for(int i = 0; i < size; i++) {
		imageAccurate->data[i].rgb[0]   = (float) image->data[i].red;
		imageAccurate->data[i].rgb[1] = (float) image->data[i].green;
		imageAccurate->data[i].rgb[2]  = (float) image->data[i].blue;
	}
	imageAccurate->x = image->x;
	imageAccurate->y = image->y;
	
	return imageAccurate;
}

PPMImage* convertToPPPMImage(AccurateImage* imageIn) {
    const int size = imageIn->x * imageIn->y;
	
	PPMImage *imageOut;
    imageOut = (PPMImage*)malloc(sizeof(PPMImage));
    imageOut->data = (PPMPixel*)malloc(size * sizeof(PPMPixel));

    imageOut->x = imageIn->x;
    imageOut->y = imageIn->y;

    for(int i = 0; i < size; i++) {
        imageOut->data[i].red = imageIn->data[i].rgb[0];
        imageOut->data[i].green = imageIn->data[i].rgb[1];
        imageOut->data[i].blue = imageIn->data[i].rgb[2];
    }
    return imageOut;
}

// blur one color channel
void blurIteration2(AccurateImage* image, AccurateImage* scratch, int colourType) {
	
	const int width = image->x;
	const int height = image->y;

	float sum;
	
	// Iterate over each pixel
	for(int y = 0; y < height; y++) {
		sum = image->data[y * width + 0].rgb[colourType];
		sum += image->data[y * width + 1].rgb[colourType];
		sum += image->data[y * width + 2].rgb[colourType];
		scratch->data[y * width + 0].rgb[colourType] = sum / 3;
		sum += image->data[y * width + 3].rgb[colourType];
		scratch->data[y * width + 1].rgb[colourType] = sum / 4;
		sum += image->data[y * width + 4].rgb[colourType];
		scratch->data[y * width + 2].rgb[colourType] = sum / DIVISOR2;

		for(int x = 3; x < width - 2; x++) {
			sum -= image->data[y * width + x - 3].rgb[colourType];
			sum += image->data[y * width + x + 2].rgb[colourType];
			scratch->data[y * width + x].rgb[colourType] = sum / DIVISOR2;
		}

		sum -= image->data[y * width + width - 5].rgb[colourType];
		scratch->data[y * width + width - 2].rgb[colourType] = sum / 4;
		sum -= image->data[y * width + width - 4].rgb[colourType];
		scratch->data[y * width + width - 1].rgb[colourType] = sum / 3;
	}

	for(int x = 0; x < width; x++) {
		sum = scratch->data[0 * width + x].rgb[colourType];
		sum += scratch->data[1 * width + x].rgb[colourType];
		sum += scratch->data[2 * width + x].rgb[colourType];
		image->data[0 * width + x].rgb[colourType] = sum / 3;
		sum += scratch->data[3 * width + x].rgb[colourType];
		image->data[1 * width + x].rgb[colourType] = sum / 4;
		sum += scratch->data[4 * width + x].rgb[colourType];
		image->data[2 * width + x].rgb[colourType] = sum / DIVISOR2;

		for(int y = 3; y < height - 2; y++) {
			sum -= scratch->data[(y - 3) * width + x].rgb[colourType];
			sum += scratch->data[(y + 2) * width + x].rgb[colourType];
			image->data[y * width + x].rgb[colourType] = sum / DIVISOR2;
		}

		sum -= scratch->data[(height - 5) * width + x].rgb[colourType];
		image->data[(height - 2) * width + x].rgb[colourType] = sum / 4;
		sum -= scratch->data[(height - 4) * width + x].rgb[colourType];
		image->data[(height - 1) * width + x].rgb[colourType] = sum / 3;
	}
}


// Perform the final step, and return it as ppm.
PPMImage* imageDifference(AccurateImage* imageInSmall, AccurateImage* imageInLarge) {
	const int width = imageInSmall->x;
	const int height = imageInSmall->y;
	const int size = width * height;

	PPMImage* imageOut;
	imageOut = (PPMImage*)malloc(sizeof(PPMImage));
	imageOut->data = (PPMPixel*)malloc(size * sizeof(PPMPixel));

	imageOut->x = width;
	imageOut->y = height;

	for(int i = 0; i < size; i++) {
		float value = (imageInLarge->data[i].rgb[0] - imageInSmall->data[i].rgb[0]);
		if(value > 255)
			imageOut->data[i].red = 255;
		else if (value < -1.0) {
			value = 257.0+value;
			if(value > 255)
				imageOut->data[i].red = 255;
			else
				imageOut->data[i].red = floor(value);
		} else if (value > -1.0 && value < 0.0) {
			imageOut->data[i].red = 0;
		} else {
			imageOut->data[i].red = floor(value);
		}

		value = (imageInLarge->data[i].rgb[1] - imageInSmall->data[i].rgb[1]);
		if(value > 255)
			imageOut->data[i].green = 255;
		else if (value < -1.0) {
			value = 257.0+value;
			if(value > 255)
				imageOut->data[i].green = 255;
			else
				imageOut->data[i].green = floor(value);
		} else if (value > -1.0 && value < 0.0) {
			imageOut->data[i].green = 0;
		} else {
			imageOut->data[i].green = floor(value);
		}

		value = (imageInLarge->data[i].rgb[2] - imageInSmall->data[i].rgb[2]);
		if(value > 255)
			imageOut->data[i].blue = 255;
		else if (value < -1.0) {
			value = 257.0+value;
			if(value > 255)
				imageOut->data[i].blue = 255;
			else
				imageOut->data[i].blue = floor(value);
		} else if (value > -1.0 && value < 0.0) {
			imageOut->data[i].blue = 0;
		} else {
			imageOut->data[i].blue = floor(value);
		}
	}
	return imageOut;
}


int main(int argc, char** argv) {
    // read image
    PPMImage *image;
    // select where to read the image from
    if(argc > 1) {
        // from file for debugging (with argument)
        image = readPPM("flower.ppm");
    } else {
        // from stdin for cmb
        image = readStreamPPM(stdin);
    }
	
	
	AccurateImage* imageAccurate1_tiny = convertToAccurateImage(image);
	AccurateImage* imageAccurate2_tiny = convertToAccurateImage(image);
	
	// Process the tiny case:
	for(int colour = 0; colour < 3; colour++) {
		int size = 2;
        blurIteration2(imageAccurate1_tiny, imageAccurate2_tiny, colour);
        blurIteration2(imageAccurate1_tiny, imageAccurate2_tiny, colour);
        blurIteration2(imageAccurate1_tiny, imageAccurate2_tiny, colour);
        blurIteration2(imageAccurate1_tiny, imageAccurate2_tiny, colour);
        blurIteration2(imageAccurate1_tiny, imageAccurate2_tiny, colour);
	}
	
	
	AccurateImage* imageAccurate1_small = convertToAccurateImage(image);
	AccurateImage* imageAccurate2_small = convertToAccurateImage(image);
	
	// Process the small case:
	for(int colour = 0; colour < 3; colour++) {
		int size = 3;
        blurIteration2(imageAccurate1_small, imageAccurate2_small, colour);
        blurIteration2(imageAccurate1_small, imageAccurate2_small, colour);
        blurIteration2(imageAccurate1_small, imageAccurate2_small, colour);
        blurIteration2(imageAccurate1_small, imageAccurate2_small, colour);
        blurIteration2(imageAccurate1_small, imageAccurate2_small, colour);
	}

    // an intermediate step can be saved for debugging like this
//    writePPM("imageAccurate2_tiny.ppm", convertToPPPMImage(imageAccurate2_tiny));
	
	AccurateImage* imageAccurate1_medium = convertToAccurateImage(image);
	AccurateImage* imageAccurate2_medium = convertToAccurateImage(image);
	
	// Process the medium case:
	for(int colour = 0; colour < 3; colour++) {
		int size = 5;
        blurIteration2(imageAccurate1_medium, imageAccurate2_medium, colour);
        blurIteration2(imageAccurate1_medium, imageAccurate2_medium, colour);
        blurIteration2(imageAccurate1_medium, imageAccurate2_medium, colour);
        blurIteration2(imageAccurate1_medium, imageAccurate2_medium, colour);
        blurIteration2(imageAccurate1_medium, imageAccurate2_medium, colour);
	}
	
	AccurateImage* imageAccurate1_large = convertToAccurateImage(image);
	AccurateImage* imageAccurate2_large = convertToAccurateImage(image);
	
	// Do each color channel
	for(int colour = 0; colour < 3; colour++) {
		int size = 8;
        blurIteration2(imageAccurate1_large, imageAccurate2_large, colour);
        blurIteration2(imageAccurate1_large, imageAccurate2_large, colour);
        blurIteration2(imageAccurate1_large, imageAccurate2_large, colour);
        blurIteration2(imageAccurate1_large, imageAccurate2_large, colour);
        blurIteration2(imageAccurate1_large, imageAccurate2_large, colour);
	}
	// calculate difference
	PPMImage* final_tiny = imageDifference(imageAccurate2_tiny, imageAccurate2_small);
    PPMImage* final_small = imageDifference(imageAccurate2_small, imageAccurate2_medium);
    PPMImage* final_medium = imageDifference(imageAccurate2_medium, imageAccurate2_large);
	// Save the images.
    if(argc > 1) {
        writePPM("flower_tiny.ppm", final_tiny);
        writePPM("flower_small.ppm", final_small);
        writePPM("flower_medium.ppm", final_medium);
    } else {
        writeStreamPPM(stdout, final_tiny);
        writeStreamPPM(stdout, final_small);
        writeStreamPPM(stdout, final_medium);
    }
	
}

