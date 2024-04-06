#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <omp.h>

#include "ppm.h"

// Image from:
// http://7-themes.com/6971875-funny-flowers-pictures.html

typedef struct {
     float red,green,blue;
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
		imageAccurate->data[i].red   = (float) image->data[i].red;
		imageAccurate->data[i].green = (float) image->data[i].green;
		imageAccurate->data[i].blue  = (float) image->data[i].blue;
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
        imageOut->data[i].red = imageIn->data[i].red;
        imageOut->data[i].green = imageIn->data[i].green;
        imageOut->data[i].blue = imageIn->data[i].blue;
    }
    return imageOut;
}

// blur one color channel
void blurIteration(AccurateImage *imageOut, AccurateImage *imageIn, int colourType, int size) {
	
	const int width = imageIn->x;
	const int height = imageIn->y;
	
	// Iterate over each pixel
	for(int senterX = 0; senterX < width; senterX++) {

		for(int senterY = 0; senterY < height; senterY++) {

			// For each pixel we compute the magic number
			float sum = 0;
			int countIncluded = 0;
			int currentX;
			for(int x = -size; x <= size; x++) {
				currentX = senterX + x;
				// Check if we are outside the bounds
				if(currentX < 0 || currentX >= width)
					continue;
				int currentY;
				for(int y = -size; y <= size; y++) {
					currentY = senterY + y;
					// Check if we are outside the bounds
					if(currentY < 0 || currentY >= height)
						continue;

					float* colors = &(imageIn->data[width * currentY + currentX]);
					sum += colors[colourType];

					// Keep track of how many values we have included
					countIncluded++;
				}

			}

			// Now we compute the final value

			float* colors = &(imageOut->data[width * senterY + senterX]);
			colors[colourType] = sum / countIncluded;
		}

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
		float value = (imageInLarge->data[i].red - imageInSmall->data[i].red);
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

		value = (imageInLarge->data[i].green - imageInSmall->data[i].green);
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

		value = (imageInLarge->data[i].blue - imageInSmall->data[i].blue);
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
        blurIteration(imageAccurate2_tiny, imageAccurate1_tiny, colour, size);
        blurIteration(imageAccurate1_tiny, imageAccurate2_tiny, colour, size);
        blurIteration(imageAccurate2_tiny, imageAccurate1_tiny, colour, size);
        blurIteration(imageAccurate1_tiny, imageAccurate2_tiny, colour, size);
        blurIteration(imageAccurate2_tiny, imageAccurate1_tiny, colour, size);
	}
	
	
	AccurateImage* imageAccurate1_small = convertToAccurateImage(image);
	AccurateImage* imageAccurate2_small = convertToAccurateImage(image);
	
	// Process the small case:
	for(int colour = 0; colour < 3; colour++) {
		int size = 3;
        blurIteration(imageAccurate2_small, imageAccurate1_small, colour, size);
        blurIteration(imageAccurate1_small, imageAccurate2_small, colour, size);
        blurIteration(imageAccurate2_small, imageAccurate1_small, colour, size);
        blurIteration(imageAccurate1_small, imageAccurate2_small, colour, size);
        blurIteration(imageAccurate2_small, imageAccurate1_small, colour, size);
	}

    // an intermediate step can be saved for debugging like this
//    writePPM("imageAccurate2_tiny.ppm", convertToPPPMImage(imageAccurate2_tiny));
	
	AccurateImage* imageAccurate1_medium = convertToAccurateImage(image);
	AccurateImage* imageAccurate2_medium = convertToAccurateImage(image);
	
	// Process the medium case:
	for(int colour = 0; colour < 3; colour++) {
		int size = 5;
        blurIteration(imageAccurate2_medium, imageAccurate1_medium, colour, size);
        blurIteration(imageAccurate1_medium, imageAccurate2_medium, colour, size);
        blurIteration(imageAccurate2_medium, imageAccurate1_medium, colour, size);
        blurIteration(imageAccurate1_medium, imageAccurate2_medium, colour, size);
        blurIteration(imageAccurate2_medium, imageAccurate1_medium, colour, size);
	}
	
	AccurateImage* imageAccurate1_large = convertToAccurateImage(image);
	AccurateImage* imageAccurate2_large = convertToAccurateImage(image);
	
	// Do each color channel
	for(int colour = 0; colour < 3; colour++) {
		int size = 8;
        blurIteration(imageAccurate2_large, imageAccurate1_large, colour, size);
        blurIteration(imageAccurate1_large, imageAccurate2_large, colour, size);
        blurIteration(imageAccurate2_large, imageAccurate1_large, colour, size);
        blurIteration(imageAccurate1_large, imageAccurate2_large, colour, size);
        blurIteration(imageAccurate2_large, imageAccurate1_large, colour, size);
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

