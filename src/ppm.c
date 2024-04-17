
#include "ppm.h"
#include<stdio.h>
#include<stdlib.h>


#define CREATOR "RPFELGUEIRAS"
#define RGB_COMPONENT_COLOR 255


PPMImage* readStreamPPM(FILE* restrict fp) {
	char buff[16];
	PPMImage* image;
	int c, rgb_comp_color;
	//open PPM file for reading
	if (!fp) {
		//fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	//read image format
	if (!fgets(buff, sizeof(buff), fp)) {
		//perror(filename);
		exit(1);
	}
	 
	//check the image format
	if (buff[0] != 'P' || buff[1] != '6') {
		fprintf(stderr, "Invalid image format (must be 'P6')\n");
		exit(1);
	}

	//alloc memory form image
	image = (PPMImage* restrict)aligned_alloc(16, sizeof(PPMImage));
	if (!image) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	//check for comments
	c = getc(fp);
	while (c == '#') {
	while (getc(fp) != '\n') ;
		 c = getc(fp);
	}

	ungetc(c, fp);
	//read image size information
	if (fscanf(fp, "%d %d", &image->x, &image->y) != 2) {
		fprintf(stderr, "Invalid image size (error loading )\n");
		exit(1);
	}

	//read rgb component
	if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
		fprintf(stderr, "Invalid rgb component (error loading )\n");
		exit(1);
	}

	//check rgb component depth
	if (rgb_comp_color!= 255) {
		fprintf(stderr, " does not have 8-bits components\n");
		exit(1);
	}

	while (fgetc(fp) != '\n') ;
	//memory allocation for pixel data
	image->data = (PPMPixel* restrict)aligned_alloc(16, image->x * image->y * sizeof(PPMPixel));

	if (!image) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	//read pixel data from file
     setvbuf(fp, (char*)image->data, _IOFBF, 3 * image->x * image->y);
	if (fread(image->data, 3 * image->x, image->y, fp) != image->y) {
		fprintf(stderr, "Error loading image\n");
		exit(1);
	}
	return image;
}


PPMImage* readPPM(const char* restrict filename)
{
         char buff[16];
         PPMImage *img;
         FILE* fp;
         int c, rgb_comp_color;
         //open PPM file for reading
         fp = fopen(filename, "rb");
         if (!fp) {
              fprintf(stderr, "Unable to open file '%s'\n", filename);
              exit(1);
         }

         //read image format
         if (!fgets(buff, sizeof(buff), fp)) {
              perror(filename);
              exit(1);
         }

    //check the image format
    if (buff[0] != 'P' || buff[1] != '6') {
         fprintf(stderr, "Invalid image format (must be 'P6')\n");
         exit(1);
    }

    //alloc memory form image
    img = (PPMImage* restrict)aligned_alloc(16, sizeof(PPMImage));
    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //check for comments
    c = getc(fp);
    while (c == '#') {
    while (getc(fp) != '\n') ;
         c = getc(fp);
    }

    ungetc(c, fp);
    //read image size information
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
         fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
         exit(1);
    }

    //read rgb component
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
         fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
         exit(1);
    }

    //check rgb component depth
    if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
         exit(1);
    }

    while (fgetc(fp) != '\n') ;
    //memory allocation for pixel data
    img->data = (PPMPixel* restrict)aligned_alloc(16, img->x * img->y * sizeof(PPMPixel));

    if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
    }

    //read pixel data from file
    setvbuf(fp, (char*)img->data, _IOFBF, 3 * img->x * img->y);
    if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
         fprintf(stderr, "Error loading image '%s'\n", filename);
         exit(1);
    }

    fclose(fp);
    return img;
}


void writeStreamPPM(FILE* restrict fp, const PPMImage* restrict img) {
	if (!fp) {
		fprintf(stderr, "Unable to open file\n");
		exit(1);
	}

	//write the header file
	//image format
	fprintf(fp, "P6\n");

	//comments
	fprintf(fp, "# Created by %s\n",CREATOR);

	//image size
	fprintf(fp, "%d %d\n",img->x,img->y);

	// rgb component depth
	fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

	// pixel data
     setvbuf(fp, (char*)img->data, _IOFBF, 3 * img->x * img->y);
	fwrite(img->data, 3 * img->x, img->y, fp);
}

void writePPM(const char* restrict filename, const PPMImage* restrict img)
{
    FILE* fp;
    //open file for output
    fp = fopen(filename, "wb");
    if (!fp) {
         fprintf(stderr, "Unable to open file '%s'\n", filename);
         exit(1);
    }

    //write the header file
    //image format
    fprintf(fp, "P6\n");

    //comments
    fprintf(fp, "# Created by %s\n",CREATOR);

    //image size
    fprintf(fp, "%d %d\n",img->x,img->y);

    // rgb component depth
    fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

    // pixel data
    setvbuf(fp, (char*)img->data, _IOFBF, 3 * img->x * img->y);
    fwrite(img->data, 3 * img->x, img->y, fp);
    fclose(fp);
}

void changeColorPPM(PPMImage* restrict img)
{
     int i;
     if(img){
          #pragma omp parallel for schedule(dynamic, 2) num_threads(8)
          for(i=0;i<img->x*img->y;i++){
              img->data[i].red=RGB_COMPONENT_COLOR-img->data[i].red;
              img->data[i].green=RGB_COMPONENT_COLOR-img->data[i].green;
              img->data[i].blue=RGB_COMPONENT_COLOR-img->data[i].blue;
          }
     }
}
