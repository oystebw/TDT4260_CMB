#include <stdio.h>

typedef struct {
     unsigned char red,green,blue;
} PPMPixel;

typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;

PPMImage *readStreamPPM(FILE*  fp);
PPMImage *readPPM(const char* restrict filename);
void writeStreamPPM(FILE* restrict fp, const PPMImage* restrict img);
void writePPM(const char* restrict filename, const PPMImage* restrict img);
void changeColorPPM(PPMImage* restrict img);
