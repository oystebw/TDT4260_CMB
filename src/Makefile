.PHONY: clean check run_reference run zip

LD:=g++
CC:=gcc
CXX:=g++
CFLAGS:=-g -fno-omit-frame-pointer -O2 -I. #-fopenmp -std=c99
CXXFLAGS:=-g -fno-omit-frame-pointer -O2 -I. #-fopenmp -std=c++14 -DON_VM
LDFLAGS:=#-lOpenCL -lpthread -fopenmp -DHAVE_OPENMP

C_SOURCES=$(wildcard *.c)
CPP_SOURCES=$(wildcard *.cpp)
OBJECTS:=$(C_SOURCES:.c=.o) $(CPP_SOURCES:.cpp=.o)
HEADERS:=$(wildcard *.h)
$(OBJECTS): $(HEADERS)

IMAGES:=$(wildcard flower_*.ppm)
ZIPS:=$(wildcard *.zip)
PERF_DATA:=$(wildcard perf.*)
RM:=rm -f
clean:
	$(RM) $(OBJECTS) image_processing_reference image_processing_c image_processing_opencl checker $(IMAGES) $(PERF_DATA)

VERSION ?= c
EXECUTABLE:=image_processing_$(VERSION)

$(EXECUTABLE): image_processing_$(VERSION).o ppm.o
	$(LD) $^ -o $@ $(LDFLAGS)

image_processing_reference: image_processing_reference.o ppm.o
	$(LD) $^ -o $@ $(LDFLAGS)

checker: checker.o ppm.o
	$(LD) $^ -o $@ $(LDFLAGS)

check: checker flower_tiny_correct.ppm flower_small_correct.ppm flower_medium_correct.ppm flower_tiny.ppm flower_small.ppm flower_medium.ppm
	./checker

run_reference flower_tiny_correct.ppm flower_small_correct.ppm flower_medium_correct.ppm: image_processing_reference
	./image_processing_reference

run flower_tiny.ppm flower_small.ppm flower_medium.ppm: $(EXECUTABLE) image_processing_opencl.cl
	time ./$(EXECUTABLE) 1

perf: $(EXECUTABLE)
	perf record -g -F 999 ./$(EXECUTABLE) 1
	perf report

ifeq ($(VERSION), c)
zip: $(C_SOURCES) $(CPP_SOURCES) $(HEADERS)
	zip submission_c.zip ppm.c ppm.h image_processing_c.c
else
zip: $(C_SOURCES) $(CPP_SOURCES) $(HEADERS) image_processing_opencl.cl
	zip submission_opencl.zip ppm.c ppm.h image_processing_opencl.cpp image_processing_opencl.cl
endif