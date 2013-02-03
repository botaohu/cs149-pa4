
# Comment out the following line to generate release code
DEBUG = 1

SOURCES = main.cc JPEGWriter.cc CpuReference.cc ImageCleaner.cc

ifdef DEBUG
GCCFLAGS    += -g
else
GCCFLAGS    +=
endif

all:
	g++ -ljpeg -fopenmp -o ImageCleaner $(SOURCES) $(GCCFLAGS)

clean:
	rm -f *~ *.o ImageCleaner
