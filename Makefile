CXX=g++
CXXFLAGS=-Wall -g `pkg-config --cflags opencv`
LDFLAGS=`pkg-config --libs opencv`
SOURCES=compare.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=compare

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) -o $@ $(OBJECTS) $(LDFLAGS)

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<
