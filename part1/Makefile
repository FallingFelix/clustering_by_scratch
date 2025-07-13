# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -Wall -Wshadow -Wunreachable-code \
          -Wredundant-decls -Wmissing-declarations \
          -Wuninitialized -Wno-unused-parameter -O2


# OpenCV flags
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

# Target executable
TARGET = part_one 

# Source files
SOURCES = part_one.cpp

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(OPENCV_FLAGS)

# Clean build files
clean:
	rm -f $(TARGET)

# Rebuild everything
rebuild: clean all

.PHONY: all clean rebuild
rebuild: clean all

.PHONY: all clean rebuild
.PHONY: all clean rebuild
rebuild: clean all

.PHONY: all clean rebuild
