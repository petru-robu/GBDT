# Compiler settings
CXX = g++
# Added -I$(INC_DIR) so the compiler knows where to find your .hpp files
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -I$(INC_DIR)

# Executable name
TARGET = decision_tree

# Directories
SRC_DIR = src
INC_DIR = inc
OBJ_DIR = obj

# Automatically find all .cpp files in the src directory
SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# Map the source files to object files in the obj directory
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

# Default rule
all: $(TARGET)

# Link the object files to create the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile cpp files into object files
# The '| $(OBJ_DIR)' ensures the obj directory exists before compiling
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create the object directory if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Clean up generated files
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean