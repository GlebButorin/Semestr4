# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/glebbutorin/miniscience-4th-term/2025/01_meshes/cpp/lab1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/glebbutorin/miniscience-4th-term/2025/01_meshes/cpp/lab1/build

# Include any dependencies generated for this target.
include CMakeFiles/2.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/2.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/2.dir/flags.make

CMakeFiles/2.dir/2.cpp.o: CMakeFiles/2.dir/flags.make
CMakeFiles/2.dir/2.cpp.o: /home/glebbutorin/miniscience-4th-term/2025/01_meshes/cpp/lab1/2.cpp
CMakeFiles/2.dir/2.cpp.o: CMakeFiles/2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/glebbutorin/miniscience-4th-term/2025/01_meshes/cpp/lab1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/2.dir/2.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/2.dir/2.cpp.o -MF CMakeFiles/2.dir/2.cpp.o.d -o CMakeFiles/2.dir/2.cpp.o -c /home/glebbutorin/miniscience-4th-term/2025/01_meshes/cpp/lab1/2.cpp

CMakeFiles/2.dir/2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/2.dir/2.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/glebbutorin/miniscience-4th-term/2025/01_meshes/cpp/lab1/2.cpp > CMakeFiles/2.dir/2.cpp.i

CMakeFiles/2.dir/2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/2.dir/2.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/glebbutorin/miniscience-4th-term/2025/01_meshes/cpp/lab1/2.cpp -o CMakeFiles/2.dir/2.cpp.s

# Object files for target 2
2_OBJECTS = \
"CMakeFiles/2.dir/2.cpp.o"

# External object files for target 2
2_EXTERNAL_OBJECTS =

2 : CMakeFiles/2.dir/2.cpp.o
2 : CMakeFiles/2.dir/build.make
2 : /usr/lib/x86_64-linux-gnu/libgmsh.so
2 : CMakeFiles/2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/glebbutorin/miniscience-4th-term/2025/01_meshes/cpp/lab1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable 2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/2.dir/build: 2
.PHONY : CMakeFiles/2.dir/build

CMakeFiles/2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/2.dir/clean

CMakeFiles/2.dir/depend:
	cd /home/glebbutorin/miniscience-4th-term/2025/01_meshes/cpp/lab1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/glebbutorin/miniscience-4th-term/2025/01_meshes/cpp/lab1 /home/glebbutorin/miniscience-4th-term/2025/01_meshes/cpp/lab1 /home/glebbutorin/miniscience-4th-term/2025/01_meshes/cpp/lab1/build /home/glebbutorin/miniscience-4th-term/2025/01_meshes/cpp/lab1/build /home/glebbutorin/miniscience-4th-term/2025/01_meshes/cpp/lab1/build/CMakeFiles/2.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/2.dir/depend

