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
CMAKE_SOURCE_DIR = /home/glebbutorin/Semestr4/lab3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/glebbutorin/Semestr4/lab3/build

# Include any dependencies generated for this target.
include CMakeFiles/navier-stokes.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/navier-stokes.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/navier-stokes.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/navier-stokes.dir/flags.make

CMakeFiles/navier-stokes.dir/main.cpp.o: CMakeFiles/navier-stokes.dir/flags.make
CMakeFiles/navier-stokes.dir/main.cpp.o: /home/glebbutorin/Semestr4/lab3/main.cpp
CMakeFiles/navier-stokes.dir/main.cpp.o: CMakeFiles/navier-stokes.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/glebbutorin/Semestr4/lab3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/navier-stokes.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/navier-stokes.dir/main.cpp.o -MF CMakeFiles/navier-stokes.dir/main.cpp.o.d -o CMakeFiles/navier-stokes.dir/main.cpp.o -c /home/glebbutorin/Semestr4/lab3/main.cpp

CMakeFiles/navier-stokes.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/navier-stokes.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/glebbutorin/Semestr4/lab3/main.cpp > CMakeFiles/navier-stokes.dir/main.cpp.i

CMakeFiles/navier-stokes.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/navier-stokes.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/glebbutorin/Semestr4/lab3/main.cpp -o CMakeFiles/navier-stokes.dir/main.cpp.s

# Object files for target navier-stokes
navier__stokes_OBJECTS = \
"CMakeFiles/navier-stokes.dir/main.cpp.o"

# External object files for target navier-stokes
navier__stokes_EXTERNAL_OBJECTS =

navier-stokes: CMakeFiles/navier-stokes.dir/main.cpp.o
navier-stokes: CMakeFiles/navier-stokes.dir/build.make
navier-stokes: /usr/lib/x86_64-linux-gnu/libdolfin.so.2019.2.0.64.dev0
navier-stokes: /usr/lib/x86_64-linux-gnu/libboost_timer.so
navier-stokes: /usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5.so
navier-stokes: /usr/lib/x86_64-linux-gnu/libcrypto.so
navier-stokes: /usr/lib/x86_64-linux-gnu/libcurl.so
navier-stokes: /usr/lib/x86_64-linux-gnu/libsz.so
navier-stokes: /usr/lib/x86_64-linux-gnu/libz.so
navier-stokes: /usr/lib/x86_64-linux-gnu/libdl.a
navier-stokes: /usr/lib/x86_64-linux-gnu/libm.so
navier-stokes: /usr/lib/slepcdir/slepc3.19/x86_64-linux-gnu-real/lib/libslepc_real.so
navier-stokes: /usr/lib/petscdir/petsc3.19/x86_64-linux-gnu-real/lib/libpetsc_real.so
navier-stokes: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
navier-stokes: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
navier-stokes: CMakeFiles/navier-stokes.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/glebbutorin/Semestr4/lab3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable navier-stokes"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/navier-stokes.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/navier-stokes.dir/build: navier-stokes
.PHONY : CMakeFiles/navier-stokes.dir/build

CMakeFiles/navier-stokes.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/navier-stokes.dir/cmake_clean.cmake
.PHONY : CMakeFiles/navier-stokes.dir/clean

CMakeFiles/navier-stokes.dir/depend:
	cd /home/glebbutorin/Semestr4/lab3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/glebbutorin/Semestr4/lab3 /home/glebbutorin/Semestr4/lab3 /home/glebbutorin/Semestr4/lab3/build /home/glebbutorin/Semestr4/lab3/build /home/glebbutorin/Semestr4/lab3/build/CMakeFiles/navier-stokes.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/navier-stokes.dir/depend

