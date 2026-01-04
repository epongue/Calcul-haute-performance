#---[ Compiler & Flags]—————————————————

compiler = icpc
flags = -O3 -mkl -qopenmp 
links = 

#---[ Makefile ]----------------------------------

headers = $(wildcard src/*.hpp)
sources = $(wildcard src/*.cpp)
objects = $(subst src/,obj/,$(sources:.cpp=.o))
paths += -I./src

executables: main

main: $(objects) $(headers) main.cpp
	$(compiler) -o dg $(flags) $(objects) $(paths) main.cpp $(links)

noblas: $(objects) $(headers) mainNoBLAS.cpp
	$(compiler) -o dgNoBLAS $(flags) $(objects) $(paths) mainNoBLAS.cpp $(links)

blas: $(objects) $(headers) mainBLAS.cpp
	$(compiler) -o dgBLAS  $(flags) $(objects) $(paths) mainBLAS.cpp $(links)

obj/%.o: src/%.cpp $(wildcard $(subst src/, src/,$(<:.cpp=.hpp))) $(wildcard $(subst src/, src/,$(<)))
	$(compiler) -o $@ $(flags) -c $(paths) $<

clean:
	rm -f -r dg obj/*
