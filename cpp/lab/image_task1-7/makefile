CPP = g++
COMPILE = -c -Wall
LINK = -Wall
# OBJS = mat.o task1-7.o
OBJS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))

task: $(OBJS)
	$(CPP) $(LINK) $^ -o $@
$(OBJS):%.o:%.cpp
	$(CPP) $(COMPILE) $< -o $@

.PHONY=clean cleanimg
clean:
	rm -rf *.o task 
cleanimg:
	rm -rf *.pgm *.ppm *.pbm