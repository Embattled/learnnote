CPP = g++
COMPILE = -c -Wall
LINK = -Wall
OBJS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))

task: $(OBJS)
	$(CPP) $(LINK) $^ -o $@
$(OBJS):%.o:%.cpp
	$(CPP) $(COMPILE) $< -o $@

.PHONY=clean
clean:
	rm *.o task