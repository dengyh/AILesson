all: main clear

main: BPN.o
	g++ -std=c++11 -o main BPN.o main.cpp

BPN.o: BPN.cpp BPN.h
	g++ -std=c++11 -c BPN.cpp BPN.h

clear:
	rm *.o *.gch