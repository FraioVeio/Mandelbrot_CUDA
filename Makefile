all: compile run

compile:
	echo "\n\033[1;93m█ Compiling CUDA...\033[0m\n" &&\
	mkdir -p build &&\
	cd build &&\
	cmake .. &&\
	make
run:
	echo "\n\033[1;93m█ RUN...\033[0m\n\n" &&\
	build/Mandelbrot
