# HPC-projet-mines

Compile using: gcc -o main -mavx main.c

Execute using: ./main vector_dimension nb_threads

The program will automatically modify your inputs in the following cases:
- if vector_dimension is bigger than Limit
- if vector_dimension is not multiple of 8
