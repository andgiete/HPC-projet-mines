#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

// Global variable used to store the results of every thread 
float results[16];

// Structure to pass data to every thread
struct norm_struct {
    float *U_vect;
    int n;
    int id;
};

// To measure time
double now(){
   // Retourne l'heure actuelle en secondes
   struct timeval t; double f_t;
   gettimeofday(&t, NULL);
   f_t = t.tv_usec; f_t = f_t/1000000.0; f_t +=t.tv_sec;
   return f_t;
}

// Calculate norm using standard method
float norm(void *args){
    struct norm_struct *actual_args = args;
    int n = actual_args->n;
    float *U = actual_args->U_vect;

    float sum = 0;
    for(int i=0; i<n; i++){
        sum += sqrt(fabsf(U[i]));
    }
    return sum;
}

// Calculate norm using vectorial method
float vect_norm(void *args){

    struct norm_struct *actual_args = args;
    int n = actual_args->n;
    float *U = actual_args->U_vect;

    float minus1[8] __attribute__((aligned(32))) = {-1,-1,-1,-1,-1,-1,-1,-1};
    float f[8] __attribute__((aligned(32))) = {0,0,0,0,0,0,0,0};

    float sum = 0;

    // If U is unaligned, we use the function 'mm256_loadu_ps()' to copy unaligned data to register
    __m256 result_v=_mm256_load_ps(&f[0]); 

    for( int i = 0 ; i<n; i=i+8){
        __m256 v=_mm256_load_ps(&U[i]);
        __m256 abs_v=_mm256_sqrt_ps(_mm256_mul_ps(v,v));
        __m256 sqrt_v=_mm256_sqrt_ps(abs_v);

        result_v = _mm256_add_ps(result_v,sqrt_v);
    }
	_mm256_store_ps(&f[0],result_v);

    for(int i=0; i<8; i++){
        sum += f[i];
    }
    return sum;
}

// Identical to function norm but used for multithreading method in normPar 
void *norm_th(void *args){
    struct norm_struct *actual_args = args;
    int n = actual_args->n;
    float *U = actual_args->U_vect;
    int id = actual_args->id;

    float sum = 0;
    for(int i=0; i<n; i++){
        sum += sqrt(fabsf(U[i]));
    }
    results[id]=sum;
    pthread_exit(&sum);
}

// Identical to function vect_norm but used for multithreading method in normPar
void *vect_norm_th(void *args){
    struct norm_struct *actual_args = args;
    int n = actual_args->n;
    float *U = actual_args->U_vect;
    int id = actual_args->id;

    float minus1[8] __attribute__((aligned(32))) = {-1,-1,-1,-1,-1,-1,-1,-1};
    float f[8] __attribute__((aligned(32))) = {0,0,0,0,0,0,0,0};

    float sum = 0;

    // If U is unaligned, we use the function 'mm256_loadu_ps()' to copy unaligned data to register
    __m256 result_v=_mm256_load_ps(&f[0]);

    for( int i = 0 ; i<n; i=i+8){
        __m256 v=_mm256_load_ps(&U[i]);
        __m256 abs_v=_mm256_sqrt_ps(_mm256_mul_ps(v,v));
        __m256 sqrt_v=_mm256_sqrt_ps(abs_v);

        result_v = _mm256_add_ps(result_v,sqrt_v);
    }
	_mm256_store_ps(&f[0],result_v);

    for(int i=0; i<8; i++){
        sum += f[i];
    }

    results[id]=sum;
    pthread_exit(&sum);
}

// Used to divide the task to all threads, pass functions to threads and aggregate all results in one result
float normPar(float *U, int n, int mode, int nb_threads){

    int N = n/nb_threads;
    pthread_t thread[16];
    pthread_attr_t attr;
    int rc;
    int t;
    void *status;
    struct norm_struct norm_args[16];

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for(t=0; t<nb_threads; t++) {
        norm_args[t].U_vect = &U[t*N];
        norm_args[t].n = N;
        norm_args[t].id = t;
        if(mode == 0){
            rc = pthread_create(&thread[t], &attr, norm_th, &norm_args[t]);
        } else {
            rc = pthread_create(&thread[t], &attr, vect_norm_th, &norm_args[t]);
        }
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
            }
        }

    pthread_attr_destroy(&attr);

    float result = 0;
    for(t=0; t<nb_threads; t++) {
        rc = pthread_join(thread[t], &status);
        if (rc) {
            printf("ERROR; return code from pthread_join() is %d\n", rc);
            exit(-1);
            }
        result += results[t];
    }

    return result;
}


int main(int argc, char** argv){

    int nb_threads = 1;
    int n = 1048576;

    // if number of threads is provided, use it else use 1
    if(argc == 2){
        n = atoi(argv[1]);
        if(n > 1048576) {
            n = 1048576;
            printf("Max dimension allowed is 1048576\nUsing %d as dimension\n", n);
        }
        if(n%8 != 0){
            n = n - n%8;
            printf("Dimension should be multiple of 8\nUsing %d as dimension\n", n);
        }
    }
    if(argc>2){
        n = atoi(argv[1]);
        if(n > 1048576) {
            n = 1048576;
            printf("Max dimension allowed is 1048576\nUsing %d as dimension\n", n);
        }
        if(n%8 != 0){
            n = n - n%8;
            printf("Dimension should be multiple of 8\nUsing %d as dimension\n", n);
        }
        nb_threads = atoi(argv[2]);
    }
    printf("Using %d threads\n", nb_threads);

    float U[1048576] __attribute__((aligned(32)));
    // If the size of U is not a multiple of 8:
    // We calculate restof_n = n % 8 and new_n = n - restof_n
    // We do the same calculation with new_n that is necessarily multiple of 8
    // The remaining vector is necessarily of size less than 8
    // We calculate the norm of the remaining vector using standard method
    // Execution time will not differ since standard method is quick for small vectors 

    float std_res;
    float vect_res;
    float th_res;

    // Generating random values of U with some negative numbers 
    for(int i=0; i<n; i++){
        U[i] = (rand() % 10)/10.0;
        if(i % 3 == 0) {
            U[i] = - U[i];
        }
    }

    // Initializing the structure to be passed to norm functions
    struct norm_struct norm_args = { .U_vect=&U, .n=n ,.id=0};

    // Norm calculation using standard method
    double time; 
    time = now();
    std_res = norm(&norm_args);
    time = now() - time;
    printf("Norm using std: %.2f  executed in %.8f seconds \n", std_res, time);

    // Norm calculation using vectorial method
    time = now();
    vect_res = vect_norm(&norm_args);
    time = now() - time;
    printf("Norm using vect: %.2f  executed in %.8f seconds \n", vect_res, time);

    // Norm calculation using multithreaded standard method, mode 0
    time = now();
    th_res = normPar(U, n, 0, nb_threads);
    time = now() - time;
    printf("Norm using multithreaded mode %d with %d threads: %.2f  executed in %.8f seconds \n", 0, nb_threads, th_res, time);

    // Norm calculation using multithreaded vectorial method, mode 1
    time = now();
    th_res = normPar(U, n, 1, nb_threads);
    time = now() - time;
    printf("Norm using multithreaded mode %d with %d threads: %.2f  executed in %.8f seconds \n", 1, nb_threads, th_res, time);


    // The next bloc is used to generate the csv file to plot the graph showing the different execution times per method
    int MAX = 16;
    FILE *fp;
    fp = fopen("./data.csv", "w+");
    int dimension = 32;

    for(int i=0; i<MAX; i++){
        norm_args.n = dimension;
        time = now();
        std_res = norm(&norm_args);
        time = now() - time;
        fprintf(fp, "%s,%d,%f\n","std",dimension,time);

        time = now();
        vect_res = vect_norm(&norm_args);
        time = now() - time;
        fprintf(fp, "%s,%d,%f\n","vect",dimension,time);

        time = now();
        th_res = normPar(U, n, 1, 2);
        time = now() - time;
        fprintf(fp, "%s,%d,%f\n","vect2th",dimension,time);

        time = now();
        th_res = normPar(U, n, 1, 4);
        time = now() - time;
        fprintf(fp, "%s,%d,%f\n","vect4th",dimension,time);

        time = now();
        th_res = normPar(U, n, 1, 8);
        time = now() - time;
        fprintf(fp, "%s,%d,%f\n","vect8th",dimension,time);

        dimension = dimension*2;
    }
    return 0;
    
 }