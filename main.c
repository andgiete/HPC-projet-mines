#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

float results[16];

struct norm_struct {
    float *U_vect;
    int n;
    int id;
};

double now(){
   // Retourne l'heure actuelle en secondes
   struct timeval t; double f_t;
   gettimeofday(&t, NULL);
   f_t = t.tv_usec; f_t = f_t/1000000.0; f_t +=t.tv_sec;
   return f_t;
}

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

float vect_norm(void *args){

    struct norm_struct *actual_args = args;
    int n = actual_args->n;
    float *U = actual_args->U_vect;

    float minus1[8] __attribute__((aligned(32))) = {-1,-1,-1,-1,-1,-1,-1,-1};
    float f[8] __attribute__((aligned(32))) = {0,0,0,0,0,0,0,0};

    float sum = 0;

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

void *vect_norm_th(void *args){
    struct norm_struct *actual_args = args;
    int n = actual_args->n;
    float *U = actual_args->U_vect;
    int id = actual_args->id;

    float minus1[8] __attribute__((aligned(32))) = {-1,-1,-1,-1,-1,-1,-1,-1};
    float f[8] __attribute__((aligned(32))) = {0,0,0,0,0,0,0,0};

    float sum = 0;

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

    if(argc>1){
        nb_threads = atoi(argv[1]);
    }

    float U[1048576] __attribute__((aligned(32)));
    int n = 1048576;

    float std_res;
    float vect_res;
    float th_res;

    for(int i=0; i<n; i++){
        U[i] = (rand() % 10)/10.0;
        if(i % 3 == 0) {
            U[i] = - U[i];
        }
    }

    struct norm_struct norm_args = { .U_vect=&U, .n=n ,.id=0};

    double time; 
    time = now();
    std_res = norm(&norm_args);
    time = now() - time;
    printf("Norm using std: %.2f  executed in %.8f seconds \n", std_res, time);

    time = now();
    vect_res = vect_norm(&norm_args);
    time = now() - time;
    printf("Norm using vect: %.2f  executed in %.8f seconds \n", vect_res, time);

    time = now();
    th_res = normPar(U, n, 0, nb_threads);
    time = now() - time;
    printf("Norm using multithreaded mode %d with %d threads: %.2f  executed in %.8f seconds \n", 0, nb_threads, th_res, time);

    time = now();
    th_res = normPar(U, n, 1, nb_threads);
    time = now() - time;
    printf("Norm using multithreaded mode %d with %d threads: %.2f  executed in %.8f seconds \n", 1, nb_threads, th_res, time);

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