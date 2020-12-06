#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>

float norm(float *U, int n){
    float sum = 0;
    for(int i=0; i<n; i++){
        sum += sqrt(fabsf(U[i]));
    }
    return sum;
}

float vect_norm(float *U, int n){

    float f[8]={0,0,0,0,0,0,0,0};
    float minus1[8]={-1,-1,-1,-1,-1,-1,-1,-1};
    float sum = 0;

    __m256 result_v=_mm256_loadu_ps(&f[0]);

    for( int i = 0 ; i<n; i=i+8){
        __m256 v=_mm256_loadu_ps(&U[i]);
        __m256 minus_1=_mm256_loadu_ps(&minus1[0]);
        __m256 minus_v=_mm256_mul_ps(v, minus_1);
        __m256 abs_v=_mm256_max_ps(v, minus_v);
        __m256 sqrt_v=_mm256_sqrt_ps(abs_v);

        result_v = _mm256_add_ps(result_v,sqrt_v);
    }
	_mm256_store_ps(&f[0],result_v);
    for(int i=0; i<8; i++){
        sum += f[i];
    }
    return sum;
}

int main(int argc, char** argv){

    float U[1024];
    int n = 1024;

    float std_res;
    float vect_res;

    for(int i=0; i<n; i++){
        U[i] = rand() % 50;
        if(i % 3 == 0) {
            U[i] = - U[i];
        }
    }
    std_res = norm(U, n);
    printf("Norm using std: %.2f  \n", std_res);
    vect_res = vect_norm(U, n);
    printf("Norm using vect: %.2f \n", vect_res);
    return 0;
    
 }