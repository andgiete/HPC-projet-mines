#include <math.h>
#include <stdlib.h>
#include <stdio.h>

float norm( float *U, int n){
    float sum = 0;
    for(int i=0; i<n; i++){
        sum += sqrt(fabsf(U[i]));
    }
    return sum;
}

int main(int argc, char** argv){

    float my_norm = 0;
    float U[10];
    int n = 0;

    printf("Enter vector dimension: ");
    scanf("%d",&n);

    for(int i=0; i<n; i++){
        printf("Enter element %d: ", i+1);
        scanf("%f", &U[i]);
    }

    my_norm = norm(U, n);
    printf("Norm = %.2f", my_norm);
    return 0;
 }