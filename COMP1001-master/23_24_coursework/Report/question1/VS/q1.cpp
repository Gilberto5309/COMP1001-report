/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP1001 ------------------------------------------------------------------
------------------COMPUTER SYSTEMS MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/


#include <stdio.h>
#include <time.h>
#include <pmmintrin.h>
#include <process.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>
#include <omp.h>

#define M 1024*512
#define ARITHMETIC_OPERATIONS1 3*M
#define TIMES1 1

#define N 8192
#define ARITHMETIC_OPERATIONS2 4*N*N
#define TIMES2 1


//function declaration
void initialize();
void routine1(float alpha, float beta);
void routine2(float alpha, float beta);
void routine1_vec(float alpha, float beta);
void routine2_vec(float alpha, float beta);


__declspec(align(64)) float  y[M], z[M];
__declspec(align(64)) float A[N][N], x[N], w[N];

int main() {

    float alpha = 0.023f, beta = 0.045f;
    double run_time, start_time;
    unsigned int t;

    initialize();

    printf("\nRoutine1:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES1; t++)
        routine1(alpha, beta);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS1) / ((double)run_time / TIMES1));

    printf("\nRoutine2:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES2; t++)
        routine2(alpha, beta);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS2) / ((double)run_time / TIMES2));

    printf("\nRoutine1 vectorised:");
    start_time = omp_get_wtime();

    for (t = 0; t < TIMES1; t++)
        routine1_vec(alpha, beta);

    run_time = omp_get_wtime() - start_time;
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS1) / ((double)run_time / TIMES1));


    printf("\nRoutine2 vectorised:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES2; t++)
        routine2_vec(alpha, beta);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS2) / ((double)run_time / TIMES2));
    return 0;
}

void initialize() {

    unsigned int i, j;

    //initialize routine2 arrays
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = (i % 99) + (j % 14) + 0.013f;
        }

    //initialize routine1 arrays
    for (i = 0; i < N; i++) {
        x[i] = (i % 19) - 0.01f;
        w[i] = (i % 5) - 0.002f;
    }

    //initialize routine1 arrays
    for (i = 0; i < M; i++) {
        z[i] = (i % 9) - 0.08f;
        y[i] = (i % 19) + 0.07f;
    }


}




void routine1(float alpha, float beta) {

    unsigned int i;


    for (i = 0; i < M; i++)
        y[i] = alpha * y[i] + beta * z[i];

}

void routine2(float alpha, float beta) {

    unsigned int i, j;


    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            w[i] = w[i] - beta + alpha * A[i][j] * x[j];


}


void routine1_vec(float alpha, float beta) {

    //create an unsigned integer to iterate through the for loop
    unsigned int i;

    //Define 5 128bit variables that are float types
    __m128 num1, num2, num3, num4, num5;

    //Initialize num4 and num5 as alpha and beta respectively so we can use the values in our for loop
    num4 = _mm_set_ps(alpha, alpha, alpha, alpha);
    num5 = _mm_set_ps(beta, beta, beta, beta);


    //The following loop will be running the equation y[i] = alpha * y[i] + beta * z[i]
    //A for loop where 4 iterations happen per iteration
    for (i = 0; i < M; i += 4) {

        //Load 4 values from the memory address y[i] and store it in the variable num1
        num1 = _mm_loadu_ps(&y[i]);
        //Multiply the values in num1 by the values in num4 (alpha) and store the result in num1
        num1 = _mm_mul_ps(num4, num1);

        //Load 4 values from memory address &z[i] and store the values in num2
        num2 = _mm_loadu_ps(&z[i]);
        //Multiply the values in num3 by the values of num5 (beta) and store them in num2
        num2 = _mm_mul_ps(num5, num2);

        //Add the values in num1 and num2 together and store the results in num3
        num3 = _mm_add_ps(num1, num2);

        //store the values of num3 back into y[i]
        _mm_storeu_ps(&y[i], num3);
    }



}

void routine2_vec(float alpha, float beta) {

    //Create variables i and j to iterate through the for loop
    unsigned int i, j;

    //Define 6 128bit variables that are float types
    __m128 num1, num2, num3, num4, num5, num6;

    //num5 and num6 store the values of alpha and beta so we can use them in our loop
    num5 = _mm_set_ps(alpha, alpha, alpha, alpha);
    num6 = _mm_set_ps(beta, beta, beta, beta);

    //We use nested for loops that are iterated through using i and j that increment by 4 every iteration
    for (i = 0; i < N; i += 4) {
        for (j = 0; j < N; j += 4) {

            //load 4 values from the memory address w[i] and store them in num1
            num1 = _mm_loadu_ps(&w[i]);
            //Sub beta from the value in num1
            num1 = _mm_sub_ps(num1, num6);

            //Load 4 values from the memory address A[i][j] and store them in num 2
            num2 = _mm_loadu_ps(&A[i][j]);
            //load 4 values from memory address x[i] and store them in num3
            num3 = _mm_loadu_ps(&x[j]);
            //multiply the values in num2 by values in num 3 and store them in num2
            num2 = _mm_mul_ps(num2, num3);

            //multiply the values in num2 by alpha and store it back in num2
            num2 = _mm_mul_ps(num2, num5);

            //Add the values in num1 and num2 and store them in num3
            num3 = _mm_add_ps(num1, num2);

            //store the values from num3 into the array w
            _mm_storeu_ps(&w[i], num3);


        }
    }
}

