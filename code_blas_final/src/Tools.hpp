#ifndef TOOLS_HEADER
#define TOOLS_HEADER

#include <cmath>
#include <fstream>
#include <iostream>
#include <mkl.h>
#include <string>
#include <vector>

using namespace std;

#define NFacesLin 2
#define NFacesTri 3
#define NFacesTet 4
#define NFaceVertLin 1
#define NFaceVertTri 2
#define NFaceVertTet 3
#define NVertLin 2
#define NVertTri 3
#define NVertTet 4

// Compute factorial of number 'n'
int factorial(int n);

// Compute inverseMat of n-matrix A (store the inverseMat in B)
void inverseMat(vector<double> &A, vector<double> &B);

// Get coefficients of the low-storage RK method
void getRungeKuttaCoefficients(vector<double> &rk4a, vector<double> &rk4b, vector<double> &rk4c);

#endif
