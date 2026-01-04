#ifndef DG_HEADER
#define DG_HEADER

#include "Tools.hpp"

// Compute Jacobi polynomial
double jacobi(double x, double alpha, double beta, int p);

// Compute derivative of Jacobi polynomial
double jacobiDer(double x, double alpha, double beta, int p);

// Compute 2D orthonormal basis functions
void getModalBasis2D(int N, vector<double> &r, vector<double> &s, vector<double> &Vout);

// Compute 3D orthonormal basis functions
void getModalBasis3D(int N, vector<double> &r, vector<double> &s, vector<double> &t, vector<double> &Vout);

// Compute gradient of 3D orthonormal basis functions
void getModalBasisGrad3D(int N, vector<double> &r, vector<double> &s, vector<double> &t, vector<double> &Vrout, vector<double> &Vsout, vector<double> &Vtout);

// Build geometric factors
void buildGeomFactors(vector<int> &_EToV, vector<double> &_VX, vector<double> &_VY, vector<double> &_VZ, vector<double> &_rstxyz, vector<double> &_Fscale, vector<double> &_nx, vector<double> &_ny, vector<double> &_nz);

// Build elemental matrices
void buildElemMatrices(int N, vector<double> &_r, vector<double> &_s, vector<double> &_t, vector<int> &_NfToN, vector<double> &_Dr, vector<double> &_Ds, vector<double> &_Dt, vector<double> &_LIFT);

#endif
