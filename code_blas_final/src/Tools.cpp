#include "Tools.hpp"

int factorial(int n) {
  if (n == 0)
    return 1;
  else
    return n * factorial(n - 1);
}

void inverseMat(vector<double> &A, vector<double> &B) {

  int N = (int)sqrt(A.size());
  vector<double> Acopy(A);

  B.resize(N);
  for (int n = 0; n < N * N; n++)
    B[n] = 0.;
  for (int n = 0; n < N; n++)
    B[n * N + n] = 1.;

  int INFO;
  vector<int> PIV(N);

  // Solve A*X = B, X stored in B at the end
  dgesv_(&N, &N, &Acopy[0], &N, &PIV[0], &B[0], &N, &INFO);
}

// ======================================================================
// Get coefficients of the low-storage RK method
//   Output: rk4a, rk4b, rk4c
// ======================================================================

void getRungeKuttaCoefficients(vector<double> &rk4a, vector<double> &rk4b, vector<double> &rk4c) {
  rk4a.resize(5);
  rk4b.resize(5);
  rk4c.resize(5);
  rk4a[0] = 0.0;
  rk4a[1] = -567301805773.0 / 1357537059087.0;
  rk4a[2] = -2404267990393.0 / 2016746695238.0;
  rk4a[3] = -3550918686646.0 / 2091501179385.0;
  rk4a[4] = -1275806237668.0 / 842570457699.0;
  rk4b[0] = 1432997174477.0 / 9575080441755.0;
  rk4b[1] = 5161836677717.0 / 13612068292357.0;
  rk4b[2] = 1720146321549.0 / 2090206949498.0;
  rk4b[3] = 3134564353537.0 / 4481467310338.0;
  rk4b[4] = 2277821191437.0 / 14882151754819.0;
  rk4c[0] = 0.0;
  rk4c[1] = 1432997174477.0 / 9575080441755.0;
  rk4c[2] = 2526269341429.0 / 6820363962896.0;
  rk4c[3] = 2006345519317.0 / 3224310063776.0;
  rk4c[4] = 2802321613138.0 / 2924317926251.0;
}