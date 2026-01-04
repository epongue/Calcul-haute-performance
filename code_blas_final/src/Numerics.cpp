#include "Numerics.hpp"

// ======================================================================
// Orthonormal Jacobi polynomial
// ======================================================================

double jacobi(double x, double alpha, double beta, int p) {

  vector<double> PL(p + 1);

  // initial values P_0(x) and P_1(x)
  double gamma0 = pow(2, (alpha + beta + 1)) / (alpha + beta + 1) * factorial(alpha) * factorial(beta) / factorial(alpha + beta);
  PL[0] = 1.0 / sqrt(gamma0);
  if (p == 0)
    return PL[0];

  double gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0;
  PL[1] = ((alpha + beta + 2) * x / 2 + (alpha - beta) / 2) / sqrt(gamma1);
  if (p == 1)
    return PL[1];

  // repeat value in recurrence.
  double aold = 2 / (2 + alpha + beta) * sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3));

  // forward recurrence using the symmetry of the recurrence.
  for (int i = 1; i <= p - 1; ++i) {
    double h1 = 2 * i + alpha + beta;
    double anew = 2 / (h1 + 2) * sqrt((i + 1) * (i + 1 + alpha + beta) * (i + 1 + alpha) * (i + 1 + beta) / (h1 + 1) / (h1 + 3));
    double bnew = -(alpha * alpha - beta * beta) / h1 / (h1 + 2);
    PL[i + 1] = 1. / anew * (-aold * PL[i - 1] + (x - bnew) * PL[i]);
    aold = anew;
  }

  return PL[p];
}

// ======================================================================
// Derivative of orthonormal Jacobi polynomial
// ======================================================================

double jacobiDer(double x, double alpha, double beta, int p) {
  if (p == 0)
    return 0.;
  else
    return sqrt(p * (p + alpha + beta + 1)) * jacobi(x, alpha + 1, beta + 1, p - 1);
}

// ======================================================================
// Compute 2D orthonormal basis functions
//   Input: N, r, s
//   Output: V
// ======================================================================

void getModalBasis2D(int N, vector<double> &r, vector<double> &s, vector<double> &V) {

  int Nfp = r.size();

  // tensor mapping
  vector<double> a(Nfp);
  vector<double> b(Nfp);
  for (int n = 0; n < Nfp; ++n) {
    a[n] = -1;
    if (fabs(s[n] - 1) > 1e-5)
      a[n] = 2 * (1 + r[n]) / (1 - s[n]) - 1;
    b[n] = s[n];
  }

  // initialize matrix
  int sk = 0;
  for (int i = 0; i <= N; ++i) {
    for (int j = 0; j <= N - i; ++j) {
      for (int n = 0; n < Nfp; ++n) {
        double h1 = jacobi(a[n], 0, 0, i);
        double h2 = jacobi(b[n], 2 * i + 1, 0, j);
        V[sk * Nfp + n] = sqrt(2.) * h1 * h2 * pow(1 - b[n], i);
      }
      sk++;
    }
  }
}

// ======================================================================
// Compute 3D orthonormal basis functions
//   Input: N, r, s, t
//   Output: V
// ======================================================================

void getModalBasis3D(int N, vector<double> &r, vector<double> &s, vector<double> &t, vector<double> &V) {

  int Np = r.size();

  // tensor mapping
  vector<double> a(Np);
  vector<double> b(Np);
  vector<double> c(Np);
  for (int n = 0; n < Np; ++n) {
    a[n] = -1;
    b[n] = -1;
    if (fabs(s[n] + t[n]) > 1e-5)
      a[n] = 2 * (1 + r[n]) / (-s[n] - t[n]) - 1;
    if (fabs(t[n] - 1) > 1e-5)
      b[n] = 2 * (1 + s[n]) / (1 - t[n]) - 1;
    c[n] = t[n];
  }

  // initialize matrix
  int sk = 0;
  for (int i = 0; i <= N; ++i) {
    for (int j = 0; j <= N - i; ++j) {
      for (int k = 0; k <= N - i - j; ++k) {
        for (int n = 0; n < Np; ++n) {
          double h1 = jacobi(a[n], 0, 0, i);
          double h2 = jacobi(b[n], 2 * i + 1, 0, j);
          double h3 = jacobi(c[n], 2 * (i + j) + 2, 0, k);
          V[sk * Np + n] = 2 * sqrt(2.) * h1 * h2 * h3 * pow(1 - b[n], i) * pow(1 - c[n], i + j);
        }
        sk++;
      }
    }
  }
}

// ======================================================================
// Compute gradient of 3D orthonormal basis functions
//   Input: N, r, s, t
//   Output: Vr, Vs, Vt
// ======================================================================

void getModalBasisGrad3D(int N, vector<double> &r, vector<double> &s, vector<double> &t, vector<double> &Vr, vector<double> &Vs, vector<double> &Vt) {

  int Np = r.size();

  // tensor mapping
  vector<double> a(Np);
  vector<double> b(Np);
  vector<double> c(Np);
  for (int n = 0; n < Np; ++n) {
    a[n] = -1;
    b[n] = -1;
    if (fabs(s[n] + t[n]) > 1e-5)
      a[n] = 2 * (1 + r[n]) / (-s[n] - t[n]) - 1;
    if (fabs(t[n] - 1) > 1e-5)
      b[n] = 2 * (1 + s[n]) / (1 - t[n]) - 1;
    c[n] = t[n];
  }

  // initialize matrices
  int sk = 0;
  for (int i = 0; i <= N; ++i) {
    for (int j = 0; j <= N - i; ++j) {
      for (int k = 0; k <= N - i - j; ++k) {
        for (int n = 0; n < Np; ++n) {

          double fa = jacobi(a[n], 0, 0, i);
          double dfa = jacobiDer(a[n], 0, 0, i);
          double gb = jacobi(b[n], 2 * i + 1, 0, j);
          double dgb = jacobiDer(b[n], 2 * i + 1, 0, j);
          double hc = jacobi(c[n], 2 * (i + j) + 2, 0, k);
          double dhc = jacobiDer(c[n], 2 * (i + j) + 2, 0, k);

          // r-derivative
          double dmodedr = dfa * gb * hc;
          if (i > 0)
            dmodedr = dmodedr * pow(0.5 * (1 - b[n]), i - 1);
          if (i + j > 0)
            dmodedr = dmodedr * pow(0.5 * (1 - c[n]), i + j - 1);

          // s-derivative
          double dmodeds = 0.5 * (1 + a[n]) * dmodedr;
          double tmp = dgb * pow(0.5 * (1 - b[n]), i);
          if (i > 0)
            tmp = tmp + (-0.5 * i) * gb * pow(0.5 * (1 - b[n]), i - 1);
          if (i + j > 0)
            tmp = tmp * pow(0.5 * (1 - c[n]), i + j - 1);
          tmp = fa * tmp * hc;
          dmodeds += tmp;

          // t-derivative
          double dmodedt = 0.5 * (1 + a[n]) * dmodedr + 0.5 * (1 + b[n]) * tmp;
          tmp = dhc * pow(0.5 * (1 - c[n]), i + j);
          if (i + j > 0)
            tmp = tmp - 0.5 * (i + j) * hc * pow(0.5 * (1 - c[n]), i + j - 1);
          tmp = fa * gb * tmp;
          tmp = tmp * pow(0.5 * (1 - b[n]), i);
          dmodedt += tmp;

          // normalize
          Vr[sk * Np + n] = pow(2, 2 * i + j + 1.5) * dmodedr;
          Vs[sk * Np + n] = pow(2, 2 * i + j + 1.5) * dmodeds;
          Vt[sk * Np + n] = pow(2, 2 * i + j + 1.5) * dmodedt;
        }
        sk++;
      }
    }
  }
}

// ======================================================================
// Build geometric factors (volume and surface)
//   Input: K, NFacesTet, NVertTet, _EToV, _VX, _VY, _VZ
//   Output: _rstxyz, _Fscale, _nx, _ny, _nz
// ======================================================================

void buildGeomFactors(vector<int> &_EToV, vector<double> &_VX, vector<double> &_VY, vector<double> &_VZ, vector<double> &_rstxyz, vector<double> &_Fscale, vector<double> &_nx, vector<double> &_ny, vector<double> &_nz) {
  cout << "CALL - buildGeomFactors()\n";

  int K = _EToV.size() / NVertTet;

  _rstxyz.resize(K * 9);         // ... for volume terms  [K,9]
  _Fscale.resize(K * NFacesTet); // ... for surface terms [K,NFacesTet]
  _nx.resize(K * NFacesTet);     // x-component of normal to the face [K,NFacesTet]
  _ny.resize(K * NFacesTet);     // y-component of normal to the face [K,NFacesTet]
  _nz.resize(K * NFacesTet);     // z-component of normal to the face [K,NFacesTet]

  for (int k = 0; k < K; ++k) {

    int vGlo1 = _EToV[k * NVertTet + 0];
    int vGlo2 = _EToV[k * NVertTet + 1];
    int vGlo3 = _EToV[k * NVertTet + 2];
    int vGlo4 = _EToV[k * NVertTet + 3];
    double x1 = _VX[vGlo1], y1 = _VY[vGlo1], z1 = _VZ[vGlo1];
    double x2 = _VX[vGlo2], y2 = _VY[vGlo2], z2 = _VZ[vGlo2];
    double x3 = _VX[vGlo3], y3 = _VY[vGlo3], z3 = _VZ[vGlo3];
    double x4 = _VX[vGlo4], y4 = _VY[vGlo4], z4 = _VZ[vGlo4];

    double xrk = 0.5 * (x2 - x1), yrk = 0.5 * (y2 - y1), zrk = 0.5 * (z2 - z1);
    double xsk = 0.5 * (x3 - x1), ysk = 0.5 * (y3 - y1), zsk = 0.5 * (z3 - z1);
    double xtk = 0.5 * (x4 - x1), ytk = 0.5 * (y4 - y1), ztk = 0.5 * (z4 - z1);

    double Jk = xrk * (ysk * ztk - zsk * ytk) - yrk * (xsk * ztk - zsk * xtk) + zrk * (xsk * ytk - ysk * xtk);

    double rx = (ysk * ztk - zsk * ytk) / Jk;
    double ry = -(xsk * ztk - zsk * xtk) / Jk;
    double rz = (xsk * ytk - ysk * xtk) / Jk;
    double sx = -(yrk * ztk - zrk * ytk) / Jk;
    double sy = (xrk * ztk - zrk * xtk) / Jk;
    double sz = -(xrk * ytk - yrk * xtk) / Jk;
    double tx = (yrk * zsk - zrk * ysk) / Jk;
    double ty = -(xrk * zsk - zrk * xsk) / Jk;
    double tz = (xrk * ysk - yrk * xsk) / Jk;

    _rstxyz[k * 9 + 0] = rx;
    _rstxyz[k * 9 + 1] = ry;
    _rstxyz[k * 9 + 2] = rz;
    _rstxyz[k * 9 + 3] = sx;
    _rstxyz[k * 9 + 4] = sy;
    _rstxyz[k * 9 + 5] = sz;
    _rstxyz[k * 9 + 6] = tx;
    _rstxyz[k * 9 + 7] = ty;
    _rstxyz[k * 9 + 8] = tz;

    double nxLoc[NFacesTet] = {-tx, -sx, rx + sx + tx, -rx};
    double nyLoc[NFacesTet] = {-ty, -sy, ry + sy + ty, -ry};
    double nzLoc[NFacesTet] = {-tz, -sz, rz + sz + tz, -rz};

    // normalise and store
    for (int f = 0; f < NFacesTet; ++f) {
      double sJ =
          sqrt(nxLoc[f] * nxLoc[f] + nyLoc[f] * nyLoc[f] + nzLoc[f] * nzLoc[f]);
      _Fscale[k * NFacesTet + f] = sJ;
      _nx[k * NFacesTet + f] = nxLoc[f] / sJ;
      _ny[k * NFacesTet + f] = nyLoc[f] / sJ;
      _nz[k * NFacesTet + f] = nzLoc[f] / sJ;
    }
  }
}

// ======================================================================
// Build elemental matrices (volume and surface)
//   Input: NFacesTet, N, Np, Nfp, _r, _s, _t, _NfToN
//   Output: _Drst, _LIFT
// ======================================================================

void buildElemMatrices(int N, vector<double> &_r, vector<double> &_s, vector<double> &_t, vector<int> &_NfToN, vector<double> &_Dr, vector<double> &_Ds, vector<double> &_Dt, vector<double> &_LIFT) {
  cout << "CALL - buildElemMatrices()\n";

  int Np = _r.size();
  int Nfp = _NfToN.size() / NFacesTet;

  // Differentiation matrices

  vector<double> V(Np * Np);
  vector<double> Vinv(Np * Np);
  vector<double> Vr(Np * Np);
  vector<double> Vs(Np * Np);
  vector<double> Vt(Np * Np);
  _Dr.resize(Np * Np); // r-derivative matrix
  _Ds.resize(Np * Np); // s-derivative matrix
  _Dt.resize(Np * Np); // t-derivative matrix

  getModalBasis3D(N, _r, _s, _t, V);
  inverseMat(V, Vinv);
  getModalBasisGrad3D(N, _r, _s, _t, Vr, Vs, Vt);

  for (int m = 0; m < Np; ++m) {
    for (int n = 0; n < Np; ++n) {
      _Dr[n * Np + m] = 0.;
      _Ds[n * Np + m] = 0.;
      _Dt[n * Np + m] = 0.;
      for (int k = 0; k < Np; ++k) {
        _Dr[n * Np + m] += Vr[k * Np + m] * Vinv[n * Np + k];
        _Ds[n * Np + m] += Vs[k * Np + m] * Vinv[n * Np + k];
        _Dt[n * Np + m] += Vt[k * Np + m] * Vinv[n * Np + k];
      }
    }
  }

  // Lift matrix

  vector<double> _Emat(NFacesTet * Nfp * Np);
  _LIFT.resize(NFacesTet * Nfp * Np);

  for (int i = 0; i < NFacesTet * Nfp; ++i)
    for (int j = 0; j < Np; ++j)
      _Emat[Np * i + j] = 0.;

  for (int f = 0; f < NFacesTet; ++f) {

    vector<double> faceR(Nfp);
    vector<double> faceS(Nfp);
    vector<double> VFace(Nfp * Nfp);
    vector<double> massFaceInv(Nfp * Nfp);
    vector<double> massFace(Nfp * Nfp);

    for (int nf = 0; nf < Nfp; ++nf) {
      int n = _NfToN[f * Nfp + nf];
      switch (f) {
      case 0:
        faceR[nf] = _r[n];
        faceS[nf] = _s[n];
        break;
      case 1:
        faceR[nf] = _r[n];
        faceS[nf] = _t[n];
        break;
      case 2:
        faceR[nf] = _s[n];
        faceS[nf] = _t[n];
        break;
      case 3:
        faceR[nf] = _s[n];
        faceS[nf] = _t[n];
        break;
      }
    }

    getModalBasis2D(N, faceR, faceS, VFace);

    for (int i = 0; i < Nfp; i++) {
      for (int j = 0; j < Nfp; j++) {
        massFaceInv[i * Nfp + j] = 0.;
        for (int k = 0; k < Nfp; k++)
          massFaceInv[i * Nfp + j] += VFace[k * Nfp + i] * VFace[k * Nfp + j];
      }
    }

    inverseMat(massFaceInv, massFace);

    for (int nf = 0; nf < Nfp; ++nf) {
      for (int mf = 0; mf < Nfp; ++mf) {
        int idr = _NfToN[f * Nfp + nf];
        int idc = mf + f * Nfp;
        _Emat[idc * Np + idr] += massFace[mf * Nfp + nf];
      }
    }
  }

  for (int m = 0; m < Nfp * NFacesTet; m++) {
    vector<double> tmp(Np);
    for (int n = 0; n < Np; n++) {
      tmp[n] = 0.;
      for (int k = 0; k < Np; k++)
        tmp[n] += _Emat[m * Np + k] * V[n * Np + k];
    }
    for (int n = 0; n < Np; n++) {
      _LIFT[n * NFacesTet * Nfp + m] = 0.;
      for (int l = 0; l < Np; l++)
        _LIFT[n * NFacesTet * Nfp + m] += tmp[l] * V[l * Np + n];
    }
  }
}
