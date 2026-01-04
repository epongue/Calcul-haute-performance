#include "Connectivity.hpp"

int compareEntries(const void *a, const void *b) {
  const int *aI = (int *)a;
  const int *bI = (int *)b;

  int a1 = aI[2], a2 = aI[3], a3 = aI[4];
  int b1 = bI[2], b2 = bI[3], b3 = bI[4];

  if (b1 > a1)
    return -1;
  if (a1 > b1)
    return 1;

  if (b2 > a2)
    return -1;
  if (a2 > b2)
    return 1;

  if (b3 > a3)
    return -1;
  if (a3 > b3)
    return 1;

  return 0;
}

// ======================================================================
// Build connectivity matrices
//   Input: _r, _s, _t, _x, _y, _z, _EToV
//   Output: _EToE, _EToF, _NfToN, _mapP
// ======================================================================

void buildConnectivity(int K, int Nfp, int Np, vector<double> &_r, vector<double> &_s, vector<double> &_t, vector<double> &_x, vector<double> &_y, vector<double> &_z, vector<int> &_EToV, vector<int> &_EToE, vector<int> &_EToF, vector<int> &_NfToN, vector<int> &_mapP) {

  cout << "CALL - buildConnectivity()\n";

  // Build a face-to-vertex connectivity array "FToV"
  vector<int> FToV((NFacesTet * K) * (NFaceVertTet + 2), 0);

  int VfToVe[4][3] = {{0, 1, 2}, {0, 1, 3}, {1, 2, 3}, {0, 2, 3}};

  int cnt = 0;
  for (int k = 0; k < K; k++) {
    for (int f = 0; f < NFacesTet; f++) {

      // raw sort of vertex numbers on this face
      vector<int> ns(NFaceVertTet);
      for (int i = 0; i < NFaceVertTet; ++i) {
        ns[i] = _EToV[k * NVertTet + VfToVe[f][i]];
      }
      for (int i = 0; i < NFaceVertTet; ++i) {
        for (int j = i + 1; j < NFaceVertTet; ++j) {
          if (ns[j] < ns[i]) {
            int tmp = ns[i];
            ns[i] = ns[j];
            ns[j] = tmp;
          }
        }
      }

      FToV[cnt * (NFaceVertTet + 2) + 0] = k; // element number
      FToV[cnt * (NFaceVertTet + 2) + 1] = f; // face number
      for (int i = 0; i < NFaceVertTet; ++i) {
        FToV[cnt * (NFaceVertTet + 2) + 2 + i] = ns[i]; // vertex numbers
      }
      ++cnt;
    }
  }

  // sort by 3rd row (forgot column major convention)
  qsort(&FToV[0], (NFacesTet * K), (NFaceVertTet + 2) * sizeof(int), compareEntries);

  // build '_EToE' and '_EToF' connectivity arrays (1-indexed)
  _EToE.resize(K * NFacesTet);
  _EToF.resize(K * NFacesTet);
  for (int i = 0; i < K * NFacesTet; i++) {
    _EToE[i] = -1;
    _EToF[i] = -1;
  }

  // find neighbors
  for (cnt = 0; cnt < K * NFacesTet - 1; ++cnt) {
    int neighbor = 1;
    for (int vert = 0; vert < NFaceVertTet; vert++) {
      if (FToV[cnt * (NFaceVertTet + 2) + 2 + vert]
          != FToV[(cnt + 1) * (NFaceVertTet + 2) + 2 + vert])
        neighbor = 0;
    }
    if (neighbor == 1) {
      int k1 = FToV[cnt * (NFaceVertTet + 2) + 0];
      int f1 = FToV[cnt * (NFaceVertTet + 2) + 1];
      int k2 = FToV[(cnt + 1) * (NFaceVertTet + 2) + 0];
      int f2 = FToV[(cnt + 1) * (NFaceVertTet + 2) + 1];
      _EToE[k1 * NFacesTet + f1] = k2;
      _EToE[k2 * NFacesTet + f2] = k1;
      _EToF[k1 * NFacesTet + f1] = f2;
      _EToF[k2 * NFacesTet + f2] = f1;
    }
  }

  for (int k = 0; k < K; ++k) {
    for (int f = 0; f < NFacesTet; ++f) {
      if (_EToE[k * NFacesTet + f] == -1) {
        _EToE[k * NFacesTet + f] = k;
        _EToF[k * NFacesTet + f] = f;
      }
    }
  }

  // Build connectivity matrix "LocalFaceNode-to-LocalNode"
  _NfToN.resize(NFacesTet * Nfp);

  double TOL = 1e-4;
  for (int n = 0, cnt1 = 0, cnt2 = 0, cnt3 = 0, cnt4 = 0; n < Np; ++n) {
    if (fabs(1 + _t[n]) < TOL)
      _NfToN[0 * Nfp + cnt1++] = n;
    if (fabs(1 + _s[n]) < TOL)
      _NfToN[1 * Nfp + cnt2++] = n;
    if (fabs(1 + _r[n] + _s[n] + _t[n]) < TOL)
      _NfToN[2 * Nfp + cnt3++] = n;
    if (fabs(1 + _r[n]) < TOL)
      _NfToN[3 * Nfp + cnt4++] = n;
  }

  // Build connectivity matrix "GlobalFaceNode-to-NeighborGlobalNode"
  _mapP.resize(K * Nfp * NFacesTet);

  TOL = 1.e-6;
  for (int k1 = 0; k1 < K; ++k1) {
    for (int f1 = 0; f1 < NFacesTet; ++f1) {
      for (int nf1 = 0; nf1 < Nfp; ++nf1) {

        // local index of face node 'n1' (on face 'f1')
        // in the list of nodes of the ref elem
        int n1 = _NfToN[f1 * Nfp + nf1];

        // global index of face node 'n1' (on face 'f1' of local elem 'k1')
        // in the list of all face nodes of the process
        int n1GlobIndex = nf1 + f1 * Nfp + k1 * NFacesTet * Nfp;
        _mapP[n1GlobIndex] = -1;

        // global index of neighbor element
        // and local index of neighbor face
        int k2 = _EToE[k1 * NFacesTet + f1];
        int f2 = _EToF[k1 * NFacesTet + f1];

        if (k2 != k1) { // if there is a neighbor
          for (int nf2 = 0; nf2 < Nfp; ++nf2) {

            // local index of the face node 'nf2' on face 'f2'
            // in the list of nodes of the ref elem
            int n2 = _NfToN[f2 * Nfp + nf2];

            // distance based on global coordinates of face node on both sides
            double dist = sqrt(pow(_x[k1 * Np + n1] - _x[k2 * Np + n2], 2)
                               + pow(_y[k1 * Np + n1] - _y[k2 * Np + n2], 2)
                               + pow(_z[k1 * Np + n1] - _z[k2 * Np + n2], 2));

            // possible connection
            if (dist < TOL) {

              // global index of face node 'n2' (on face 'f2' of neighbor elem
              // 'k2') in the list of all face nodes of the process [1-indexing]
              _mapP[n1GlobIndex] = n2;
            }
          }
          if (_mapP[n1GlobIndex] < 0) {
            cerr << "There is a problem ...\n";
            exit(1);
          }
        }
      }
    }
  }
}