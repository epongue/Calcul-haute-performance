#ifndef CONNECTIVITY_HEADER
#define CONNECTIVITY_HEADER

#include "Tools.hpp"

void buildConnectivity(int K, int Nfp, int Np, vector<double> &_r,
                       vector<double> &_s, vector<double> &_t,
                       vector<double> &_x, vector<double> &_y,
                       vector<double> &_z, vector<int> &_EToV,
                       vector<int> &_EToE, vector<int> &_EToF,
                       vector<int> &_NfToN, vector<int> &_mapP);

#endif
