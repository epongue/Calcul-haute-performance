#ifndef NODES_HEADER
#define NODES_HEADER

#include "Tools.hpp"

// Get local coordinates of nodes on the reference element
void getReferenceNodes(int N, vector<double> &_r, vector<double> &_s, vector<double> &_t);

// Get global coordinates of nodes on the physical elements
void getPhysicalNodes(vector<int> &_EToV, vector<double> &_VX, vector<double> &_VY, vector<double> &_VZ, vector<double> &_r, vector<double> &_s, vector<double> &_t, vector<double> &_x, vector<double> &_y, vector<double> &_z);

#endif