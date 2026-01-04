#ifndef GMSH_HEADER
#define GMSH_HEADER

#include "Tools.hpp"

// Load mesh from gmsh file
void loadMeshGmsh(string fileName, int &K, vector<double> &_VX, vector<double> &_VY, vector<double> &_VZ, vector<int> &_EMsh, vector<int> &_ETag, vector<int> &_EToV);

// Export parameter in gmsh file
void exportParamGmsh(string name, vector<int> &_EMsh, vector<double> &_data);

// Export solution in gmsh file
void exportSolGmsh(int N, vector<double> &_r, vector<double> &_s, vector<double> &_t, vector<int> &_EMsh, int timeStep, double time, vector<double> &_data);

#endif
