#include "Gmsh.hpp"

// ======================================================================
// Load mesh from gmsh file
//   Input: 'fileName'
//   Output: K, _VX, _VY, _VZ, _EMsh, _ETag, _EToV
// ======================================================================

void loadMeshGmsh(string fileName, int &K, vector<double> &_VX, vector<double> &_VY, vector<double> &_VZ, vector<int> &_EMsh, vector<int> &_ETag, vector<int> &_EToV) {
  cout << "CALL - loadMeshGmsh('" << fileName << "')\n";

  ifstream meshFile(fileName);
  if (!meshFile.is_open()) {
    cerr << "ERROR: Mesh file '" << fileName << "' not opened.\n";
    exit(1);
  }

  int dummy;
  while (meshFile) {
    string line;
    meshFile >> line;

    // read nodes
    if (line == "$Nodes") {
      int allVertices;
      meshFile >> allVertices;
      _VX.resize(allVertices);
      _VY.resize(allVertices);
      _VZ.resize(allVertices);
      for (int v = 0; v < allVertices; ++v) {
        meshFile >> dummy >> _VX[v] >> _VY[v] >> _VZ[v];
      }
    }

    // read elements
    if (line == "$Elements") {
      int allElements;
      meshFile >> allElements;
      _EMsh.resize(allElements);
      _ETag.resize(allElements);
      _EToV.resize(allElements * NVertTet);

      int kLoc = 0;
      for (int k = 0; k < allElements; k++) {
        int eGmshNum, eGmshType, eTag;
        meshFile >> eGmshNum >> eGmshType >> dummy >> eTag >> dummy;
        if (eGmshType == 2) { // 2 for TRI
          meshFile >> dummy >> dummy >> dummy;
        }
        if (eGmshType == 4) {     // 4 for TET
          _EMsh[kLoc] = eGmshNum; // save the gmsh number
          _ETag[kLoc] = eTag;     // save the element group (physical gmsh label)
          for (int v = 0; v < NVertTet; ++v) {
            int vertGmsh;
            meshFile >> vertGmsh; // save the associated vertices
            _EToV[kLoc * NVertTet + v] = vertGmsh - 1; // (gmsh is 1-index, here is 0-index)
          }
          kLoc++;
        }
      }

      K = kLoc;
      _EMsh.resize(K);
      _ETag.resize(K);
      _EToV.resize(K * NVertTet);
    }
  }
  meshFile.close();

  cout << "INFO - Load " << _VX.size() << " vertices and " << _EMsh.size()
       << " tetrahedral elements from GMSH file ('" << fileName << "')\n";
}

// ======================================================================
// Export parameter in gmsh file
//   Input: name, _EMsh, _data
//   Output: -
// ======================================================================

void exportParamGmsh(string name, vector<int> &_EMsh, vector<double> &_data) {

  int K = _EMsh.size();
  string fileName = "output/info_" + name + ".msh";
  string viewName = "info_" + name;

  ofstream posFile;
  posFile.open(fileName);
  posFile << "$MeshFormat\n";
  posFile << "2.1 0 8\n";
  posFile << "$EndMeshFormat\n";
  posFile << "$ElementNodeData\n";
  posFile << 2 << "\n";
  posFile << "\"" << viewName << "\"\n"; // name of the view
  posFile << 0 << "\n";
  posFile << 1 << "\n";
  posFile << 0 << "\n";
  posFile << 3 << "\n";
  posFile << 0 << "\n";
  posFile << 1 << "\n"; // ("numComp")
  posFile << K << "\n"; // total number of elementNodeData in this file
  for (int k = 0; k < K; k++) {
    posFile << _EMsh[k] << " " << NVertTet;
    for (int v = 0; v < NVertTet; v++)
      posFile << " " << _data[k];
    posFile << "\n";
  }
  posFile << "$EndElementNodeData\n";
  posFile.close();
}

// ======================================================================
// Export solution in gmsh file
//   Input: _r, _s, _t, _EMsh, timeStep, time
//   Output: -
// ======================================================================

void exportSolGmsh(int N, vector<double> &_r, vector<double> &_s, vector<double> &_t, vector<int> &_EMsh, int timeStep, double time, vector<double> &data) {

  int K = _EMsh.size();
  int Np = _r.size();
  int Nfields = data.size()/K/Np;

  for (int iField = 0; iField < Nfields; iField++) {
    string fileName = "output/sol_field" + to_string(iField) + "_" + to_string(timeStep) + ".msh";
    string viewName = "sol_field" + to_string(iField);

    ofstream posFile;
    posFile.open(fileName);
    posFile << "$MeshFormat\n";
    posFile << "2.1 0 8\n";
    posFile << "$EndMeshFormat\n";

    // build matrices with powers of monomials & interpolation coefficients
    vector<int> monom(Np * 3);
    vector<double> vdm(Np * Np);
    for (int i = 0, n = 0; i <= N; i++)
      for (int j = 0; j <= N; j++)
        for (int k = 0; k <= N; k++)
          if (i + j + k <= N) {
            monom[3 * n + 0] = i;
            monom[3 * n + 1] = j;
            monom[3 * n + 2] = k;
            n++;
          }
    for (int m = 0; m < Np; m++)
      for (int n = 0; n < Np; n++)
        vdm[n * Np + m] = pow((_r[n] + 1) / 2., monom[3 * m + 0]) * pow((_s[n] + 1) / 2., monom[3 * m + 1]) * pow((_t[n] + 1) / 2., monom[3 * m + 2]);

    vector<double> coeff(Np * Np);
    inverseMat(vdm, coeff);

    // write the interpolation scheme
    posFile << "$InterpolationScheme\n";
    posFile << "\"MyInterpScheme\"\n";
    posFile << 1 << "\n";
    posFile << 5 << " " << 2 << "\n";   // 5 is for TET -- 2 is coded as-is in gmsh
    posFile << Np << " " << Np << "\n"; // size of matrix 'coeff'
    for (int m = 0; m < Np; m++) {
      for (int n = 0; n < Np; n++)
        posFile << coeff[n * Np + m] << " ";
      posFile << "\n";
    }
    posFile << Np << " " << 3 << "\n"; // size of matrix 'monom'
    for (int n = 0; n < Np; n++) {
      for (int d = 0; d < 3; d++)
        posFile << monom[3 * n + d] << " ";
      posFile << "\n";
    }
    posFile << "$EndInterpolationScheme\n";

    // write element node data
    posFile << "$ElementNodeData\n";
    posFile << 2 << "\n";
    posFile << "\"" << viewName << "\"\n"; // name of the view
    posFile << "\"MyInterpScheme\"\n";
    posFile << 1 << "\n";
    posFile << time << "\n";
    posFile << 3 << "\n";
    posFile << timeStep << "\n";
    posFile << 1 << "\n"; // ("numComp")
    posFile << K << "\n"; // total number of elementNodeData in this file
    for (int k = 0; k < K; k++) {
      posFile << _EMsh[k] << " " << Np;
      for (int v = 0; v < Np; v++)
        posFile << " " << data[Nfields * Np * k + Nfields * v + iField];
      posFile << "\n";
    }
    posFile << "$EndElementNodeData\n";

    posFile.close();
  }

  double newMax = -1000.;
  for (int k = 0; k < K; k++)
    for (int v = 0; v < Np; v++)
      for (int iField = 0; iField < Nfields; iField++)
        newMax = fmax(newMax, data[k * Np * Nfields + v * Nfields + iField]);

  cout << "  fields exported (max value: " << newMax << " at time " << time
       << " and iter " << timeStep << ")\n";
}