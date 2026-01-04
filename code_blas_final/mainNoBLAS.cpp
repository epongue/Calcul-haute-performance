#include "Connectivity.hpp"
#include "Gmsh.hpp"
#include "Nodes.hpp"
#include "Numerics.hpp"
#include "Tools.hpp"
#include <chrono>

int main(int argc, char **argv)
{

  // ======================================================================
  // 1A) Get simulation parameters
  // ======================================================================

  string meshFile;               // Mesh file
  int N;                         // Degree of polynomial bases
  int iOutputGmsh;               // Number of time steps between gmsh output
  double FinalTime;              // Final time of the simulation
  vector<int> _CoefDataBaseNum;  // Database for parameters
  vector<double> _CoefDataBase1; // Database for parameters
  vector<double> _CoefDataBase2; // Database for parameters

  // Read setup file
  ifstream setupfile("setup");
  if (!setupfile.is_open())
  {
    cerr << "ERROR - Setup file not opened.\n";
    exit(1);
  }
  string line;
  while (setupfile)
  {
    setupfile >> line;
    if (line == "$Mesh")
      setupfile >> meshFile;
    if (line == "$PolynomialDegree")
      setupfile >> N;
    if (line == "$OutputStep")
      setupfile >> iOutputGmsh;
    if (line == "$Duration")
      setupfile >> FinalTime;
    if (line == "$Param")
    {
      int PARAM;
      setupfile >> PARAM;
      _CoefDataBaseNum.resize(PARAM);
      _CoefDataBase1.resize(PARAM);
      _CoefDataBase2.resize(PARAM);
      for (int i = 0; i < PARAM; i++)
      {
        setupfile >> _CoefDataBaseNum[i];
        setupfile >> _CoefDataBase1[i];
        setupfile >> _CoefDataBase2[i];
      }
    }
  }

  // ======================================================================
  // 1B) Load mesh
  // ======================================================================

  int K;              // Number of elements in the mesh
  vector<double> _VX; // Coordinate 'x' of the vertices of the mesh [#vert]
  vector<double> _VY; // Coordinate 'y' of the vertices of the mesh [#vert]
  vector<double> _VZ; // Coordinate 'z' of the vertices of the mesh [#vert]
  vector<int> _EMsh;  // List of orginal gmsh numbers of elements [K]
  vector<int> _ETag;  // List of element tag (gmsh: physical tag) [K]
  vector<int> _EToV;  // Element-to-vertex connectivity array [K,NVertTet]

  // Load mesh
  loadMeshGmsh(meshFile, K, _VX, _VY, _VZ, _EMsh, _ETag, _EToV);

  // ======================================================================
  // 1C) Get finite element nodes
  // ======================================================================

  int Np = (N + 1) * (N + 2) * (N + 3) / 6; // Number of nodes per element
  int Nfp = (N + 1) * (N + 2) / 2;          // Number of nodes per face
  vector<double> _r, _s, _t;                // Coordinates of reference nodes [Np]
  vector<double> _x, _y, _z;                // Coordinates of physical nodes [K,Np]

  // Get local coordinates of nodes on the reference element
  getReferenceNodes(N, _r, _s, _t);

  // Get global coordinates of nodes on the physical elements
  getPhysicalNodes(_EToV, _VX, _VY, _VZ, _r, _s, _t, _x, _y, _z);

  // ======================================================================
  // 1D) Build connectivity matrices, elemental matrices & geometric factors
  // ======================================================================

  vector<int> _EToE;  // Element-to-Element [K,NFacesTet]
  vector<int> _EToF;  // Element-to-Face [K,NFacesTet]
  vector<int> _NfToN; // LocalFaceNode-to-LocalNode [NFacesTet,Nfp]
  vector<int> _mapP;  // GlobalFaceNode-to-NeighborGlobalNode [K,Nfp,NFacesTet]

  // Build connectivity matrices
  buildConnectivity(K, Nfp, Np, _r, _s, _t, _x, _y, _z, _EToV, _EToE, _EToF, _NfToN, _mapP);

  vector<double> _Dr;   // r-derivative matrix [Np,Np]
  vector<double> _Ds;   // s-derivative matrix [Np,Np]
  vector<double> _Dt;   // t-derivative matrix [Np,Np]
  vector<double> _LIFT; // Lift matrix [NFacesTet,Nfp,Np]

  // Build elemental matrices
  buildElemMatrices(N, _r, _s, _t, _NfToN, _Dr, _Ds, _Dt, _LIFT);

  vector<double> _rstxyz; // ... for volume terms  [K,9]
  vector<double> _Fscale; // ... for surface terms [K,NFacesTet]
  vector<double> _nx;     // x-component of normal to the face [K,NFacesTet]
  vector<double> _ny;     // y-component of normal to the face [K,NFacesTet]
  vector<double> _nz;     // z-component of normal to the face [K,NFacesTet]

  // Build geometric factors
  buildGeomFactors(_EToV, _VX, _VY, _VZ, _rstxyz, _Fscale, _nx, _ny, _nz);

  // ======================================================================
  // 1E) Build physical coefficient maps
  // ======================================================================

  vector<double> _c(K);
  vector<double> _rho(K);

  for (int k = 0; k < K; k++)
  {
    int i = 0;
    while (i < _CoefDataBaseNum.size())
    {
      if (_CoefDataBaseNum[i] == _ETag[k])
      {
        _c[k] = _CoefDataBase1[i];
        _rho[k] = _CoefDataBase2[i];
        break;
      }
      i++;
    }
    if (i == _CoefDataBaseNum.size())
    {
      cerr << "ERROR - ETag " << _ETag[k] << " not found in parameter database\n";
      exit(1);
    }
  }
  exportParamGmsh("info_c", _EMsh, _c);
  exportParamGmsh("info_rho", _EMsh, _rho);

  // ======================================================================
  // 1F) Build time stepping scheme
  // ======================================================================

  double dt = 1e9;
  for (int k = 0; k < K; k++)
  {
    for (int f = 0; f < NFacesTet; ++f)
    {
      double val = 1. / (_c[k] * (N + 1) * (N + 1) * _Fscale[k * NFacesTet + f]);
      if (val < dt)
        dt = val;
    }
  }
  double CFL = 0.75;                 // CFL
  dt *= CFL;                         // Time step
  int Nsteps = ceil(FinalTime / dt); // Number of global time steps

  vector<double> rk4a, rk4b, rk4c;
  getRungeKuttaCoefficients(rk4a, rk4b, rk4c);

  cout << "INFO - DT = " << dt << " and Nsteps = " << Nsteps << "\n";

  // ======================================================================
  // 1G) Memory storage for fields, RHS and residual at nodes
  // ======================================================================

  int Nfields = 4; // Number of unknown fields

  // Memory storage at each node of each element
  int KNpNfields = K * Np * Nfields;
  vector<double> _valQ(KNpNfields, 0.); // Values of fields
  vector<double> _rhsQ(KNpNfields, 0.); // RHS
  vector<double> _resQ(KNpNfields, 0.); // residual

  // Initialization
  for (int k = 0; k < K; ++k)
  {
    for (int n = 0; n < Np; ++n)
    {
      int kNpn = k * Np + n;
      double x = _x[kNpn];
      double y = _y[kNpn];
      double z = _z[kNpn];
      _valQ[k * Np * Nfields + n * Nfields + 0] = exp(-(x * x + y * y + z * z) / 0.1);
    }
  }

  cout << "INFO - Total number of DOF = " << K * Np * Nfields << "\n";

  // Export initial solution
  if (iOutputGmsh > 0)
    exportSolGmsh(N, _r, _s, _t, _EMsh, 0, 0., _valQ);

  // ======================================================================
  // 2) RUN
  // ======================================================================

  auto start_time = std::chrono::high_resolution_clock::now();
  long long int op_counter = 0;

  // Global time iteration
  int NfpXNFacesTet = Nfp * NFacesTet;

  for (int nGlo = 0; nGlo < Nsteps; ++nGlo)
  {
    double runTime = nGlo * dt; // Time at the beginning of the step

    // Local time iteration
    for (int nLoc = 0; nLoc < 5; ++nLoc)
    {
      double a = rk4a[nLoc];
      double b = rk4b[nLoc];

// ======================== (1) UPDATE RHS
#pragma omp parallel for reduction(+ : op_counter)
      for (int k = 0; k < K; ++k)
      {

        double c = _c[k];
        double rho = _rho[k];

        // ======================== (1.1) UPDATE PENALTY VECTOR

        vector<double> s_p_flux(NfpXNFacesTet);
        vector<double> s_u_flux(NfpXNFacesTet);
        vector<double> s_v_flux(NfpXNFacesTet);
        vector<double> s_w_flux(NfpXNFacesTet);

        // Fetch medium parameters
        // Pré-calculs indépendants de nf

        double inv_rho_c = 1. / (rho * c);
        int kNfacesTetNfp = k * NFacesTet * Nfp;
        int kNpNfields = k * Np * Nfields;

        for (int f = 0; f < NFacesTet; f++)
        {
          int indice_n = k * NFacesTet + f;
          // Fetch normal
          int k2 = _EToE[indice_n]; // int k2, je suppose que le type des éléments de _EToE n'ont pas été modifié depuis son initialisation
          int k2NpNfields = k2 * Np * Nfields;
          double cP = _c[k2];
          double rhoP = _rho[k2];
          double inv_rhoPcP = 1. / (rhoP * cP);
          double common_denom = 1. / (inv_rhoPcP + inv_rho_c);
          double sum_rho_c = rhoP * cP + rho * c;

          double inv_sum_rho_c = 1 / sum_rho_c;
          double nx = _nx[indice_n];
          double ny = _ny[indice_n];
          double nz = _nz[indice_n];
          // Compute penalty terms
          for (int nf = f * Nfp; nf < (f + 1) * Nfp; nf++)
          {

            // Index of node in current element
            int n1 = _NfToN[nf];

            // Index of node in neighbor element
            int n2 = _mapP[nf + kNfacesTetNfp];
            int idxM = kNpNfields + n1 * Nfields;
            // Load values 'minus' corresponding to current element
            double pM = _valQ[idxM + 0];
            double uM = _valQ[idxM + 1];
            double vM = _valQ[idxM + 2];
            double wM = _valQ[idxM + 3];
            double nMdotuM = (nx * uM + ny * vM + nz * wM);

            op_counter += 5;
            if (n2 >= 0)
            { // ... if there is a neighbor element ...

              // Load values 'plus' corresponding to neighbor element
              int idxP = k2NpNfields + n2 * Nfields;
              double pP = _valQ[idxP + 0];
              double uP = _valQ[idxP + 1];
              double vP = _valQ[idxP + 2];
              double wP = _valQ[idxP + 3];
              double nMdotuP = nx * uP + ny * vP + nz * wP; // 5
              // Penalty terms for interface between two elements
              double dp = pP - pM;           // 1 = nb d'opérations flottantes comptées à cette ligne
              double du = nMdotuP - nMdotuM; // 1

              double tmp = common_denom * (du - inv_rhoPcP * dp); // 3

              s_p_flux[nf] = c * tmp;                                   // 1 = nb d'opérations flottantes comptées à cette ligne
              double scale = c * inv_sum_rho_c * (dp - rhoP * cP * du); // 5 ...
              s_u_flux[nf] = nx * scale;                                // 1 ...
              s_v_flux[nf] = ny * scale;                                // 1 ...
              s_w_flux[nf] = nz * scale;                                // 1 ...
              op_counter += 51;
            }
            else
            {

              // Homogeneous Dirichlet on 'p'
              double tmp = -2. * pM * inv_rhoPcP;
              // Homogeneous Dirichlet on 'u'
              // double tmp = 2*nMdotuM;
              // ABC
              // double tmp = nMdotuM - 1./(rhoP*cP) * pM;

              // Penalty terms for boundary of the domain
              double scaled_tmp = c * tmp;

              s_p_flux[nf] = -common_denom * scaled_tmp;
              double scaled = scaled_tmp * inv_sum_rho_c;
              s_u_flux[nf] = nx * scaled;
              s_v_flux[nf] = ny * scaled;
              s_w_flux[nf] = nz * scaled;
              op_counter += 25;
            }
          }
        }

        // ======================== (1.2) COMPUTING VOLUME TERMS

        // Load geometric factors
        int k9 = k * 9;
        double rx = _rstxyz[k9 + 0];
        double ry = _rstxyz[k9 + 1];
        double rz = _rstxyz[k9 + 2];
        double sx = _rstxyz[k9 + 3];
        double sy = _rstxyz[k9 + 4];
        double sz = _rstxyz[k9 + 5];
        double tx = _rstxyz[k9 + 6];
        double ty = _rstxyz[k9 + 7];
        double tz = _rstxyz[k9 + 8];

        // Load fields
        vector<double> s_p(Np);
        vector<double> s_u(Np);
        vector<double> s_v(Np);
        vector<double> s_w(Np);
        for (int n = 0; n < Np; ++n)
        {
          int kXNpXNfieldsPnPNXfields = k * Np * Nfields + n * Nfields;
          s_p[n] = _valQ[kXNpXNfieldsPnPNXfields + 0];
          s_u[n] = _valQ[kXNpXNfieldsPnPNXfields + 1];
          s_v[n] = _valQ[kXNpXNfieldsPnPNXfields + 2];
          s_w[n] = _valQ[kXNpXNfieldsPnPNXfields + 3];
        }

        // Compute mat-vec product for surface term
        for (int n = 0; n < Np; ++n)
        {
          double dpdr = 0, dpds = 0, dpdt = 0;
          double dudr = 0, duds = 0, dudt = 0;
          double dvdr = 0, dvds = 0, dvdt = 0;
          double dwdr = 0, dwds = 0, dwdt = 0;
          for (int m = 0; m < Np; ++m)
          {
            double Dr = _Dr[n + m * Np];
            dpdr += Dr * s_p[m];
            dudr += Dr * s_u[m];
            dvdr += Dr * s_v[m];
            dwdr += Dr * s_w[m];
            double Ds = _Ds[n + m * Np];
            dpds += Ds * s_p[m];
            duds += Ds * s_u[m];
            dvds += Ds * s_v[m];
            dwds += Ds * s_w[m];
            double Dt = _Dt[n + m * Np];
            dpdt += Dt * s_p[m];
            dudt += Dt * s_u[m];
            dvdt += Dt * s_v[m];
            dwdt += Dt * s_w[m];
            op_counter += 24;
          }

          double dpdx = rx * dpdr + sx * dpds + tx * dpdt;
          double dpdy = ry * dpdr + sy * dpds + ty * dpdt;
          double dpdz = rz * dpdr + sz * dpds + tz * dpdt;
          double dudx = rx * dudr + sx * duds + tx * dudt;
          double dvdy = ry * dvdr + sy * dvds + ty * dvdt;
          double dwdz = rz * dwdr + sz * dwds + tz * dwdt;
          double divU = dudx + dvdy + dwdz;

          // Compute RHS (only part corresponding to volume terms)
          double inv_roh = -1. / rho;
          int kXNpXNfieldsPnPNXfields = k * Np * Nfields + n * Nfields;
          _rhsQ[kXNpXNfieldsPnPNXfields + 0] = -c * c * rho * divU;
          _rhsQ[kXNpXNfieldsPnPNXfields + 1] = inv_roh * dpdx;
          _rhsQ[kXNpXNfieldsPnPNXfields + 2] = inv_roh * dpdy;
          _rhsQ[kXNpXNfieldsPnPNXfields + 3] = inv_roh * dpdz;
          op_counter += 41;
        }

        // ======================== (1.3) COMPUTING SURFACE TERMS

        for (int n = 0; n < Np; ++n)
        {
          int kXNpXNfieldsPnPNXfields = k * Np * Nfields + n * Nfields;
          double p_lift_total = 0.0, u_lift_total = 0.0, v_lift_total = 0.0, w_lift_total = 0.0;

          for (int f = 0; f < NFacesTet; f++)
          {
            double p_lift = 0.0, u_lift = 0.0, v_lift = 0.0, w_lift = 0.0;
            for (int m = f * Nfp; m < (f + 1) * Nfp; m++)
            {
              double tmp = _LIFT[n * NFacesTet * Nfp + m];
              p_lift += tmp * s_p_flux[m];
              u_lift += tmp * s_u_flux[m];
              v_lift += tmp * s_v_flux[m];
              w_lift += tmp * s_w_flux[m];
              op_counter += 8;
            }
            double Fscale = _Fscale[k * NFacesTet + f];
            p_lift_total += p_lift * Fscale;
            u_lift_total += u_lift * Fscale;
            v_lift_total += v_lift * Fscale;
            w_lift_total += w_lift * Fscale;
            op_counter += 8;
          }

          // Mise à jour groupée de _rhsQ
          _rhsQ[kXNpXNfieldsPnPNXfields + 0] -= p_lift_total;
          _rhsQ[kXNpXNfieldsPnPNXfields + 1] -= u_lift_total;
          _rhsQ[kXNpXNfieldsPnPNXfields + 2] -= v_lift_total;
          _rhsQ[kXNpXNfieldsPnPNXfields + 3] -= w_lift_total;
          op_counter += 4;
        }
      }

// ======================== (2) UPDATE RESIDUAL + FIELDS
// MODIFICATION DE valQ
#pragma omp parallel for reduction(+ : op_counter)
      for (int k = 0; k < K; ++k)
      {
        int base_k = k * Np * Nfields;
#pragma omp simd
        for (int n = 0; n < Np; ++n)
        {
          int base = base_k + n * Nfields;
          _resQ[base + 0] = a * _resQ[base + 0] + dt * _rhsQ[base + 0];
          _resQ[base + 1] = a * _resQ[base + 1] + dt * _rhsQ[base + 1];
          _resQ[base + 2] = a * _resQ[base + 2] + dt * _rhsQ[base + 2];
          _resQ[base + 3] = a * _resQ[base + 3] + dt * _rhsQ[base + 3];
          _valQ[base + 0] = _valQ[base + 0] + b * _resQ[base + 0];
          _valQ[base + 1] = _valQ[base + 1] + b * _resQ[base + 1];
          _valQ[base + 2] = _valQ[base + 2] + b * _resQ[base + 2];
          _valQ[base + 3] = _valQ[base + 3] + b * _resQ[base + 3];
          op_counter += 20;
        }
      }
    }

    // Export solution
    if ((iOutputGmsh > 0) && ((nGlo + 1) % iOutputGmsh == 0))
      exportSolGmsh(N, _r, _s, _t, _EMsh, nGlo + 1, runTime + dt, _valQ);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;
  std::cout << "Temps total: " << elapsed.count() << " secondes." << std::endl;
  std::cout << "Nb opération flottantes= " << op_counter << std::endl;
  // ======================================================================
  // 3) Post-processing
  // ======================================================================

  // Export final solution
   if (iOutputGmsh > 0)
      exportSolGmsh(N, _r, _s, _t, _EMsh, Nsteps, Nsteps * dt, _valQ);
  exportSolGmsh(N, _r, _s, _t, _EMsh, Nsteps, Nsteps * dt, _valQ);

  return 0;
}
