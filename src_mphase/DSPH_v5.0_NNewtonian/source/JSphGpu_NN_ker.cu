//HEAD_DSPH
/*
<DUALSPHYSICS>  Copyright (c) 2019 by Dr Jose M. Dominguez et al. (see http://dual.sphysics.org/index.php/developers/).

EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

This file is part of DualSPHysics.

DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.

DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>.
*/

/// \file JSphGpu_ker.cu \brief Implements functions and CUDA kernels for the Particle Interaction and System Update.

#include "JSphGpu_NN_ker.h"
//#include <cfloat>
//#include <math_constants.h>

#define MAXNUMBERPHASE 10

__constant__ StPhaseCte PHASECTE[MAXNUMBERPHASE];
__constant__ StPhaseArray PHASEARRAY[MAXNUMBERPHASE];
//__constant__ StPhaseDruckerPrager PHASEDRUCKERPRAGER[MAXNUMBERPHASE];


namespace cusphNN {
#include "FunctionsBasic_iker.h" //orig
#include "FunctionsMath_iker.h" //orig
#include "FunSphKernel_iker.h"
#include "FunSphEos_iker.h"
#undef _JCellSearch_iker_
#include "JCellSearch_iker.h"


//==============================================================================
/// Stores constants for the GPU interaction.
/// Graba constantes para la interaccion a la GPU.
//==============================================================================
void CteInteractionUp_NN(unsigned phasecount,const StPhaseCte *phasecte,const StPhaseArray *phasearray) {
  cudaMemcpyToSymbol(PHASECTE,phasecte,sizeof(StPhaseCte)*phasecount);
  cudaMemcpyToSymbol(PHASEARRAY,phasearray,sizeof(StPhaseArray)*phasecount);
}
void CteInteractionUp_NN(unsigned phasecount, const StPhaseDruckerPrager *phaseDruckerPrager){
  cudaMemcpyToSymbol(PHASEDRUCKERPRAGER,phaseDruckerPrager,sizeof(StPhaseDruckerPrager)*phasecount);
}//DEBUG

//------------------------------------------------------------------------------
/// Doubles the position of the indicated particle using a displacement.
/// Duplicate particles are considered valid and are always within
/// the domain.
/// This kernel applies to single-GPU and multi-GPU because the calculations are made
/// from domposmin.
/// It controls the cell coordinates not exceed the maximum.
///
/// Duplica la posicion de la particula indicada aplicandole un desplazamiento.
/// Las particulas duplicadas se considera que siempre son validas y estan dentro
/// del dominio.
/// Este kernel vale para single-gpu y multi-gpu porque los calculos se hacen 
/// a partir de domposmin.
/// Se controla que las coordendas de celda no sobrepasen el maximo.
//------------------------------------------------------------------------------
__device__ void KerPeriodicDuplicatePos(unsigned pnew,unsigned pcopy
  ,bool inverse,double dx,double dy,double dz,uint3 cellmax
  ,double2 *posxy,double *posz,unsigned *dcell)
{
  //-Obtains position of the particle to be duplicated.
  //-Obtiene pos de particula a duplicar.
  double2 rxy=posxy[pcopy];
  double rz=posz[pcopy];
  //-Applies displacement.
  rxy.x+=(inverse ? -dx : dx);
  rxy.y+=(inverse ? -dy : dy);
  rz+=(inverse ? -dz : dz);
  //-Computes cell coordinates within the domain.
  //-Calcula coordendas de celda dentro de dominio.
  unsigned cx=unsigned((rxy.x-CTE.domposminx)/CTE.scell);
  unsigned cy=unsigned((rxy.y-CTE.domposminy)/CTE.scell);
  unsigned cz=unsigned((rz-CTE.domposminz)/CTE.scell);
  //-Adjust cell coordinates if they exceed the maximum.
  //-Ajusta las coordendas de celda si sobrepasan el maximo.
  cx=(cx<=cellmax.x ? cx : cellmax.x);
  cy=(cy<=cellmax.y ? cy : cellmax.y);
  cz=(cz<=cellmax.z ? cz : cellmax.z);
  //-Stores position and cell of the new particles.
  //-Graba posicion y celda de nuevas particulas.
  posxy[pnew]=rxy;
  posz[pnew]=rz;
  dcell[pnew]=PC__Cell(CTE.cellcode,cx,cy,cz);
}
//------------------------------------------------------------------------------
/// Creates periodic particles from a list of particles to duplicate for non-Newtonian models.
/// It is assumed that all particles are valid.
/// This kernel applies to single-GPU and multi-GPU because it uses domposmin.
///
/// Crea particulas periodicas a partir de una lista con las particulas a duplicar.
/// Se presupone que todas las particulas son validas.
/// Este kernel vale para single-gpu y multi-gpu porque usa domposmin. 
//------------------------------------------------------------------------------
template<bool varspre> __global__ void KerPeriodicDuplicateSymplectic_NN(unsigned n,unsigned pini
  ,uint3 cellmax,double3 perinc,const unsigned *listp,unsigned *idp,typecode *code,unsigned *dcell
  ,double2 *posxy,double *posz,float4 *velrhop,float *auxnn,double2 *posxypre,double *poszpre,float4 *velrhoppre)
{
  const unsigned p=blockIdx.x*blockDim.x+threadIdx.x; //-Number of particle.
  if(p<n) {
    const unsigned pnew=p+pini;
    const unsigned rp=listp[p];
    const unsigned pcopy=(rp&0x7FFFFFFF);
    //-Adjusts cell position of the new particles.
    //-Ajusta posicion y celda de nueva particula.
    KerPeriodicDuplicatePos(pnew,pcopy,(rp>=0x80000000),perinc.x,perinc.y,perinc.z,cellmax,posxy,posz,dcell);
    //-Copies the remaining data.
    //-Copia el resto de datos.
    idp[pnew]=idp[pcopy];
    code[pnew]=CODE_SetPeriodic(code[pcopy]);
    velrhop[pnew]=velrhop[pcopy];
    if(varspre) {
      posxypre[pnew]=posxypre[pcopy];
      poszpre[pnew]=poszpre[pcopy];
      velrhoppre[pnew]=velrhoppre[pcopy];
    }
    if(auxnn)auxnn[pnew]=auxnn[pcopy];
  }
}

//==============================================================================
/// Creates periodic particles from a list of particles to duplicate for non-Newotnian formulation..
/// Crea particulas periodicas a partir de una lista con las particulas a duplicar.
//==============================================================================
void PeriodicDuplicateSymplectic(unsigned n,unsigned pini
  ,tuint3 domcells,tdouble3 perinc,const unsigned *listp,unsigned *idp,typecode *code,unsigned *dcell
  ,double2 *posxy,double *posz,float4 *velrhop,float *auxnn,double2 *posxypre,double *poszpre,float4 *velrhoppre)
{
  if(n) {
    uint3 cellmax=make_uint3(domcells.x-1,domcells.y-1,domcells.z-1);
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    if(posxypre!=NULL)KerPeriodicDuplicateSymplectic_NN<true><<<sgrid,SPHBSIZE>>>(n,pini,cellmax,Double3(perinc),listp,idp,code,dcell,posxy,posz,velrhop,auxnn,posxypre,poszpre,velrhoppre);
    else                 KerPeriodicDuplicateSymplectic_NN<false><<<sgrid,SPHBSIZE>>>(n,pini,cellmax,Double3(perinc),listp,idp,code,dcell,posxy,posz,velrhop,auxnn,posxypre,poszpre,velrhoppre);
  }
}

//------------------------------------------------------------------------------
/// Creates periodic particles from a list of particles to duplicate.
/// It is assumed that all particles are valid.
/// This kernel applies to single-GPU and multi-GPU because it uses domposmin.
///
/// Crea particulas periodicas a partir de una lista con las particulas a duplicar.
/// Se presupone que todas las particulas son validas.
/// Este kernel vale para single-gpu y multi-gpu porque usa domposmin. 
//------------------------------------------------------------------------------
__global__ void KerPeriodicDuplicateVerlet(unsigned n,unsigned pini,uint3 cellmax,double3 perinc
  ,const unsigned *listp,unsigned *idp,typecode *code,unsigned *dcell
  ,double2 *posxy,double *posz,float4 *velrhop,float *auxnn,float4 *velrhopm1)
{
  const unsigned p=blockIdx.x*blockDim.x+threadIdx.x; //-Number of particle.
  if(p<n) {
    const unsigned pnew=p+pini;
    const unsigned rp=listp[p];
    const unsigned pcopy=(rp&0x7FFFFFFF);
    //-Adjusts cell position of the new particles.
    //-Ajusta posicion y celda de nueva particula.
    KerPeriodicDuplicatePos(pnew,pcopy,(rp>=0x80000000),perinc.x,perinc.y,perinc.z,cellmax,posxy,posz,dcell);
    //-Copies the remaining data.
    //-Copia el resto de datos.
    idp[pnew]=idp[pcopy];
    code[pnew]=CODE_SetPeriodic(code[pcopy]);
    velrhop[pnew]=velrhop[pcopy];
    velrhopm1[pnew]=velrhopm1[pcopy];
    if(auxnn)auxnn[pnew]=auxnn[pcopy];
  }
}

//==============================================================================
/// Creates periodic particles from a list of particles to duplicate.
/// Crea particulas periodicas a partir de una lista con las particulas a duplicar.
//==============================================================================
void PeriodicDuplicateVerlet(unsigned n,unsigned pini,tuint3 domcells,tdouble3 perinc
  ,const unsigned *listp,unsigned *idp,typecode *code,unsigned *dcell
  ,double2 *posxy,double *posz,float4 *velrhop,float *auxnn,float4 *velrhopm1)
{
  if(n) {
    uint3 cellmax=make_uint3(domcells.x-1,domcells.y-1,domcells.z-1);
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerPeriodicDuplicateVerlet<<<sgrid,SPHBSIZE>>>(n,pini,cellmax,Double3(perinc),listp,idp,code,dcell,posxy,posz,velrhop,auxnn,velrhopm1);
  }
}

//##############################################################################
//# Kernels for calculating NN Tensors.
//# Kernels para calcular tensores.
//##############################################################################
//------------------------------------------------------------------------------
/// Velocity gradients for non-Newtonian models using FDAs approach.
/// Gradientes de velocidad usando FDAs.
//------------------------------------------------------------------------------
__device__ void GetVelocityGradients_FDA(float rr2,float drx,float dry,float drz
  ,float dvx,float dvy,float dvz,tmatrix3f &dvelp1,float &div_vel)
{
  //vel gradients
  dvelp1.a11=dvx*drx/rr2; dvelp1.a12=dvx*dry/rr2; dvelp1.a13=dvx*drz/rr2; //Fan et al., 2010
  dvelp1.a21=dvy*drx/rr2; dvelp1.a22=dvy*dry/rr2; dvelp1.a23=dvy*drz/rr2;
  dvelp1.a31=dvz*drx/rr2; dvelp1.a32=dvz*dry/rr2; dvelp1.a33=dvz*drz/rr2;
  div_vel=(dvelp1.a11+dvelp1.a22+dvelp1.a33)/3.f;
}

//==============================================================================
//symetric tensors
//==============================================================================
/// Calculates the Stress Tensor (symetric)
/// Obtener tensor de velocidad de deformacion symetric.
//==============================================================================
__device__ void GetStressTensor_sym(float2 &d_p1_xx_xy,float2 &d_p1_xz_yy,float2 &d_p1_yz_zz,float visco_etap1
  ,float &I_t,float &II_t,float &J1_t,float &J2_t,float &tau_tensor_magn
  ,float2 &tau_xx_xy,float2 &tau_xz_yy,float2 &tau_yz_zz)
{
  //Stress tensor and invariant
  tau_xx_xy.x=2.f*visco_etap1*(d_p1_xx_xy.x);	tau_xx_xy.y=2.f*visco_etap1*d_p1_xx_xy.y;		tau_xz_yy.x=2.f*visco_etap1*d_p1_xz_yy.x;
  tau_xz_yy.y=2.f*visco_etap1*(d_p1_xz_yy.y);	tau_yz_zz.x=2.f*visco_etap1*d_p1_yz_zz.x;
  tau_yz_zz.y=2.f*visco_etap1*(d_p1_yz_zz.y);
  //I_t - the first invariant -
  I_t=tau_xx_xy.x+tau_xz_yy.y+tau_yz_zz.y;
  //II_t - the second invariant - expnaded form witout symetry 
  float II_t_1=tau_xx_xy.x*tau_xz_yy.y+tau_xz_yy.y*tau_yz_zz.y+tau_xx_xy.x*tau_yz_zz.y;
  float II_t_2=tau_xx_xy.y*tau_xx_xy.y+tau_yz_zz.x*tau_yz_zz.x+tau_xz_yy.x*tau_xz_yy.x;
  II_t=-II_t_1+II_t_2;
  //stress tensor magnitude
  tau_tensor_magn=sqrt(II_t);
  //if (II_t < 0.f) {
  //	printf("****tau_tensor_magn is negative**** \n");
  //}
  //Main Stress rate invariants
  J1_t=I_t; J2_t=I_t*I_t-2.f*II_t;
}

/// Calculates the Stress Tensor - multilayer fluid (symetric)
//==============================================================================
__device__ void GetStressTensorMultilayerFluid_sym(float2 &d_p1_xx_xy,float2 &d_p1_xz_yy,float2 &d_p1_yz_zz,float visco_etap1
  ,float2 &tau_xx_xy,float2 &tau_xz_yy,float2 &tau_yz_zz)
{
  //Stress tensor and invariant
  tau_xx_xy.x=2.f*visco_etap1*(d_p1_xx_xy.x);	tau_xx_xy.y=2.f*visco_etap1*d_p1_xx_xy.y;		tau_xz_yy.x=2.f*visco_etap1*d_p1_xz_yy.x;
  tau_xz_yy.y=2.f*visco_etap1*(d_p1_xz_yy.y);	tau_yz_zz.x=2.f*visco_etap1*d_p1_yz_zz.x;
  tau_yz_zz.y=2.f*visco_etap1*(d_p1_yz_zz.y);
}

//==============================================================================
__device__ void GetStressInvariant(float &I1_t,float &J2_t,float &tau_xx, float &tau_xy, float &tau_xz, float &tau_yy, float &tau_yz, float &tau_zz)
{
  I1_t = tau_xx + tau_yy + tau_zz;
  J2_t = (pow((tau_xx - tau_zz),2) + pow((tau_yy - tau_zz),2) + pow((tau_yy - tau_xx),2)) / 6. 
                  + tau_xy * tau_xy + tau_yz * tau_yz + tau_xz * tau_xz;
}

// _device_ void DPVariable(const float)



__device__ void GetDPYieldFunction(float &f, const float &J2_t, const float &I1_t, const float &DP_AlphaPhi, const float &DP_kc)
{
  f = sqrt(J2_t) + DP_AlphaPhi * I1_t - DP_kc;
}

//==============================================================================
/// Calculates the Soil effective stress Tensor (symetric) based on Drucker-Prager
/// Note all the sigma are effective stress tensor
//==============================================================================
__device__ void GetStressTensorMultilayerSoil_sym(float2 &d_rate_xx_xy,float2 &d_rate_xz_yy,float2 &d_rate_yz_zz, float3 &dtspinrate_xyz 
    ,float2 &tau_xx_xy,float2 &tau_xz_yy,float2 &tau_yz_zz, float2 &Dp_xx_xy, float2 &Dp_xz_yy, float2 &Dp_yz_zz, const float DP_K, const float DP_G
    ,const float MC_phi, const float MC_c, const float MC_psi, double &dt)
{
  // Effective stress tensor before update
  tmatrix3f Sigma_tensor= {tau_xx_xy.x, tau_xx_xy.y, tau_xz_yy.x, 
                           tau_xx_xy.y, tau_xz_yy.y, tau_yz_zz.x, 
                           tau_xz_yy.x, tau_yz_zz.x, tau_yz_zz.y};
                          
  // Elastic stiffness tensor, isotropic, homogeneous
  float K4G3 = float(DP_K + 4.*DP_G/3.);
  float K2G3 = float(DP_K - 2.*DP_G/3.);
  // Only the left-top corner entries, the rest right-bottom corner is just DP_G for every diagnoal entry
  tmatrix3f De = {K4G3, K2G3, K2G3,
                  K2G3, K4G3, K2G3,
                  K2G3, K2G3, K4G3};
  // inverse of the elastic stiffness tensor, isotropic, homogeneous
  float invK4G3 = (K2G3 + K4G3) / (-2 * K2G3*K2G3 + K2G3*K4G3 + K4G3*K4G3);
  float invK2G3 = -K2G3 / (-2 * K2G3*K2G3 + K2G3*K4G3 + K4G3*K4G3);
  // Only the left-top corner entries, the rest right-bottom corner is just 1/DP_G for every diagnoal entry
  tmatrix3f invDe = {invK4G3, invK2G3, invK2G3,
                  invK2G3, invK4G3, invK2G3,
                  invK2G3, invK2G3, invK4G3};
  // Elastic predictor effective tensor
  tmatrix3f Sigma_e_pre;
  Sigma_e_pre.a11 = Sigma_tensor.a11 + (De.a11 * d_rate_xx_xy.x + De.a12 * d_rate_xz_yy.y + De.a13 * d_rate_yz_zz.y)*dt; //d_rate is the strain rate tensor 
  Sigma_e_pre.a22 = Sigma_tensor.a22 + (De.a21 * d_rate_xx_xy.x + De.a22 * d_rate_xz_yy.y + De.a23 * d_rate_yz_zz.y)*dt; 
  Sigma_e_pre.a33 = Sigma_tensor.a33 + (De.a31 * d_rate_xx_xy.x + De.a32 * d_rate_xz_yy.y + De.a33 * d_rate_yz_zz.y)*dt; 
  Sigma_e_pre.a12 = Sigma_tensor.a12 + DP_G * d_rate_xx_xy.y * dt;
  Sigma_e_pre.a13 = Sigma_tensor.a13 + DP_G * d_rate_xz_yy.x * dt;
  Sigma_e_pre.a23 = Sigma_tensor.a23 + DP_G * d_rate_yz_zz.x * dt;
  Sigma_e_pre.a21 = Sigma_e_pre.a12;
  Sigma_e_pre.a31 = Sigma_e_pre.a13;
  Sigma_e_pre.a32 = Sigma_e_pre.a23;
  // Store current plastic strain Dp in a temporary variable Dp_temp
  float2 Dp_temp_xx_xy=make_float2(0,0);
  float2 Dp_temp_xz_yy=make_float2(0,0);
  float2 Dp_temp_yz_zz=make_float2(0,0);
  Dp_temp_xx_xy.x = Dp_xx_xy.x; Dp_temp_xx_xy.y = Dp_xx_xy.y; Dp_temp_xz_yy.x = Dp_xz_yy.x;
  Dp_temp_xz_yy.y = Dp_xz_yy.y; Dp_temp_yz_zz.x = Dp_yz_zz.x; 
  Dp_temp_yz_zz.y = Dp_yz_zz.y;     

  // Evaluation yield condition
  float DP_AlphaPhi = tan(MC_phi) / sqrt(9. + 12.*tan(MC_phi)*tan(MC_phi)); // Use MC phi and c to calculate DP parameters for the yield criterion, need to check plane strain/stress or other conditions.
  float DP_kc = 3.* MC_c / sqrt(9. + 12.*tan(MC_phi)*tan(MC_phi)); 
  float DP_psi = tan(MC_psi) / sqrt(9. + 12. * tan(MC_psi)*tan(MC_psi)); // Use MC psi (dilation angle) to calculate DP dilation angle for the flow rule
  float I1_t, J2_t, f, dlambda;
  GetStressInvariant(I1_t, J2_t, Sigma_e_pre.a11, Sigma_e_pre.a12, Sigma_e_pre.a13, Sigma_e_pre.a22, Sigma_e_pre.a23, Sigma_e_pre.a33);
  GetDPYieldFunction(f, J2_t, I1_t, DP_psi, DP_kc); // Calculate yield function f 
  
  int iter; // set maximum iteration by checking iter < iterLimit if needed. iterLimit not used for now.
  iter = 0;

  if(f<0){
      // Elastic
      Sigma_tensor.a11 = Sigma_e_pre.a11;
      Sigma_tensor.a12 = Sigma_e_pre.a12;
      Sigma_tensor.a13 = Sigma_e_pre.a13;
      Sigma_tensor.a21 = Sigma_e_pre.a21;
      Sigma_tensor.a22 = Sigma_e_pre.a22;
      Sigma_tensor.a23 = Sigma_e_pre.a23;
      Sigma_tensor.a31 = Sigma_e_pre.a31;
      Sigma_tensor.a32 = Sigma_e_pre.a32;
      Sigma_tensor.a33 = Sigma_e_pre.a33;
      Dp_xx_xy.x = Dp_temp_xx_xy.x; Dp_xx_xy.y = Dp_temp_xx_xy.y; Dp_xz_yy.x = Dp_temp_xz_yy.x;
      Dp_xz_yy.y = Dp_temp_xz_yy.y; Dp_yz_zz.x = Dp_temp_yz_zz.x; 
      Dp_yz_zz.y = Dp_temp_yz_zz.y; 
      }
  else{ // plastic corrector
      double err = 1e-5;
      while (f > err){
          dlambda = f /(9.*DP_K*DP_AlphaPhi*DP_psi + DP_G); //Not sure how this is derived. Assume non-associated flow rule? Plastic multiplier
          float GJ2 = DP_G/sqrt(J2_t);
          float Ddg_dsigmaxx = 3*DP_K * DP_psi + GJ2 * (Sigma_e_pre.a11 - I1_t/3.); // Need to check if this is derived correctly
          float Ddg_dsigmayy = 3*DP_K * DP_psi + GJ2 * (Sigma_e_pre.a22 - I1_t/3.);
          float Ddg_dsigmazz = 3*DP_K * DP_psi + GJ2 * (Sigma_e_pre.a33 - I1_t/3.);
          float Ddg_dsigmaxy = GJ2 * Sigma_e_pre.a12;
          float Ddg_dsigmaxz = GJ2 * Sigma_e_pre.a13;
          float Ddg_dsigmayz = GJ2 * Sigma_e_pre.a23;

          // Calculate increment of plastic stress component dsigmap= dlambda * Ddg_dsigma
          float dsigmap_xx = dlambda * Ddg_dsigmaxx;
          float dsigmap_yy = dlambda * Ddg_dsigmayy;
          float dsigmap_zz = dlambda * Ddg_dsigmazz;
          float dsigmap_xy = dlambda * Ddg_dsigmaxy;
          float dsigmap_xz = dlambda * Ddg_dsigmaxz;
          float dsigmap_yz = dlambda * Ddg_dsigmayz;
          // Update stress
          Sigma_e_pre.a11 -= dsigmap_xx;    Sigma_e_pre.a12 -= dsigmap_xy;    Sigma_e_pre.a13 -= dsigmap_xz;
          Sigma_e_pre.a21 -= dsigmap_xy;    Sigma_e_pre.a22 -= dsigmap_yy;    Sigma_e_pre.a23 -= dsigmap_yz;
          Sigma_e_pre.a31 -= dsigmap_xz;    Sigma_e_pre.a32 -= dsigmap_yz;    Sigma_e_pre.a33 -= dsigmap_zz;
          // Plastic strain increment dDp = invD * dsigmap
		  float dDp_xx, dDp_yy, dDp_zz, dDp_yz, dDp_xz, dDp_xy;
          dDp_xx = invDe.a11*dsigmap_xx + invDe.a12*dsigmap_yy + invDe.a13*dsigmap_zz;
			    dDp_yy = invDe.a21*dsigmap_xx + invDe.a22*dsigmap_yy + invDe.a23*dsigmap_zz;
			    dDp_zz = invDe.a31*dsigmap_xx + invDe.a32*dsigmap_yy + invDe.a33*dsigmap_zz;
			    dDp_xy = 1./DP_G * dsigmap_xy;
			    dDp_yz = 1./DP_G * dsigmap_yz;
			    dDp_xz = 1./DP_G * dsigmap_xz;
          //Update plastic strain
          Dp_temp_xx_xy.x += dDp_xx;     Dp_temp_xx_xy.y += dDp_xy;     Dp_temp_xz_yy.x += dDp_xz;
          Dp_temp_xz_yy.y += dDp_yy;     Dp_temp_yz_zz.x += dDp_yz;
          Dp_temp_yz_zz.y += dDp_zz;
          // Check updated f
          GetStressInvariant(I1_t, J2_t, Sigma_e_pre.a11, Sigma_e_pre.a12, Sigma_e_pre.a13, Sigma_e_pre.a22, Sigma_e_pre.a23, Sigma_e_pre.a33);
          GetDPYieldFunction(f, J2_t, I1_t, DP_AlphaPhi, DP_kc); // Update yield function f 
          iter += 1;
      }
      // Update stress and plastic strain
      Sigma_tensor.a11 = Sigma_e_pre.a11; Sigma_tensor.a12 = Sigma_e_pre.a12; Sigma_tensor.a13 = Sigma_e_pre.a13;
      Sigma_tensor.a21 = Sigma_e_pre.a21; Sigma_tensor.a22 = Sigma_e_pre.a22; Sigma_tensor.a23 = Sigma_e_pre.a23;
      Sigma_tensor.a31 = Sigma_e_pre.a31; Sigma_tensor.a32 = Sigma_e_pre.a32; Sigma_tensor.a33 = Sigma_e_pre.a33;
      Dp_xx_xy.x = Dp_temp_xx_xy.x; Dp_xx_xy.y = Dp_temp_xx_xy.y; Dp_xz_yy.x = Dp_temp_xz_yy.x;
      Dp_xz_yy.y = Dp_temp_xz_yy.y; Dp_yz_zz.x = Dp_temp_yz_zz.x;
      Dp_yz_zz.y = Dp_temp_yz_zz.y;
    }
 
  // Correct final updated effective stress with spin rate tensor
  tmatrix3f W_tensor = {0, dtspinrate_xyz.x, dtspinrate_xyz.y,
                       -dtspinrate_xyz.x, 0, dtspinrate_xyz.z,
                       -dtspinrate_xyz.y, -dtspinrate_xyz.z,0};

  Sigma_tensor.a11 = Sigma_tensor.a11 + ((W_tensor.a11*Sigma_tensor.a11+W_tensor.a12*Sigma_tensor.a21+W_tensor.a13*Sigma_tensor.a31) - (Sigma_tensor.a11*W_tensor.a11+Sigma_tensor.a12*W_tensor.a21+Sigma_tensor.a13*W_tensor.a31))*dt;
  Sigma_tensor.a12 = Sigma_tensor.a12 + ((W_tensor.a11*Sigma_tensor.a12+W_tensor.a12*Sigma_tensor.a22+W_tensor.a13*Sigma_tensor.a32) - (Sigma_tensor.a11*W_tensor.a12+Sigma_tensor.a12*W_tensor.a22+Sigma_tensor.a13*W_tensor.a32))*dt;
  Sigma_tensor.a13 = Sigma_tensor.a13 + ((W_tensor.a11*Sigma_tensor.a13+W_tensor.a12*Sigma_tensor.a23+W_tensor.a13*Sigma_tensor.a33) - (Sigma_tensor.a11*W_tensor.a13+Sigma_tensor.a12*W_tensor.a23+Sigma_tensor.a13*W_tensor.a33))*dt;
  Sigma_tensor.a21 = Sigma_tensor.a13 + ((W_tensor.a21*Sigma_tensor.a11+W_tensor.a22*Sigma_tensor.a21+W_tensor.a23*Sigma_tensor.a31) - (Sigma_tensor.a21*W_tensor.a11+Sigma_tensor.a22*W_tensor.a21+Sigma_tensor.a23*W_tensor.a31))*dt;
  Sigma_tensor.a22 = Sigma_tensor.a22 + ((W_tensor.a21*Sigma_tensor.a12+W_tensor.a22*Sigma_tensor.a22+W_tensor.a23*Sigma_tensor.a32) - (Sigma_tensor.a21*W_tensor.a12+Sigma_tensor.a22*W_tensor.a22+Sigma_tensor.a23*W_tensor.a32))*dt;
  Sigma_tensor.a23 = Sigma_tensor.a23 + ((W_tensor.a21*Sigma_tensor.a13+W_tensor.a22*Sigma_tensor.a23+W_tensor.a23*Sigma_tensor.a33) - (Sigma_tensor.a21*W_tensor.a13+Sigma_tensor.a22*W_tensor.a23+Sigma_tensor.a23*W_tensor.a33))*dt;
  Sigma_tensor.a31 = Sigma_tensor.a31 + ((W_tensor.a31*Sigma_tensor.a11+W_tensor.a32*Sigma_tensor.a21+W_tensor.a33*Sigma_tensor.a31) - (Sigma_tensor.a31*W_tensor.a11+Sigma_tensor.a32*W_tensor.a21+Sigma_tensor.a33*W_tensor.a31))*dt;
  Sigma_tensor.a32 = Sigma_tensor.a32 + ((W_tensor.a31*Sigma_tensor.a12+W_tensor.a32*Sigma_tensor.a22+W_tensor.a33*Sigma_tensor.a32) - (Sigma_tensor.a31*W_tensor.a12+Sigma_tensor.a32*W_tensor.a22+Sigma_tensor.a33*W_tensor.a32))*dt;
  Sigma_tensor.a33 = Sigma_tensor.a33 + ((W_tensor.a31*Sigma_tensor.a13+W_tensor.a32*Sigma_tensor.a23+W_tensor.a33*Sigma_tensor.a33) - (Sigma_tensor.a31*W_tensor.a13+Sigma_tensor.a32*W_tensor.a23+Sigma_tensor.a33*W_tensor.a33))*dt;

  tau_xx_xy.x = Sigma_tensor.a11;
  tau_xx_xy.y = 0.5f * (Sigma_tensor.a12 + Sigma_tensor.a21);
  tau_xz_yy.x = 0.5f * (Sigma_tensor.a13 + Sigma_tensor.a31);
  tau_yz_zz.x = 0.5f * (Sigma_tensor.a23 + Sigma_tensor.a32);
  tau_xz_yy.y = Sigma_tensor.a22;
  tau_yz_zz.y = Sigma_tensor.a33;
}

//==============================================================================
/// Calculates the Stress Tensor for the soil phase in Multilayer module (symetric)
//==============================================================================

//==============================================================================
/// Perform return mapping with DruckPrager Yeild Criterion
/// Input Current Elastic Stress, DruckPrager Parameters, Ouput EP Stress, Plastic Strain
/// Note that the inverse of elastic stiffness matrix is needed for plastic strain 
//==============================================================================
__device__ void GetStressTensor_PlasticCorrector(float2 &sigma_xx_xy,float2 &sigma_xz_yy,float2 &sigma_yz_zz, float2 &ep_tensor_xx_xy, float2 &ep_tensor_xz_yy, float2 &ep_tensor_yz_zz
, const float DP_K, const float DP_G,const float MC_phi, const float MC_c, const float MC_psi)
{
		// Get elastic stress tensor (obtained in time integration at previous timestep)
		float2 tsigmap1_xx_xy = sigma_xx_xy;
		float2 tsigmap1_xz_yy = sigma_xz_yy;
		float2 tsigmap1_yz_zz = sigma_yz_zz;

    // Store current plastic strain ep in a temporary variable ep_temp
    float2 ep_temp_xx_xy=ep_tensor_xx_xy;
    float2 ep_temp_xz_yy=ep_tensor_xz_yy;
    float2 ep_temp_yz_zz=ep_tensor_yz_zz;

    // Build the inversed elastic stiffness matrix for evaluating plastic strain  
    float K4G3 = float(DP_K + 4.*DP_G / 3.);
	  float K2G3 = float(DP_K - 2.*DP_G / 3.); 
	  float invK4G3 = (K2G3 + K4G3) / (-2 * K2G3*K2G3 + K2G3*K4G3 + K4G3*K4G3);
	  float invK2G3 = -K2G3 / (-2 * K2G3*K2G3 + K2G3*K4G3 + K4G3*K4G3);
	  float invm_a11 = invK4G3; float invm_a12 = invK2G3; float invm_a13 = invK2G3;
	  float invm_a21 = invK2G3; float invm_a22 = invK4G3; float invm_a23 = invK2G3;
	  float invm_a31 = invK2G3; float invm_a32 = invK2G3; float invm_a33 = invK4G3;
    
    // Evaluation yield condition
    float DP_AlphaPhi = tan(MC_phi) / sqrt(9. + 12.*tan(MC_phi)*tan(MC_phi)); // Use MC phi and c to calculate DP parameters for the yield criterion, need to check plane strain/stress or other conditions.
    float DP_kc = 3.* MC_c / sqrt(9. + 12.*tan(MC_phi)*tan(MC_phi)); 
    float DP_psi = tan(MC_psi) / sqrt(9. + 12. * tan(MC_psi)*tan(MC_psi)); // Use MC psi (dilation angle) to calculate DP dilation angle for the flow rule
    float I1_t, J2_t, f;
    GetStressInvariant(I1_t, J2_t, tsigmap1_xx_xy.x, tsigmap1_xx_xy.y, tsigmap1_xz_yy.x, tsigmap1_xz_yy.y, tsigmap1_yz_zz.x, tsigmap1_yz_zz.y);
    GetDPYieldFunction(f, J2_t, I1_t, DP_psi, DP_kc); // Calculate yield function f 

		// Plastic corrector if violates yield condition
		if (f > 1e-5f) {
			int iter = 0;
			while (abs(f) > 1e-5f) {
				if (iter++ > 80) break;
          float dlambda = f /(9.*DP_K*DP_AlphaPhi*DP_psi + DP_G); //Plastic multiplier
          float GJ2 = DP_G/sqrt(J2_t);
          //evaluate De : plastic potential
          float Deppxx = 3*DP_K * DP_psi + GJ2 * (tsigmap1_xx_xy.x - I1_t/3.); // Need to check if this is derived correctly
          float Deppyy = 3*DP_K * DP_psi + GJ2 * (tsigmap1_xz_yy.y - I1_t/3.);
          float Deppzz = 3*DP_K * DP_psi + GJ2 * (tsigmap1_yz_zz.y - I1_t/3.);
          float Deppxy = GJ2 * tsigmap1_xx_xy.y;
          float Deppxz = GJ2 * tsigmap1_xz_yy.x;
          float Deppyz = GJ2 * tsigmap1_yz_zz.x;
          // Calculate increment of plastic stress increment dsigmap= dlambda * pp
          float dsigmap_xx = dlambda * Deppxx;
          float dsigmap_yy = dlambda * Deppyy;
          float dsigmap_zz = dlambda * Deppzz;
          float dsigmap_xy = dlambda * Deppxy;
          float dsigmap_xz = dlambda * Deppxz;
          float dsigmap_yz = dlambda * Deppyz;
          // Update elastic-plastic stress
          tsigmap1_xx_xy.x -= dsigmap_xx;    tsigmap1_xx_xy.y -= dsigmap_xy;
          tsigmap1_xz_yy.x -= dsigmap_xz;    tsigmap1_xz_yy.y -= dsigmap_yy;
          tsigmap1_yz_zz.x -= dsigmap_yz;    tsigmap1_yz_zz.y -= dsigmap_zz;
          // Plastic strain increment dep = invD * dsigmap
		      float dep_xx, dep_yy, dep_zz, dep_yz, dep_xz, dep_xy;
          dep_xx = invm_a11*dsigmap_xx + invm_a12*dsigmap_yy + invm_a13*dsigmap_zz;
			    dep_yy = invm_a21*dsigmap_xx + invm_a22*dsigmap_yy + invm_a23*dsigmap_zz;
			    dep_zz = invm_a31*dsigmap_xx + invm_a32*dsigmap_yy + invm_a33*dsigmap_zz;
			    dep_xy = 0.5f/DP_G*dsigmap_xy;
			    dep_yz = 0.5f/DP_G*dsigmap_yz;
			    dep_xz = 0.5f/DP_G*dsigmap_xz;
          //Update plastic strain
          ep_temp_xx_xy.x += dep_xx;     ep_temp_xx_xy.y += dep_xy;     ep_temp_xz_yy.x += dep_xz;
          ep_temp_xz_yy.y += dep_yy;     ep_temp_yz_zz.x += dep_yz;
          ep_temp_yz_zz.y += dep_zz;
          // Check updated f
          GetStressInvariant(I1_t, J2_t, tsigmap1_xx_xy.x, tsigmap1_xx_xy.y, tsigmap1_xz_yy.x, tsigmap1_xz_yy.y, tsigmap1_yz_zz.x, tsigmap1_yz_zz.y);
          GetDPYieldFunction(f, J2_t, I1_t, DP_AlphaPhi, DP_kc); // Update yield function f 
          iter += 1;
			}
		}
    ep_tensor_xx_xy.x = ep_temp_xx_xy.x; ep_tensor_xx_xy.y = ep_temp_xx_xy.y; 
    ep_tensor_xz_yy.x = ep_temp_xz_yy.x; ep_tensor_xz_yy.y = ep_temp_xz_yy.y; 
    ep_tensor_yz_zz.x = ep_temp_yz_zz.x; ep_tensor_yz_zz.y = ep_temp_yz_zz.y;
		sigma_xx_xy = make_float2(tsigmap1_xx_xy.x, tsigmap1_xx_xy.y);
		sigma_xz_yy = make_float2(tsigmap1_xz_yy.x, tsigmap1_xz_yy.y);
		sigma_yz_zz = make_float2(tsigmap1_yz_zz.x, tsigmap1_yz_zz.y);
}

//==============================================================================
/// Calculate strain/spin rate tensor
/// Input velgradient (3*3); Ouput strain (3*2)/spin (3*1) rate tensor
//==============================================================================
__device__ void GetStrainSpinRateTensor_sym(float3 gradvp1_xx_xy_xz,float3 gradvp1_yx_yy_yz,float3 gradvp1_zx_zy_zz
  ,float2 &e_tensor_xx_xy,float2 &e_tensor_xz_yy,float2 &e_tensor_yz_zz,float3 &w_tensor_xy_yz_xz)
{
  //Build strain rate tensor
  float e_tensor_xx_xy.x=gradvp1_xx_xy_xz.x;		
  float e_tensor_xz_yy.y=gradvp1_yx_yy_yz.y;	  
  float e_tensor_yz_zz.y=gradvp1_zx_zy_zz.z;
  float e_tensor_xx_xy.y=0.5f*(gradvp1_xx_xy_xz.y+gradvp1_yx_yy_yz.x);
  float e_tensor_yz_zz.x=0.5f*(gradvp1_yx_yy_yz.z+gradvp1_zx_zy_zz.y);
  float e_tensor_xz_yy.x=0.5f*(gradvp1_xx_xy_xz.z+gradvp1_zx_zy_zz.x);

  //Build spin rate tensor
  float w_tensor_xy_yz_xz.x = 0.5f*(gradvp1_xx_xy_xz.y-gradvp1_yx_yy_yz.x);
  float w_tensor_xy_yz_xz.y = 0.5f*(gradvp1_yx_yy_yz.z-gradvp1_zx_zy_zz.y);
  float w_tensor_xy_yz_xz.z = 0.5f*(gradvp1_xx_xy_xz.z-gradvp1_zx_zy_zz.x);
}

//==============================================================================
/// Calculate elastic stress rate tensor
/// Input Strain/Spin Rate, Elastic Parameters, Stress, Ouput Elastic Stress Rate Tensor
//==============================================================================
__device__ void GetStressRateTensor_Elastic(float2 e_tensor_xx_xy,float2 e_tensor_xz_yy,float2 e_tensor_yz_zz,float3 w_tensor_xy_yz_xz
,float2 sigma_xx_xy,float2 sigma_xz_yy,float2 sigma_yz_zz
,const float DP_K, const float DP_G
,float2 &rsigma_xx_xy,float2 &rsigma_xz_yy,float2 &rsigma_yz_zz)
{
  //Build Elastic stiffness matrix
	float K4G3 = float(DP_K + 4.*DP_G / 3.);
	float K2G3 = float(DP_K - 2.*DP_G / 3.);
	float m_a11 = K4G3; float m_a12 = K2G3; float m_a13 = K2G3;
	float m_a21 = K2G3; float m_a22 = K4G3; float m_a23 = K2G3;
	float m_a31 = K2G3; float m_a32 = K2G3; float m_a33 = K4G3; 

  //Get stress, stran rate and spin rate
  float sigmaxx = sigma_xx_xy.x;
  float sigmaxy = sigma_xx_xy.y;
  float sigmaxz = sigma_xz_yy.x;
  float sigmayy = sigma_xz_yy.y;
  float sigmayz = sigma_yz_zz.x;
  float sigmazz = sigma_yz_zz.y;

  float exx = e_tensor_xx_xy.x;
  float exy = e_tensor_xx_xy.y;
  float exz = e_tensor_xz_yy.x;
  float eyy = e_tensor_xz_yy.y;
  float eyz = e_tensor_yz_zz.x;
  float ezz = e_tensor_yz_zz.y;

  float wxy = w_tensor_xy_yz_xz.x;
  float wyz = w_tensor_xy_yz_xz.y;
  float wxz = w_tensor_xy_yz_xz.z;
  
  //Jaumann stress rate
  float Jxx = - 2.f*sigmaxy*wxy - 2.f*sigmaxz*wxz;
  float Jxy = sigmaxx*wxy - sigmayy*wxy - sigmaxz*wyz - sigmayz*wxz;
  float Jxz = sigmaxx*wxz + sigmaxy*wyz - sigmayz*wxy - sigmazz*wxz;
  float Jyy = 2.f*sigmaxy*wxy - 2.f*sigmayz*wyz;
  float Jyz = sigmaxy*wxz + sigmaxz*wxy + sigmayy*wyz - sigmazz*wyz;
  float Jzz = 2.f*sigmaxz*wxz + 2.f*sigmayz*wyz;
 
  //Construct stress rate equation
  rsigma_xx_xy.x = (m_a11*exx+m_a12*eyy+m_a13*ezz);//+Jxx
  rsigma_xz_yy.y = (m_a21*exx+m_a22*eyy+m_a23*ezz);//+Jyy
  rsigma_yz_zz.y = (m_a31*exx+m_a32*eyy+m_a33*ezz);//+Jzz
  rsigma_xx_xy.y = 2.f*DP_G*exy;//+Jxy
  rsigma_yz_zz.x = 2.f*DP_G*eyz;//+Jyz
  rsigma_xz_yy.x = 2.f*DP_G*exz;//+Jxz  
}

//==============================================================================
/// Calculates the Strain Rate Tensor (symetric).
/// Obtener tensor de velocidad de deformacion symetric.
//==============================================================================
__device__ void GetStrainRateTensor_tsym(float2 &dvelp1_xx_xy,float2 &dvelp1_xz_yy,float2 &dvelp1_yz_zz
  ,float &I_D,float &II_D,float &J1_D,float &J2_D,float &div_D_tensor,float &D_tensor_magn
  ,float2 &D_tensor_xx_xy,float2 &D_tensor_xz_yy,float2 &D_tensor_yz_zz)
{
  //Strain tensor and invariant	
  float div_vel=(dvelp1_xx_xy.x+dvelp1_xz_yy.y+dvelp1_yz_zz.y)/3.f;
  D_tensor_xx_xy.x=dvelp1_xx_xy.x-div_vel;		D_tensor_xx_xy.y=0.5f*(dvelp1_xx_xy.y);		D_tensor_xz_yy.x=0.5f*(dvelp1_xz_yy.x);
  D_tensor_xz_yy.y=dvelp1_xz_yy.y-div_vel;	D_tensor_yz_zz.x=0.5f*(dvelp1_yz_zz.x);
  D_tensor_yz_zz.y=dvelp1_yz_zz.y-div_vel;
  //the off-diagonal entries of velocity gradients are i.e. 0.5f*(du/dy+dvdx) with dvelp1.xy=du/dy+dvdx
  div_D_tensor=(D_tensor_xx_xy.x+D_tensor_xz_yy.y+D_tensor_yz_zz.y)/3.f;

  ////I_D - the first invariant -
  I_D=D_tensor_xx_xy.x+D_tensor_xz_yy.y+D_tensor_yz_zz.y;
  //II_D - the second invariant - expnaded form witout symetry 
  float II_D_1=D_tensor_xx_xy.x*D_tensor_xz_yy.y+D_tensor_xz_yy.y*D_tensor_yz_zz.y+D_tensor_xx_xy.x*D_tensor_yz_zz.y;
  float II_D_2=D_tensor_xx_xy.y*D_tensor_xx_xy.y+D_tensor_yz_zz.x*D_tensor_yz_zz.x+D_tensor_xz_yy.x*D_tensor_xz_yy.x;
  II_D=-II_D_1+II_D_2;
  ////deformation tensor magnitude
  D_tensor_magn=sqrt((II_D));

  //Main Strain rate invariants
  J1_D=I_D; J2_D=I_D*I_D-2.f*II_D;
}

//==============================================================================
/// Calculate strain rate tensor and the spin rate tensor.
//==============================================================================
__device__ void GetStrainSpinRateTensor(float2 &dvelp1_xx_xy,float2 &dvelp1_xz_yy,float2 &dvelp1_yz_zz
  ,float2 &D_tensor_xx_xy,float2 &D_tensor_xz_yy,float2 &D_tensor_yz_zz, float3 &W_tensor_xyz)
{
  //Strain rate tensor
  D_tensor_xx_xy.x=dvelp1_xx_xy.x;		
  D_tensor_xz_yy.y=dvelp1_xz_yy.y;	  
  D_tensor_yz_zz.y=dvelp1_yz_zz.y;
  D_tensor_xx_xy.y=0.5f*(dvelp1_xx_xy_xz.y+dvelp1_yx_yy_yz.x);
  D_tensor_xz_yy.x=0.5f*(dvelp1_xx_xy_xz.z+dvelp1_zx_zy_zz.x);
  D_tensor_yz_zz.x=0.5f*(dvelp1_yx_yy_yz.z+dvelp1_zx_zy_zz.y);

  //Spin rate tensor
  W_tensor_xyz.x = 0.5f*(dvelp1_xx_xy_xz.y-dvelp1_yx_yy_yz.x);
  W_tensor_xyz.y = 0.5f*(dvelp1_xx_xy_xz.z-dvelp1_zx_zy_zz.x);
  W_tensor_xyz.z = 0.5f*(dvelp1_yx_yy_yz.z-dvelp1_zx_zy_zz.y);
}

//==============================================================================
/// Velocity gradients using SPH approach (No sym is considered)
/// Gradientes de velocidad usando SPH.
//==============================================================================
__device__ void GetVelocityGradients_SPH(float massp2,const float4 &velrhop2,float dvx,float dvy,float dvz,float frx,float fry,float frz
  ,float3 &grap1_xx_xy_xz,float3 &grap1_yx_yy_yz,float3 &grap1_zx_zy_zz)
{
  ///SPH vel gradients calculation
  const float volp2=-massp2/velrhop2.w;
  float dv=dvx*volp2;  grap1_xx_xy_xz.x+=dv*frx; grap1_xx_xy_xz.y+=dv*fry; grap1_xx_xy_xz.z+=dv*frz;
        dv=dvy*volp2;  grap1_yx_yy_yz.x+=dv*frx; grap1_yx_yy_yz.y+=dv*fry; grap1_yx_yy_yz.z+=dv*frz;
        dv=dvz*volp2;  grap1_zx_zy_zz.x+=dv*frx; grap1_zx_zy_zz.y+=dv*fry; grap1_zx_zy_zz.z+=dv*frz;
}


//==============================================================================
/// Velocity gradients using SPH approach.
/// Gradientes de velocidad usando SPH.
//==============================================================================
__device__ void GetVelocityGradients_SPH_tsym(float massp2,const float4 &velrhop2,float dvx,float dvy,float dvz,float frx,float fry,float frz
  ,float2 &grap1_xx_xy,float2 &grap1_xz_yy,float2 &grap1_yz_zz)
{
  ///SPH vel gradients calculation
  const float volp2=-massp2/velrhop2.w;
  float dv=dvx*volp2;  grap1_xx_xy.x+=dv*frx; grap1_xx_xy.y+=dv*fry; grap1_xz_yy.x+=dv*frz;
  dv=dvy*volp2;  grap1_xx_xy.y+=dv*frx;	grap1_xz_yy.y+=dv*fry; grap1_yz_zz.x+=dv*frz;
  dv=dvz*volp2;  grap1_xz_yy.x+=dv*frx; grap1_yz_zz.x+=dv*fry; grap1_yz_zz.y+=dv*frz;
}

//==============================================================================
/// Calculate strain rate tensor (full matrix).
/// Obtener tensor de velocidad de deformacion (full matrix).
//==============================================================================
__device__ void GetStrainRateTensor(const tmatrix3f &dvelp1,float div_vel,float &I_D,float &II_D,float &J1_D
  ,float &J2_D,float &div_D_tensor,float &D_tensor_magn,tmatrix3f &D_tensor)
{
  //Strain tensor and invariant
  D_tensor.a11=dvelp1.a11-div_vel;          D_tensor.a12=0.5f*(dvelp1.a12+dvelp1.a21);      D_tensor.a13=0.5f*(dvelp1.a13+dvelp1.a31);
  D_tensor.a21=0.5f*(dvelp1.a21+dvelp1.a12);      D_tensor.a22=dvelp1.a22-div_vel;          D_tensor.a23=0.5f*(dvelp1.a23+dvelp1.a32);
  D_tensor.a31=0.5f*(dvelp1.a31+dvelp1.a13);      D_tensor.a32=0.5f*(dvelp1.a32+dvelp1.a23);      D_tensor.a33=dvelp1.a33-div_vel;
  div_D_tensor=(D_tensor.a11+D_tensor.a22+D_tensor.a33)/3.f;

  //I_D - the first invariant -
  I_D=D_tensor.a11+D_tensor.a22+D_tensor.a33;
  //II_D - the second invariant - expnaded form witout symetry 
  float II_D_1=D_tensor.a11*D_tensor.a22+D_tensor.a22*D_tensor.a33+D_tensor.a11*D_tensor.a33;
  float II_D_2=D_tensor.a12*D_tensor.a21+D_tensor.a23*D_tensor.a32+D_tensor.a13*D_tensor.a31;
  II_D=II_D_1-II_D_2;
  //deformation tensor magnitude
  D_tensor_magn=sqrt((II_D*II_D));

  //Main Strain rate invariants
  J1_D=I_D; J2_D=I_D*I_D-2.f*II_D;
}
//==============================================================================
/// Calculates the effective visocity.
/// Calcule la viscosidad efectiva.
//==============================================================================
__device__ void KerGetEta_Effective(const typecode ppx,float tau_yield,float D_tensor_magn,float visco
  ,float m_NN,float n_NN,float &visco_etap1)
{

  if(D_tensor_magn<=ALMOSTZERO)D_tensor_magn=ALMOSTZERO;
  float miou_yield=(PHASECTE[ppx].tau_max ? PHASECTE[ppx].tau_max/(2.0f*D_tensor_magn) : (tau_yield)/(2.0f*D_tensor_magn)); //HPB will adjust eta		

  //if tau_max exists
  bool bi_region=PHASECTE[ppx].tau_max && D_tensor_magn<=PHASECTE[ppx].tau_max/(2.f*PHASECTE[ppx].Bi_multi*visco);
  if(bi_region) { //multiplier
    miou_yield=PHASECTE[ppx].Bi_multi*visco;
  }
  //Papanastasiou
  float miouPap=miou_yield *(1.f-exp(-m_NN*D_tensor_magn));
  float visco_etap1_term1=(PHASECTE[ppx].tau_max ? miou_yield : (miouPap>m_NN*tau_yield||D_tensor_magn==ALMOSTZERO ? m_NN*tau_yield : miouPap));

  //HB
  float miouHB=visco*pow(D_tensor_magn,(n_NN-1.0f));
  float visco_etap1_term2=(bi_region ? visco : (miouPap>m_NN*tau_yield||D_tensor_magn==ALMOSTZERO ? visco : miouHB));

  visco_etap1=visco_etap1_term1+visco_etap1_term2;

  /*
  //use according to you criteria
  - Herein we limit visco_etap1 at very low shear rates
  */
}
//------------------------------------------------------------------------------
/// Calclulate stress tensor.
/// Calcular tensor de estres.
//------------------------------------------------------------------------------
__device__ void GetStressTensor(const tmatrix3f &D_tensor,float visco_etap1,float &I_t,float &II_t,float &J1_t
  ,float &J2_t,float &tau_tensor_magn,tmatrix3f &tau_tensor)
{
  //Stress tensor and invariant
  tau_tensor.a11=2.f*visco_etap1*(D_tensor.a11);	tau_tensor.a12=2.f*visco_etap1*D_tensor.a12;		tau_tensor.a13=2.f*visco_etap1*D_tensor.a13;
  tau_tensor.a21=2.f*visco_etap1*D_tensor.a21;		tau_tensor.a22=2.f*visco_etap1*(D_tensor.a22);	tau_tensor.a23=2.f*visco_etap1*D_tensor.a23;
  tau_tensor.a31=2.f*visco_etap1*D_tensor.a31;		tau_tensor.a32=2.f*visco_etap1*D_tensor.a32;		tau_tensor.a33=2.f*visco_etap1*(D_tensor.a33);

  //I_t - the first invariant -
  I_t=tau_tensor.a11+tau_tensor.a22+tau_tensor.a33;
  //II_t - the second invariant - expnaded form witout symetry 
  float II_t_1=tau_tensor.a11*tau_tensor.a22+tau_tensor.a22*tau_tensor.a33+tau_tensor.a11*tau_tensor.a33;
  float II_t_2=tau_tensor.a12*tau_tensor.a21+tau_tensor.a23*tau_tensor.a32+tau_tensor.a13*tau_tensor.a31;
  II_t=II_t_1-II_t_2;
  //stress tensor magnitude
  tau_tensor_magn=sqrt(II_t*II_t);

  //Main Strain rate invariants
  J1_t=I_t; J2_t=I_t*I_t-2.f*II_t;
}

//##############################################################################
//# Kernels for calculating forces (Pos-Double) for non-Newtonian models.
//# Kernels para calculo de fuerzas (Pos-Double) para modelos no-Newtonianos.
//##############################################################################
//------------------------------------------------------------------------------
/// Interaction of a particle with a set of particles. Bound-Fluid/Float
/// Realiza la interaccion de una particula con un conjunto de ellas. Bound-Fluid/Float
//------------------------------------------------------------------------------
template<TpKernel tker,TpFtMode ftmode,bool symm>
__device__ void KerInteractionForcesBoundBox_NN
(unsigned p1,const unsigned &pini,const unsigned &pfin
  ,const float *ftomassp
  ,const float4 *poscell,const float4 *velrhop,const typecode *code,const unsigned* idp
  ,float massf,const float4 &pscellp1,const float4 &velrhop1,float &arp1,float &visc)
{
  for(int p2=pini; p2<pfin; p2++) {
    const float4 pscellp2=poscell[p2];
    float drx=pscellp1.x-pscellp2.x+CTE.poscellsize*(CEL_GetX(__float_as_int(pscellp1.w))-CEL_GetX(__float_as_int(pscellp2.w)));
    float dry=pscellp1.y-pscellp2.y+CTE.poscellsize*(CEL_GetY(__float_as_int(pscellp1.w))-CEL_GetY(__float_as_int(pscellp2.w)));
    float drz=pscellp1.z-pscellp2.z+CTE.poscellsize*(CEL_GetZ(__float_as_int(pscellp1.w))-CEL_GetZ(__float_as_int(pscellp2.w)));
    if(symm)dry=pscellp1.y+pscellp2.y+CTE.poscellsize*CEL_GetY(__float_as_int(pscellp2.w)); //<vs_syymmetry>
    const float rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO) {
      //-Computes kernel.
      const float fac=cufsph::GetKernel_Fac<tker>(rr2);
      const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.

      float4 velrhop2=velrhop[p2];
      if(symm)velrhop2.y=-velrhop2.y; //<vs_syymmetry>

      //-Obtains particle mass p2 if there are floating bodies.
      //-Obtiene masa de particula p2 en caso de existir floatings.
      float ftmassp2;    //-Contains mass of floating body or massf if fluid. | Contiene masa de particula floating o massf si es fluid.
      bool compute=true; //-Deactivated when DEM is used and is float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
      if(USE_FLOATING) {
        const typecode cod=code[p2];
        bool ftp2=CODE_IsFloating(cod);
        ftmassp2=(ftp2 ? ftomassp[CODE_GetTypeValue(cod)] : massf);
        compute=!(USE_FTEXTERNAL && ftp2); //-Deactivated when DEM or Chrono is used and is bound-float. | Se desactiva cuando se usa DEM o Chrono y es bound-float.
      }

      if(compute) {
        //-Density derivative (Continuity equation).
        const float dvx=velrhop1.x-velrhop2.x,dvy=velrhop1.y-velrhop2.y,dvz=velrhop1.z-velrhop2.z;
        arp1+=(USE_FLOATING ? ftmassp2 : massf)*(dvx*frx+dvy*fry+dvz*frz)*(velrhop1.w/velrhop2.w);

        {//===== Viscosity ===== 
          const float dot=drx*dvx+dry*dvy+drz*dvz;
          const float dot_rr2=dot/(rr2+CTE.eta2);
          visc=max(dot_rr2,visc);
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
/// Particle interaction for non-Newtonian models. Bound-Fluid/Float 
/// Realiza interaccion entre particulas para modelos no-Newtonianos. Bound-Fluid/Float
//------------------------------------------------------------------------------
template<TpKernel tker,TpFtMode ftmode,bool symm>
__global__ void KerInteractionForcesBound_NN(unsigned n,unsigned pinit
  ,int scelldiv,int4 nc,int3 cellzero,const int2 *beginendcellfluid,const unsigned *dcell
  ,const float *ftomassp
  ,const float4 *poscell,const float4 *velrhop,const typecode *code,const unsigned *idp
  ,float *viscdt,float *ar)
{
  const unsigned p=blockIdx.x*blockDim.x+threadIdx.x; //-Number of thread.
  if(p<n) {
    const unsigned p1=p+pinit;      //-Number of particle.
    float visc=0,arp1=0;

    //-Loads particle p1 data.
    const float4 pscellp1=poscell[p1];
    const float4 velrhop1=velrhop[p1];
    const bool rsymp1=(symm && CEL_GetPartY(__float_as_uint(pscellp1.w))==0); //<vs_syymmetry>

    //-Obtains neighborhood search limits.
    int ini1,fin1,ini2,fin2,ini3,fin3;
    cunsearch::InitCte(dcell[p1],scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

    //-Boundary-Fluid interaction.
    for(int c3=ini3; c3<fin3; c3+=nc.w)for(int c2=ini2; c2<fin2; c2+=nc.x) {
      unsigned pini,pfin=0; cunsearch::ParticleRange(c2,c3,ini1,fin1,beginendcellfluid,pini,pfin);
      if(pfin) {
        KerInteractionForcesBoundBox_NN<tker,ftmode,false>(p1,pini,pfin,ftomassp,poscell,velrhop,code,idp,CTE.massf,pscellp1,velrhop1,arp1,visc);
        if(symm && rsymp1)KerInteractionForcesBoundBox_NN<tker,ftmode,true >(p1,pini,pfin,ftomassp,poscell,velrhop,code,idp,CTE.massf,pscellp1,velrhop1,arp1,visc);
      }
    }
    //-Stores results.
    if(arp1||visc) {
      ar[p1]+=arp1;
      if(visc>viscdt[p1])viscdt[p1]=visc;
    }
  }
}
//======================Start of FDA approach===================================
//------------------------------------------------------------------------------
/// Interaction of a particle with a set of particles for non-Newtonian models using the FDA approach. (Fluid/Float-Fluid/Float/Bound)
/// Realiza la interaccion de una particula con un conjunto de ellas para modelos no Newtonianos que utilizan el enfoque de la FDA. (Fluid/Float-Fluid/Float/Bound)
//------------------------------------------------------------------------------
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,TpDensity tdensity,bool shift,bool symm>
__device__ void KerInteractionForcesFluidBox_FDA(bool boundp2,unsigned p1
  ,const unsigned &pini,const unsigned &pfin,float visco,float *visco_eta
  ,const float *ftomassp,float2 *tauff
  ,const float4 *poscell,const float4 *velrhop,const typecode *code,const unsigned *idp
  ,float massp2,const typecode pp1,bool ftp1
  ,const float4 &pscellp1,const float4 &velrhop1,float pressp1
  ,float2 &taup1_xx_xy,float2 &taup1_xz_yy,float2 &taup1_yz_zz
  ,float2 &grap1_xx_xy,float2 &grap1_xz_yy,float2 &grap1_yz_zz
  ,float3 &acep1,float &arp1,float &visc,float &visceta,float &visco_etap1,float &deltap1
  ,TpShifting shiftmode,float4 &shiftposfsp1)
{
  for(int p2=pini; p2<pfin; p2++){
    const float4 pscellp2=poscell[p2];
    float drx=pscellp1.x-pscellp2.x+CTE.poscellsize*(CEL_GetX(__float_as_int(pscellp1.w))-CEL_GetX(__float_as_int(pscellp2.w)));
    float dry=pscellp1.y-pscellp2.y+CTE.poscellsize*(CEL_GetY(__float_as_int(pscellp1.w))-CEL_GetY(__float_as_int(pscellp2.w)));
    float drz=pscellp1.z-pscellp2.z+CTE.poscellsize*(CEL_GetZ(__float_as_int(pscellp1.w))-CEL_GetZ(__float_as_int(pscellp2.w)));
    if(symm)dry=pscellp1.y+pscellp2.y+CTE.poscellsize*CEL_GetY(__float_as_int(pscellp2.w)); //<vs_syymmetry>
    const float rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO){
      //-Computes kernel.
      const float fac=cufsph::GetKernel_Fac<tker>(rr2);
      const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.

      //-Obtains mass of particle p2 for NN and if any floating bodies exist.
      const typecode cod=code[p2];
      const typecode pp2=(boundp2 ? pp1 : CODE_GetTypeValue(cod)); //<vs_non-Newtonian>
      float massp2=(boundp2 ? CTE.massb : PHASEARRAY[pp2].mass); //massp2 not neccesary to go in _Box function
      //Note if you masses are very different more than a ratio of 1.3 then: massp2 = (boundp2 ? PHASEARRAY[pp1].mass : PHASEARRAY[pp2].mass);

      //-Obtiene masa de particula p2 en caso de existir floatings.
      bool ftp2=false;         //-Indicates if it is floating. | Indica si es floating.
      float ftmassp2;    //-Contains mass of floating body or massf if fluid. | Contiene masa de particula floating o massp2 si es bound o fluid.
      bool compute=true; //-Deactivated when DEM is used and is float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
      if(USE_FLOATING) {
        ftp2=CODE_IsFloating(cod);
        ftmassp2=(ftp2 ? ftomassp[CODE_GetTypeValue(cod)] : massp2);
#ifdef DELTA_HEAVYFLOATING
        if(ftp2 && tdensity==DDT_DDT && ftmassp2<=(massp2*1.2f))deltap1=FLT_MAX;
#else
        if(ftp2 && tdensity==DDT_DDT)deltap1=FLT_MAX;
#endif
        if(ftp2 && shift && shiftmode==SHIFT_NoBound)shiftposfsp1.x=FLT_MAX; //-Cancels shifting with floating bodies. | Con floatings anula shifting.
        compute=!(USE_FTEXTERNAL && ftp1&&(boundp2||ftp2)); //-Deactivated when DEM or Chrono is used and is float-float or float-bound. | Se desactiva cuando se usa DEM o Chrono y es float-float o float-bound.
      }

      float4 velrhop2=velrhop[p2];
      if(symm)velrhop2.y=-velrhop2.y; //<vs_syymmetry>

      //===== Aceleration ===== 
      if(compute) {
        const float pressp2=cufsph::ComputePressCte_NN(velrhop2.w,PHASEARRAY[pp2].rho,PHASEARRAY[pp2].CteB,PHASEARRAY[pp2].Gamma,PHASEARRAY[pp2].Cs0,cod);
        const float prs=(pressp1+pressp2)/(velrhop1.w*velrhop2.w)
          +(tker==KERNEL_Cubic ? cufsph::GetKernelCubic_Tensil(rr2,velrhop1.w,pressp1,velrhop2.w,pressp2) : 0);
        const float p_vpm=-prs*(USE_FLOATING ? ftmassp2 : massp2);
        acep1.x+=p_vpm*frx; acep1.y+=p_vpm*fry; acep1.z+=p_vpm*frz;
      }

      //-Density derivative (Continuity equation).
      float dvx=velrhop1.x-velrhop2.x,dvy=velrhop1.y-velrhop2.y,dvz=velrhop1.z-velrhop2.z;
      if(compute)arp1+=(USE_FLOATING ? ftmassp2 : massp2)*(dvx*frx+dvy*fry+dvz*frz)*(velrhop1.w/velrhop2.w);

      const float cbar=max(PHASEARRAY[pp1].Cs0,PHASEARRAY[pp2].Cs0);
      const float dot3=(tdensity!=DDT_None||shift ? drx*frx+dry*fry+drz*frz : 0);
      //-Density derivative (DeltaSPH Molteni).
      if(tdensity==DDT_DDT && deltap1!=FLT_MAX) {
        const float rhop1over2=velrhop1.w/velrhop2.w;
        const float visc_densi=CTE.ddtkh*cbar*(rhop1over2-1.f)/(rr2+CTE.eta2);
        const float delta=(pp1==pp2 ? visc_densi*dot3*(USE_FLOATING ? ftmassp2 : massp2) : 0); //<vs_non-Newtonian>
        //deltap1=(boundp2? FLT_MAX: deltap1+delta);
        deltap1=(boundp2 && CTE.tboundary==BC_DBC ? FLT_MAX : deltap1+delta);
      }
      //-Density Diffusion Term (Fourtakas et al 2019). //<vs_dtt2_ini>
      if((tdensity==DDT_DDT2||(tdensity==DDT_DDT2Full&&!boundp2))&&deltap1!=FLT_MAX&&!ftp2) {
        const float rh=1.f+CTE.ddtgz*drz;
        const float drhop=CTE.rhopzero*pow(rh,1.f/CTE.gamma)-CTE.rhopzero;
        const float visc_densi=CTE.ddtkh*cbar*((velrhop2.w-velrhop1.w)-drhop)/(rr2+CTE.eta2);
        const float delta=(pp1==pp2 ? visc_densi*dot3*massp2/velrhop2.w : 0); //<vs_non-Newtonian>
        deltap1=(boundp2 ? FLT_MAX : deltap1-delta);
      } //<vs_dtt2_end>		

      //-Shifting correction.
      if(shift && shiftposfsp1.x!=FLT_MAX) {
        bool heavyphase=(PHASEARRAY[pp1].mass>PHASEARRAY[pp2].mass && pp1!=pp2 ? true : false); //<vs_non-Newtonian>
        const float massrhop=(USE_FLOATING ? ftmassp2 : massp2)/velrhop2.w;
        const bool noshift=(boundp2&&(shiftmode==SHIFT_NoBound||(shiftmode==SHIFT_NoFixed && CODE_IsFixed(code[p2]))));
        shiftposfsp1.x=(noshift ? FLT_MAX : (heavyphase ? 0 : shiftposfsp1.x+massrhop*frx)); //-Removes shifting for the boundaries. | Con boundary anula shifting.
        shiftposfsp1.y+=(heavyphase ? 0 : massrhop*fry);
        shiftposfsp1.z+=(heavyphase ? 0 : massrhop*frz);
        shiftposfsp1.w-=(heavyphase ? 0 : massrhop*dot3);
      }

      //===== Viscosity ===== 
      if(compute) {
        const float dot=drx*dvx+dry*dvy+drz*dvz;
        const float dot_rr2=dot/(rr2+CTE.eta2);
        visc=max(dot_rr2,visc);  //ViscDt=max(dot/(rr2+Eta2),ViscDt);
        //<vs_non-Newtonian>
        const float visco_NN=PHASECTE[pp2].visco;
        if(tvisco==VISCO_Artificial) {//-Artificial viscosity.
          if(dot<0) {
            const float amubar=CTE.kernelh*dot_rr2;  //amubar=CTE.kernelh*dot/(rr2+CTE.eta2);
            const float robar=(velrhop1.w+velrhop2.w)*0.5f;
            const float pi_visc=(-visco_NN*cbar*amubar/robar)*(USE_FLOATING ? ftmassp2 : massp2);
            acep1.x-=pi_visc*frx; acep1.y-=pi_visc*fry; acep1.z-=pi_visc*frz;
          }
        }
        else if(tvisco==VISCO_LaminarSPS||tvisco==VISCO_ConstEq) {
          {
            //vel gradients
            if(boundp2) { //this applies no slip on stress tensor
              dvx=2.f*velrhop1.x; dvy=2.f*velrhop1.y; dvz=2.f*velrhop1.z;  //fomraly I should use the moving BC vel as ug=2ub-uf
            }
            tmatrix3f dvelp1; float div_vel;
            GetVelocityGradients_FDA(rr2,drx,dry,drz,dvx,dvy,dvz,dvelp1,div_vel);

            //Strain rate tensor 
            tmatrix3f D_tensor; float div_D_tensor; float D_tensor_magn;
            float I_D,II_D; float J1_D,J2_D;
            GetStrainRateTensor(dvelp1,div_vel,I_D,II_D,J1_D,J2_D,div_D_tensor,D_tensor_magn,D_tensor);

            //Effective viscosity
            float m_NN=PHASECTE[pp2].m_NN; float n_NN=PHASECTE[pp2].n_NN; float tau_yield=PHASECTE[pp2].tau_yield;
            KerGetEta_Effective(pp1,tau_yield,D_tensor_magn,visco_NN,m_NN,n_NN,visco_etap1);
            visceta=max(visceta,visco_etap1);

            if(tvisco==VISCO_LaminarSPS){ //-Laminar contribution.
              //Morris Operator
              const float temp=2.f*(visco_etap1)/((rr2+CTE.eta2)*velrhop2.w);  //-Note this is the Morris operator and not Lo and Shao
              const float vtemp=(USE_FLOATING ? ftmassp2 : massp2)*temp*(drx*frx+dry*fry+drz*frz);
              acep1.x+=vtemp*dvx; acep1.y+=vtemp*dvy; acep1.z+=vtemp*dvz;

            }
            else if(tvisco==VISCO_ConstEq) {
              //stress tensor tau 
              tmatrix3f tau_tensor; float tau_tensor_magn;
              float I_t,II_t; float J1_t,J2_t;
              GetStressTensor(D_tensor,visco_etap1,I_t,II_t,J1_t,J2_t,tau_tensor_magn,tau_tensor);

              //viscous forces
              float taux=(tau_tensor.a11*frx+tau_tensor.a12*fry+tau_tensor.a13*frz)/(velrhop2.w); //Morris 1997
              float tauy=(tau_tensor.a21*frx+tau_tensor.a22*fry+tau_tensor.a23*frz)/(velrhop2.w);
              float tauz=(tau_tensor.a31*frx+tau_tensor.a32*fry+tau_tensor.a33*frz)/(velrhop2.w);
              const float mtemp=(USE_FLOATING ? ftmassp2 : massp2);
              acep1.x+=taux*mtemp; acep1.y+=tauy*mtemp; acep1.z+=tauz*mtemp;
            }
          }
          //-SPS turbulence model.
          //-SPS turbulence model is disabled in v5.0 NN version
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
/// Interaction between particles for non-Newtonian models using the FDA approach. Fluid/Float-Fluid/Float or Fluid/Float-Bound.
/// Includes artificial/laminar/Const Eq. viscosity and normal/DEM floating bodies.
///
/// Realiza interaccion entre particulas para modelos no-Newtonianos que utilizan el enfoque de la FDA. Fluid/Float-Fluid/Float or Fluid/Float-Bound
/// Incluye visco artificial/laminar y floatings normales/dem.
//------------------------------------------------------------------------------
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,TpDensity tdensity,bool shift,bool symm>
__global__ void KerInteractionForcesFluid_NN_FDA(unsigned n,unsigned pinit,float viscob,float viscof,float *visco_eta
  ,int scelldiv,int4 nc,int3 cellzero,const int2 *begincell,unsigned cellfluid,const unsigned *dcell
  ,const float *ftomassp,float2 *tauff,float2 *gradvelff
  ,const float4 *poscell,const float4 *velrhop
  ,const typecode *code,const unsigned *idp
  ,float *viscdt,float *viscetadt,float *ar,float3 *ace,float *delta
  ,TpShifting shiftmode,float4 *shiftposfs)
{
  const unsigned p=blockIdx.x*blockDim.x+threadIdx.x; //-Number of particle.
  if(p<n) {
    unsigned p1=p+pinit;      //-Number of particle.
    float visc=0,arp1=0,deltap1=0;
    float3 acep1=make_float3(0,0,0);

    //-Variables for Shifting.
    float4 shiftposfsp1;
    if(shift)shiftposfsp1=shiftposfs[p1];

    //-Obtains data of particle p1 in case there are floating bodies.		
    bool ftp1;       //-Indicates if it is floating. | Indica si es floating.
    const typecode cod=code[p1];
    if(USE_FLOATING) {
      ftp1=CODE_IsFloating(cod);
      if(ftp1 && tdensity!=DDT_None)deltap1=FLT_MAX; //-DDT is not applied to floating particles.
      if(ftp1 && shift)shiftposfsp1.x=FLT_MAX; //-Shifting is not calculated for floating bodies. | Para floatings no se calcula shifting.
    }

    //-Obtains basic data of particle p1.
    const float4 pscellp1=poscell[p1];
    const float4 velrhop1=velrhop[p1];
    //<vs_non-Newtonian>
    const typecode pp1=CODE_GetTypeValue(cod);
    float visco_etap1=0;
    float visceta=0;

    //Obtain pressure		
    const float pressp1=cufsph::ComputePressCte_NN(velrhop1.w,PHASEARRAY[pp1].rho,PHASEARRAY[pp1].CteB,PHASEARRAY[pp1].Gamma,PHASEARRAY[pp1].Cs0,cod);
    const bool rsymp1=(symm && CEL_GetPartY(__float_as_uint(pscellp1.w))==0); //<vs_syymmetry>

    //-Variables for Laminar+SPS.
    float2 taup1_xx_xy,taup1_xz_yy,taup1_yz_zz;
    if(tvisco!=VISCO_Artificial) {
      taup1_xx_xy=tauff[p1*3];
      taup1_xz_yy=tauff[p1*3+1];
      taup1_yz_zz=tauff[p1*3+2];
    }
    //-Variables for Laminar+SPS (computation).
    float2 grap1_xx_xy,grap1_xz_yy,grap1_yz_zz;
    if(tvisco!=VISCO_Artificial) {
      grap1_xx_xy=make_float2(0,0);
      grap1_xz_yy=make_float2(0,0);
      grap1_yz_zz=make_float2(0,0);
    }

    //-Obtains neighborhood search limits.
    int ini1,fin1,ini2,fin2,ini3,fin3;
    cunsearch::InitCte(dcell[p1],scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

    //-Interaction with fluids.
    ini3+=cellfluid; fin3+=cellfluid;
    for(int c3=ini3; c3<fin3; c3+=nc.w)for(int c2=ini2; c2<fin2; c2+=nc.x){
      unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
      if(pfin){
        KerInteractionForcesFluidBox_FDA<tker,ftmode,tvisco,tdensity,shift,false>(false,p1,pini,pfin,viscof,visco_eta,ftomassp,tauff,poscell,velrhop,code,idp,CTE.massf,pp1,ftp1,pscellp1,velrhop1,pressp1,taup1_xx_xy,taup1_xz_yy,taup1_yz_zz,grap1_xx_xy,grap1_xz_yy,grap1_yz_zz,acep1,arp1,visc,visceta,visco_etap1,deltap1,shiftmode,shiftposfsp1);
        if(symm && rsymp1)	KerInteractionForcesFluidBox_FDA<tker,ftmode,tvisco,tdensity,shift,true >(false,p1,pini,pfin,viscof,visco_eta,ftomassp,tauff,poscell,velrhop,code,idp,CTE.massf,pp1,ftp1,pscellp1,velrhop1,pressp1,taup1_xx_xy,taup1_xz_yy,taup1_yz_zz,grap1_xx_xy,grap1_xz_yy,grap1_yz_zz,acep1,arp1,visc,visceta,visco_etap1,deltap1,shiftmode,shiftposfsp1); //<vs_syymmetry>
      }
    }
    //-Interaction with boundaries.
    ini3-=cellfluid; fin3-=cellfluid;
    for(int c3=ini3; c3<fin3; c3+=nc.w)for(int c2=ini2; c2<fin2; c2+=nc.x){
      unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
      if(pfin){
        KerInteractionForcesFluidBox_FDA<tker,ftmode,tvisco,tdensity,shift,false>(true,p1,pini,pfin,viscob,visco_eta,ftomassp,tauff,poscell,velrhop,code,idp,CTE.massf,pp1,ftp1,pscellp1,velrhop1,pressp1,taup1_xx_xy,taup1_xz_yy,taup1_yz_zz,grap1_xx_xy,grap1_xz_yy,grap1_yz_zz,acep1,arp1,visc,visceta,visco_etap1,deltap1,shiftmode,shiftposfsp1);
        if(symm && rsymp1)	KerInteractionForcesFluidBox_FDA<tker,ftmode,tvisco,tdensity,shift,true >(true,p1,pini,pfin,viscob,visco_eta,ftomassp,tauff,poscell,velrhop,code,idp,CTE.massf,pp1,ftp1,pscellp1,velrhop1,pressp1,taup1_xx_xy,taup1_xz_yy,taup1_yz_zz,grap1_xx_xy,grap1_xz_yy,grap1_yz_zz,acep1,arp1,visc,visceta,visco_etap1,deltap1,shiftmode,shiftposfsp1);  //<vs_syymmetry>
      }
    }
    //-Stores results.
    if(shift||arp1||acep1.x||acep1.y||acep1.z||visc||visceta||visco_etap1) {
      if(tdensity!=DDT_None) {
        if(delta) {
          const float rdelta=delta[p1];
          delta[p1]=(rdelta==FLT_MAX||deltap1==FLT_MAX ? FLT_MAX : rdelta+deltap1);
        }
        else if(deltap1!=FLT_MAX)arp1+=deltap1;
      }
      ar[p1]+=arp1;
      float3 r=ace[p1]; r.x+=acep1.x; r.y+=acep1.y; r.z+=acep1.z; ace[p1]=r;
      if(visc>viscdt[p1])viscdt[p1]=visc;
      if(visceta>viscetadt[p1])viscetadt[p1]=visceta;
      if(tvisco==VISCO_LaminarSPS) {
        float2 rg;
        rg=gradvelff[p1*3];		 rg=make_float2(rg.x+grap1_xx_xy.x,rg.y+grap1_xx_xy.y);  gradvelff[p1*3]=rg;
        rg=gradvelff[p1*3+1];  rg=make_float2(rg.x+grap1_xz_yy.x,rg.y+grap1_xz_yy.y);  gradvelff[p1*3+1]=rg;
        rg=gradvelff[p1*3+2];  rg=make_float2(rg.x+grap1_yz_zz.x,rg.y+grap1_yz_zz.y);  gradvelff[p1*3+2]=rg;
      }
      if(shift)shiftposfs[p1]=shiftposfsp1;
      //auxnn[p1] = visco_etap1; //to be used if an auxilary is needed for debug or otherwise.
    }
  }
}

//==============================================================================
/// Interaction for the force computation for non-Newtonian models using the FDA approach.
/// Interaccion para el calculo de fuerzas para modelos no-Newtonianos que utilizan el enfoque de la FDA.
//==============================================================================
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,TpDensity tdensity,bool shift>
void Interaction_ForcesGpuT_NN_FDA(const StInterParmsg &t)
{
  //-Collects kernel information.
#ifndef DISABLE_BSMODES
  if(t.kerinfo) {
    cusph::Interaction_ForcesT_KerInfo<tker,ftmode,true,tdensity,shift,false>(t.kerinfo);
    return;
  }
#endif
  const StDivDataGpu &dvd=t.divdatag;
  const int2* beginendcell=dvd.beginendcell;
  //-Interaction Fluid-Fluid & Fluid-Bound.
  if(t.fluidnum) {
    dim3 sgridf=GetSimpleGridSize(t.fluidnum,t.bsfluid);
    if(t.symmetry) //<vs_syymmetry_ini>
      KerInteractionForcesFluid_NN_FDA<tker,ftmode,tvisco,tdensity,shift,true ><<<sgridf,t.bsfluid,0,t.stm>>>
      (t.fluidnum,t.fluidini,t.viscob,t.viscof,t.visco_eta,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell,dvd.cellfluid,t.dcell
        ,t.ftomassp,(float2*)t.tau,(float2*)t.gradvel,t.poscell,t.velrhop,t.code,t.idp
        ,t.viscdt,t.viscetadt,t.ar,t.ace,t.delta,t.shiftmode,t.shiftposfs);
    else //<vs_syymmetry_end>
      KerInteractionForcesFluid_NN_FDA<tker,ftmode,tvisco,tdensity,shift,false><<<sgridf,t.bsfluid,0,t.stm>>>
      (t.fluidnum,t.fluidini,t.viscob,t.viscof,t.visco_eta,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell,dvd.cellfluid,t.dcell
        ,t.ftomassp,(float2*)t.tau,(float2*)t.gradvel,t.poscell,t.velrhop,t.code,t.idp
        ,t.viscdt,t.viscetadt,t.ar,t.ace,t.delta,t.shiftmode,t.shiftposfs);
  }
  //-Interaction Boundary-Fluid.
  if(t.boundnum) {
    const int2* beginendcellfluid=dvd.beginendcell+dvd.cellfluid;
    dim3 sgridb=GetSimpleGridSize(t.boundnum,t.bsbound);
    //printf("bsbound:%u\n",bsbound);
    if(t.symmetry) //<vs_syymmetry_ini>
      KerInteractionForcesBound_NN<tker,ftmode,true ><<<sgridb,t.bsbound,0,t.stm>>>
      (t.boundnum,t.boundini,dvd.scelldiv,dvd.nc,dvd.cellzero,beginendcell+dvd.cellfluid,t.dcell
        ,t.ftomassp,t.poscell,t.velrhop,t.code,t.idp,t.viscdt,t.ar);
    else //<vs_syymmetry_end>
      KerInteractionForcesBound_NN<tker,ftmode,false><<<sgridb,t.bsbound,0,t.stm>>>
      (t.boundnum,t.boundini,dvd.scelldiv,dvd.nc,dvd.cellzero,beginendcellfluid,t.dcell
        ,t.ftomassp,t.poscell,t.velrhop,t.code,t.idp,t.viscdt,t.ar);
  }
}
//======================END of FDA==============================================

//======================Start of SPH============================================
//------------------------------------------------------------------------------
/// Interaction of a particle with a set of particles for non-Newtonian models using the SPH approach with Const Eq. (Fluid/Float-Fluid/Float/Bound)
/// Realiza la interaccion de una particula con un conjunto de ellas para modelos no-Newtonianos que utilizan el enfoque de la SPH Const. Eq. (Fluid/Float-Fluid/Float/Bound)
//------------------------------------------------------------------------------
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,bool symm>
__device__ void KerInteractionForcesFluidBox_SPH_ConsEq(bool boundp2,unsigned p1
  ,const unsigned &pini,const unsigned &pfin,float visco,float *visco_eta
  ,const float *ftomassp,float2 *tauff
  ,const float4 *poscell,const float4 *velrhop
  ,const typecode *code,const unsigned *idp
  ,const typecode pp1,bool ftp1
  ,const float4 &pscellp1,const float4 &velrhop1
  ,float2 &taup1_xx_xy,float2 &taup1_xz_yy,float2 &taup1_yz_zz
  ,float3 &acep1,float &visc,float &visco_etap1)
{
  for(int p2=pini; p2<pfin; p2++) {
    const float4 pscellp2=poscell[p2];
    float drx=pscellp1.x-pscellp2.x+CTE.poscellsize*(CEL_GetX(__float_as_int(pscellp1.w))-CEL_GetX(__float_as_int(pscellp2.w)));
    float dry=pscellp1.y-pscellp2.y+CTE.poscellsize*(CEL_GetY(__float_as_int(pscellp1.w))-CEL_GetY(__float_as_int(pscellp2.w)));
    float drz=pscellp1.z-pscellp2.z+CTE.poscellsize*(CEL_GetZ(__float_as_int(pscellp1.w))-CEL_GetZ(__float_as_int(pscellp2.w)));
    if(symm)dry=pscellp1.y+pscellp2.y+CTE.poscellsize*CEL_GetY(__float_as_int(pscellp2.w)); //<vs_syymmetry>
    const float rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO) {
      //-Computes kernel.
      const float fac=cufsph::GetKernel_Fac<tker>(rr2);
      const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.

      //-Obtains mass of particle p2 for NN and if any floating bodies exist.
      const typecode cod=code[p2];
      const typecode pp2=(boundp2 ? pp1 : CODE_GetTypeValue(cod)); //<vs_non-Newtonian>
      float massp2=(boundp2 ? CTE.massb : PHASEARRAY[pp2].mass); //massp2 not neccesary to go in _Box function
      //Note if you masses are very different more than a ratio of 1.3 then: massp2 = (boundp2 ? PHASEARRAY[pp1].mass : PHASEARRAY[pp2].mass);

      //-Obtiene masa de particula p2 en caso de existir floatings.
      bool ftp2=false;         //-Indicates if it is floating. | Indica si es floating.
      float ftmassp2;    //-Contains mass of floating body or massf if fluid. | Contiene masa de particula floating o massp2 si es bound o fluid.
      bool compute=true; //-Deactivated when DEM is used and is float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
      if(USE_FLOATING) {
        const typecode cod=code[p2];
        ftp2=CODE_IsFloating(cod);
        ftmassp2=(ftp2 ? ftomassp[CODE_GetTypeValue(cod)] : massp2);
        compute=!(USE_FTEXTERNAL && ftp1&&(boundp2||ftp2)); //-Deactivated when DEM or Chrono is used and is float-float or float-bound. | Se desactiva cuando se usa DEM o Chrono y es float-float o float-bound.
      }
      float4 velrhop2=velrhop[p2];
      if(symm)velrhop2.y=-velrhop2.y; //<vs_syymmetry>

      //-velocity dvx.
      const float dvx=velrhop1.x-velrhop2.x,dvy=velrhop1.y-velrhop2.y,dvz=velrhop1.z-velrhop2.z;
      const float cbar=max(PHASEARRAY[pp2].Cs0,PHASEARRAY[pp2].Cs0);

      //===== Viscosity ===== 
      if(compute) {
        const float dot=drx*dvx+dry*dvy+drz*dvz;
        const float dot_rr2=dot/(rr2+CTE.eta2);
        visc=max(dot_rr2,visc);  //ViscDt=max(dot/(rr2+Eta2),ViscDt);

        //<vs_non-Newtonian>				
        float2 tau_sum_xx_xy,tau_sum_xz_yy,tau_sum_yz_zz;
        float2 taup2_xx_xy=tauff[p2*3];
        float2 taup2_xz_yy=tauff[p2*3+1];
        float2 taup2_yz_zz=tauff[p2*3+2];
        //boundary particles only
        if(boundp2) {
          taup2_xx_xy=make_float2(taup1_xx_xy.x,taup1_xx_xy.y); // use (-) for slip and (+1) for no slip
          taup2_xz_yy=make_float2(taup1_xz_yy.x,taup1_xz_yy.y); //
          taup2_yz_zz=make_float2(taup1_yz_zz.x,taup1_yz_zz.y); //
        }

        tau_sum_xx_xy.x=taup1_xx_xy.x+taup2_xx_xy.x; tau_sum_xx_xy.y=taup1_xx_xy.y+taup2_xx_xy.y;	tau_sum_xz_yy.x=taup1_xz_yy.x+taup2_xz_yy.x;
        tau_sum_xz_yy.y=taup1_xz_yy.y+taup2_xz_yy.y;	tau_sum_yz_zz.x=taup1_yz_zz.x+taup2_yz_zz.x;
        tau_sum_yz_zz.y=taup1_yz_zz.y+taup2_yz_zz.y;

        float taux=(tau_sum_xx_xy.x*frx+tau_sum_xx_xy.y*fry+tau_sum_xz_yy.x*frz)/(velrhop2.w);
        float tauy=(tau_sum_xx_xy.y*frx+tau_sum_xz_yy.y*fry+tau_sum_yz_zz.x*frz)/(velrhop2.w);
        float tauz=(tau_sum_xz_yy.x*frx+tau_sum_yz_zz.x*fry+tau_sum_yz_zz.y*frz)/(velrhop2.w);
        //store stresses
        massp2=(USE_FLOATING ? ftmassp2 : massp2);
        acep1.x+=taux*massp2; acep1.y+=tauy*massp2; acep1.z+=tauz*massp2;
      }
    }
  }
}

template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,bool symm>
__device__ void KerInteractionForcesMultilayerGranularBox_SPH_ConsEq(bool boundp2,unsigned p1
  ,const unsigned &pini,const unsigned &pfin
  ,const float *ftomassp,float2 *tauff
  ,const float4 *poscell,const float4 *velrhop
  ,const typecode *code,const unsigned *idp
  ,const typecode pp1,bool ftp1
  ,const float4 &pscellp1,const float4 &velrhop1
  ,float2 &taup1_xx_xy,float2 &taup1_xz_yy,float2 &taup1_yz_zz
  ,float3 &acep1)
{
  for(int p2=pini; p2<pfin; p2++) {
    const float4 pscellp2=poscell[p2];
    float drx=pscellp1.x-pscellp2.x+CTE.poscellsize*(CEL_GetX(__float_as_int(pscellp1.w))-CEL_GetX(__float_as_int(pscellp2.w)));
    float dry=pscellp1.y-pscellp2.y+CTE.poscellsize*(CEL_GetY(__float_as_int(pscellp1.w))-CEL_GetY(__float_as_int(pscellp2.w)));
    float drz=pscellp1.z-pscellp2.z+CTE.poscellsize*(CEL_GetZ(__float_as_int(pscellp1.w))-CEL_GetZ(__float_as_int(pscellp2.w)));
    if(symm)dry=pscellp1.y+pscellp2.y+CTE.poscellsize*CEL_GetY(__float_as_int(pscellp2.w)); //<vs_syymmetry>
    const float rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO) {
      //-Computes kernel.
      const float fac=cufsph::GetKernel_Fac<tker>(rr2);
      const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.

      //-Obtains mass of particle p2 for NN and if any floating bodies exist.
      const typecode cod=code[p2];
      const typecode pp2=(boundp2 ? pp1 : CODE_GetTypeValue(cod)); //<vs_non-Newtonian>
      if(pp2 == 1){
          float massp2=(boundp2 ? CTE.massb : PhaseDruckerPrager[pp2].mass); 
          //Note if you masses are very different more than a ratio of 1.3 then: massp2 = (boundp2 ? PHASEARRAY[pp1].mass : PHASEARRAY[pp2].mass);

          //-Obtiene masa de particula p2 en caso de existir floatings.
          bool ftp2=false;         //-Indicates if it is floating. | Indica si es floating.
          float ftmassp2;    //-Contains mass of floating body or massf if fluid. | Contiene masa de particula floating o massp2 si es bound o fluid.
          bool compute=true; //-Deactivated when DEM is used and is float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
          if(USE_FLOATING) {
            const typecode cod=code[p2];
            ftp2=CODE_IsFloating(cod);
            ftmassp2=(ftp2 ? ftomassp[CODE_GetTypeValue(cod)] : massp2);
            compute=!(USE_FTEXTERNAL && ftp1&&(boundp2||ftp2)); //-Deactivated when DEM or Chrono is used and is float-float or float-bound. | Se desactiva cuando se usa DEM o Chrono y es float-float o float-bound.
          }

          float4 velrhop2=velrhop[p2];
          if(symm)velrhop2.y=-velrhop2.y; //<vs_syymmetry>

          if(compute) {
            //<vs_non-Newtonian>				
            float2 tau_sum_xx_xy,tau_sum_xz_yy,tau_sum_yz_zz;
            float2 taup2_xx_xy=tauff[p2*3];
            float2 taup2_xz_yy=tauff[p2*3+1];
            float2 taup2_yz_zz=tauff[p2*3+2];
            //boundary particles only
            //if(boundp2) {
            //  taup2_xx_xy=make_float2(taup1_xx_xy.x,taup1_xx_xy.y); // use (-) for slip and (+1) for no slip
            //  taup2_xz_yy=make_float2(taup1_xz_yy.x,taup1_xz_yy.y); //
            //  taup2_yz_zz=make_float2(taup1_yz_zz.x,taup1_yz_zz.y); //
            //}

            tau_sum_xx_xy.x=taup1_xx_xy.x/pow(velrhop1.w,2) + taup2_xx_xy.x/pow(velrhop2.w,2); 
            tau_sum_xx_xy.y=taup1_xx_xy.y/pow(velrhop1.w,2) + taup2_xx_xy.y/pow(velrhop2.w,2);	
            tau_sum_xz_yy.x=taup1_xz_yy.x/pow(velrhop1.w,2) + taup2_xz_yy.x/pow(velrhop2.w,2);
            tau_sum_xz_yy.y=taup1_xz_yy.y/pow(velrhop1.w,2) + taup2_xz_yy.y/pow(velrhop2.w,2);	
            tau_sum_yz_zz.x=taup1_yz_zz.x/pow(velrhop1.w,2) + taup2_yz_zz.x/pow(velrhop2.w,2);
            tau_sum_yz_zz.y=taup1_yz_zz.y/pow(velrhop1.w,2) + taup2_yz_zz.y/pow(velrhop2.w,2);

            float taux=tau_sum_xx_xy.x*frx+tau_sum_xx_xy.y*fry+tau_sum_xz_yy.x*frz;
            float tauy=tau_sum_xx_xy.y*frx+tau_sum_xz_yy.y*fry+tau_sum_yz_zz.x*frz;
            float tauz=tau_sum_xz_yy.x*frx+tau_sum_yz_zz.x*fry+tau_sum_yz_zz.y*frz;
            //store stresses
            massp2=(USE_FLOATING ? ftmassp2 : massp2);
            acep1.x+=taux*massp2; acep1.y+=tauy*massp2; acep1.z+=tauz*massp2;
          } 
      }
    }
  }
}
//------------------------------------------------------------------------------
/// Interaction between particles for non-Newtonian models using the SPH approach with Const. Eq. Fluid/Float-Fluid/Float or Fluid/Float-Bound.
/// Includes Const. Eq. viscosity and normal/DEM floating bodies que utilizan el enfoque de la SPH Const. Eq..
///
/// Realiza interaccion entre particulas. Fluid/Float-Fluid/Float or Fluid/Float-Bound
/// Incluye visco artificial/laminar y floatings normales/dem.
//------------------------------------------------------------------------------
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,bool symm>
__global__ void KerInteractionForcesFluid_NN_SPH_ConsEq(unsigned n,unsigned pinit,float viscob,float viscof,float *visco_eta
  ,int scelldiv,int4 nc,int3 cellzero,const int2 *begincell,unsigned cellfluid,const unsigned *dcell
  ,const float *ftomassp,float2 *tauff,float *auxnn,const float4 *poscell,const float4 *velrhop
  ,const typecode *code,const unsigned *idp,float3 *ace)
{
  const unsigned p=blockIdx.x*blockDim.x+threadIdx.x; //-Number of particle.
  if(p<n) {
    unsigned p1=p+pinit;      //-Number of particle.			
    float3 acep1=make_float3(0,0,0);
    float visc=0;

    //-Obtains data of particle p1 in case there are floating bodies.
    //-Obtiene datos de particula p1 en caso de existir floatings.
    bool ftp1;       //-Indicates if it is floating. | Indica si es floating.
    const typecode cod=code[p1];
    if(USE_FLOATING) {
      const typecode cod=code[p1];
      ftp1=CODE_IsFloating(cod);
    }

    //-Obtains basic data of particle p1.
    const float4 pscellp1=poscell[p1];
    const float4 velrhop1=velrhop[p1];
    const bool rsymp1=(symm && CEL_GetPartY(__float_as_uint(pscellp1.w))==0); //<vs_syymmetry>
    //<vs_non-Newtonian>
    const typecode pp1=CODE_GetTypeValue(cod);
    float visco_etap1=visco_eta[p1];

    //-Variables for tau.			
    float2 taup1_xx_xy=tauff[p1*3];
    float2 taup1_xz_yy=tauff[p1*3+1];
    float2 taup1_yz_zz=tauff[p1*3+2];

    //-Obtains neighborhood search limits.
    int ini1,fin1,ini2,fin2,ini3,fin3;
    cunsearch::InitCte(dcell[p1],scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

    //-Interaction with fluids.
    ini3+=cellfluid; fin3+=cellfluid;
    for(int c3=ini3; c3<fin3; c3+=nc.w)for(int c2=ini2; c2<fin2; c2+=nc.x) {
      unsigned pini,pfin=0; cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
      if(pfin) {
        if (TVisco != VISCO_SoilWater){
            KerInteractionForcesFluidBox_SPH_ConsEq<tker,ftmode,tvisco,false>(false,p1,pini,pfin,viscof,visco_eta,ftomassp,tauff,poscell,velrhop,code,idp,pp1,ftp1,pscellp1,velrhop1,taup1_xx_xy,taup1_xz_yy,taup1_yz_zz,acep1,visc,visco_etap1);
            if(symm && rsymp1)	KerInteractionForcesFluidBox_SPH_ConsEq<tker,ftmode,tvisco,true>(false,p1,pini,pfin,viscof,visco_eta,ftomassp,tauff,poscell,velrhop,code,idp,pp1,ftp1,pscellp1,velrhop1,taup1_xx_xy,taup1_xz_yy,taup1_yz_zz,acep1,visc,visco_etap1); //<vs_syymmetry>
        }
        if (TVisco == VISCO_SoilWater){
        //    if (pp1 == 0){
        //        KerInteractionForcesMultilayerFluidBox_SPH_ConsEq<tker,ftmode,tvisco,false>(false,p1,pini,pfin,viscof,visco_eta,ftomassp,tauff,poscell,velrhop,code,idp,pp1,ftp1,pscellp1,velrhop1,taup1_xx_xy,taup1_xz_yy,taup1_yz_zz,acep1,visc,visco_etap1);
        //        if(symm && rsymp1)	KerInteractionForcesMultilayerFluidBox_SPH_ConsEq<tker,ftmode,tvisco,true>(false,p1,pini,pfin,viscof,visco_eta,ftomassp,tauff,poscell,velrhop,code,idp,pp1,ftp1,pscellp1,velrhop1,taup1_xx_xy,taup1_xz_yy,taup1_yz_zz,acep1,visc,visco_etap1); //<vs_syymmetry>
        //    } 
            if (pp1 == 1){
                KerInteractionForcesMultilayerGranularBox_SPH_ConsEq<tker,ftmode,tvisco,false>(false,p1,pini,pfin,ftomassp,tauff,poscell,velrhop,code,idp,pp1,ftp1,pscellp1,velrhop1,taup1_xx_xy,taup1_xz_yy,taup1_yz_zz,acep1);
                if(symm && rsymp1)	KerInteractionForcesMultilayerGranularBox_SPH_ConsEq<tker,ftmode,tvisco,true>(false,p1,pini,pfin,ftomassp,tauff,poscell,velrhop,code,idp,pp1,ftp1,pscellp1,velrhop1,taup1_xx_xy,taup1_xz_yy,taup1_yz_zz,acep1); //<vs_syymmetry>
            }            
        }
    }
    //-Interaction with boundaries.
    ini3-=cellfluid; fin3-=cellfluid;
    for(int c3=ini3; c3<fin3; c3+=nc.w)for(int c2=ini2; c2<fin2; c2+=nc.x) {
      unsigned pini,pfin=0; cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
      if(pfin) {
        KerInteractionForcesFluidBox_SPH_ConsEq<tker,ftmode,tvisco,false>(true,p1,pini,pfin,viscob,visco_eta,ftomassp,tauff,poscell,velrhop,code,idp,pp1,ftp1,pscellp1,velrhop1,taup1_xx_xy,taup1_xz_yy,taup1_yz_zz,acep1,visc,visco_etap1);
        if(symm && rsymp1)	KerInteractionForcesFluidBox_SPH_ConsEq<tker,ftmode,tvisco,true>(true,p1,pini,pfin,viscob,visco_eta,ftomassp,tauff,poscell,velrhop,code,idp,pp1,ftp1,pscellp1,velrhop1,taup1_xx_xy,taup1_xz_yy,taup1_yz_zz,acep1,visc,visco_etap1); //<vs_syymmetry>
      }
    }

    //-Stores results.
    if(acep1.x||acep1.y||acep1.z) {
      float3 r=ace[p1]; r.x+=acep1.x; r.y+=acep1.y; r.z+=acep1.z; ace[p1]=r;
      //auxnn[p1] = visco_etap1; // to be used if an auxilary is needed.
      }
    } 
  }
}


//==============================================================================
/// Calculates the strain rate tensor and effective viscocity for each particle for non-Newtonian models.
/// Calcula el tensor de la velocidad de deformacion y la viscosidad efectiva para cada particula para modelos no-Newtonianos.
//==============================================================================
template<TpFtMode ftmode,TpVisco tvisco,bool symm>
__global__ void KerInteractionForcesFluid_NN_SPH_Visco_Stress_tensor(unsigned n,unsigned pinit,float *visco_eta
  ,int scelldiv,int4 nc,int3 cellzero,const int2 *begincell,unsigned cellfluid,const unsigned *dcell
  ,const float *ftomassp,float2 *tauff,float2 *pstrain,float2 *d_tensorff,float2 *gradvelff,float *auxnn,const float4 *poscell,const float4 *velrhop
  ,const typecode *code,const unsigned *idp, double dt)
{
  const unsigned p=blockIdx.x*blockDim.x+threadIdx.x; //-Number of particle.
  if(p<n) {
    unsigned p1=p+pinit;      //-Number of particle.
    const typecode cod=code[p1];
    const typecode pp1=CODE_GetTypeValue(cod);
    //<vs_non-Newtonian>
    if(tvisco != VISCO_SoilWater)float visco_etap1=visco_eta[p1];;

    float2 taup1_xx_xy=make_float2(0,0);
    float2 taup1_xz_yy=make_float2(0,0);
    float2 taup1_yz_zz=make_float2(0,0);
    float I_t,II_t; float J1_t,J2_t; float tau_tensor_magn;
    if(tvisco != VISCO_SoilWater) GetStressTensor_sym(dtsrp1_xx_xy,dtsrp1_xz_yy,dtsrp1_yz_zz,visco_etap1,I_t,II_t,J1_t,J2_t,tau_tensor_magn,taup1_xx_xy,taup1_xz_yy,taup1_yz_zz);
    if(tvisco == VISCO_SoilWater){
        float2 grap1_xx_xy,grap1_xz_yy,grap1_yz_zz;
        grap1_xx_xy=gradvelff[p1*3];
        grap1_xz_yy=gradvelff[p1*3+1];
        grap1_yz_zz=gradvelff[p1*3+2];

        //Strain rate tensor, for symmetric tensor, they are the same?
        float2 dtsrp1_xx_xy, dtsrp1_xz_yy, dtsrp1_yz_zz;
        dtsrp1_xx_xy.x = grap1_xx_xy.x; dtsrp1_xx_xy.y = grap1_xx_xy.y; grap1_xz_yy.x = dtsrp1_xz_yy.x;
        dtsrp1_xz_yy.y = grap1_xz_yy.y; dtsrp1_yz_zz.x = grap1_yz_zz.x;
        dtsrp1_yz_zz.y = grap1_yz_zz.y;

        //Spin rate tensor
        float3 dtspinratep1 =  make_float3(0,0,0);
        GetStrainSpinRateTensor(grap1_xx_xy,grap1_xz_yy,grap1_yz_zz,dtsrp1_xx_xy,dtsrp1_xz_yy,dtsrp1_yz_zz,dtspinratep1); //No need for symmetric tensor?
        //if (pp1 == 0){
        //  const float visco_etap1 = PHASECTE[pp1].visco;
        //  GetStressTensorMultilayerFluid_sym(dtsrp1_xx_xy,dtsrp1_xz_yy,dtsrp1_yz_zz,visco_etap1,taup1_xx_xy,taup1_xz_yy,taup1_yz_zz);
        //}
        if(pp1 == 1) {
          const float DP_K = PHASEDRUCKERPRAGER[pp1].DP_K; ///<  Elastic bulk modulus
          const float DP_G = PHASEDRUCKERPRAGER[pp1].DP_G;    ///< Elastic shear modulus
          const float MC_phi = PHASEDRUCKERPRAGER[pp1].MC_phi;    ///< Friction angle in MC model, to be converted to DP yield surface parameters DP_AlphaPhi and DP_kc
          const float MC_c = PHASEDRUCKERPRAGER[pp1].MC_c;    ///< Cohesion in MC model, to be converted to DP yield surface parameters DP_AlphaPhi and DP_kc
          const float MC_psi = PHASEDRUCKERPRAGER[pp1].MC_psi;    ///< Dilatancy angle in MC model, to be converted to DP non-associate flow rule parameter DP_psi
          GetStressTensorMultilayerSoil_sym(dtsrp1_xx_xy, dtsrp1_xz_yy, dtsrp1_yz_zz, dtspinratep1, taup1_xx_xy, taup1_xz_yy, taup1_yz_zz, 
                                                       Dpp1_xx_xy, Dpp1_xz_yy, Dpp1_yz_zz, DP_K, DP_G, MC_AlphaPhi, MC_c, MC_psi, dt);
        pstrain[p1*3]=make_float2(Dpp1_xx_xy.x,Dpp1_xx_xy.y); // plastic strain component
        pstrain[p1*3+1]=make_float2(Dpp1_xz_yy.x,Dpp1_xz_yy.y);
        pstrain[p1*3+2]=make_float2(Dpp1_yz_zz.x,Dpp1_yz_zz.y);
        }
        tauff[p1*3]=make_float2(taup1_xx_xy.x,taup1_xx_xy.y);
        tauff[p1*3+1]=make_float2(taup1_xz_yy.x,taup1_xz_yy.y);
        tauff[p1*3+2]=make_float2(taup1_yz_zz.x,taup1_yz_zz.y);
    }
    //-Stores results.
    if(tvisco!=VISCO_Artificial && tvisco != VISCO_SoilWater) {
      //save deformation tensor
      float2 rg;
      rg=tauff[p1*3];  rg=make_float2(rg.x+taup1_xx_xy.x,rg.y+taup1_xx_xy.y);  tauff[p1*3]=rg;
      rg=tauff[p1*3+1];  rg=make_float2(rg.x+taup1_xz_yy.x,rg.y+taup1_xz_yy.y);  tauff[p1*3+1]=rg;
      rg=tauff[p1*3+2];  rg=make_float2(rg.x+taup1_yz_zz.x,rg.y+taup1_yz_zz.y);  tauff[p1*3+2]=rg;
    }
  }
}

//------------------------------------------------------------------------------
/// Interaction of a particle with a set of particles for non-Newtonian models using the SPH approach. (Fluid/Float-Fluid/Float/Bound)
/// Realiza la interaccion de una particula con un conjunto de ellas para modelos no-Newtonianos que utilizan el enfoque de la SPH. (Fluid/Float-Fluid/Float/Bound)
//------------------------------------------------------------------------------
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,bool symm>
__device__ void KerInteractionForcesFluidBox_SPH_Morris(bool boundp2,unsigned p1
  ,const unsigned &pini,const unsigned &pfin,float visco,float *visco_eta
  ,const float *ftomassp
  ,const float4 *poscell,const float4 *velrhop
  ,const typecode *code,const unsigned *idp
  ,const typecode pp1,bool ftp1
  ,const float4 &pscellp1,const float4 &velrhop1
  ,float3 &acep1,float &visc,float &visco_etap1)
{
  for(int p2=pini; p2<pfin; p2++) {
    const float4 pscellp2=poscell[p2];
    float drx=pscellp1.x-pscellp2.x+CTE.poscellsize*(CEL_GetX(__float_as_int(pscellp1.w))-CEL_GetX(__float_as_int(pscellp2.w)));
    float dry=pscellp1.y-pscellp2.y+CTE.poscellsize*(CEL_GetY(__float_as_int(pscellp1.w))-CEL_GetY(__float_as_int(pscellp2.w)));
    float drz=pscellp1.z-pscellp2.z+CTE.poscellsize*(CEL_GetZ(__float_as_int(pscellp1.w))-CEL_GetZ(__float_as_int(pscellp2.w)));
    if(symm)dry=pscellp1.y+pscellp2.y+CTE.poscellsize*CEL_GetY(__float_as_int(pscellp2.w)); //<vs_syymmetry>
    const float rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO) {
      //-Computes kernel.
      const float fac=cufsph::GetKernel_Fac<tker>(rr2);
      const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.

      //-Obtains mass of particle p2 for NN and if any floating bodies exist.
      const typecode cod=code[p2];
      const typecode pp2=(boundp2 ? pp1 : CODE_GetTypeValue(cod)); //<vs_non-Newtonian>
      float massp2=(boundp2 ? CTE.massb : PHASEARRAY[pp2].mass); //massp2 not neccesary to go in _Box function
      //Note if you masses are very different more than a ratio of 1.3 then: massp2 = (boundp2 ? PHASEARRAY[pp1].mass : PHASEARRAY[pp2].mass);

      bool ftp2=false;        //-Indicates if it is floating. | Indica si es floating.
      float ftmassp2;						//-Contains mass of floating body or massf if fluid. | Contiene masa de particula floating o massp2 si es bound o fluid.
      bool compute=true;			//-Deactivated when DEM is used and is float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
      if(USE_FLOATING) {
        const typecode cod=code[p2];
        ftp2=CODE_IsFloating(cod);
        ftmassp2=(ftp2 ? ftomassp[CODE_GetTypeValue(cod)] : massp2);
        compute=!(USE_FTEXTERNAL && ftp1&&(boundp2||ftp2)); //-Deactivated when DEM or Chrono is used and is float-float or float-bound. | Se desactiva cuando se usa DEM o Chrono y es float-float o float-bound.
      }

      float4 velrhop2=velrhop[p2];
      if(symm)velrhop2.y=-velrhop2.y; //<vs_syymmetry>

      //-velocity dvx.
      float dvx=velrhop1.x-velrhop2.x,dvy=velrhop1.y-velrhop2.y,dvz=velrhop1.z-velrhop2.z;
      if(boundp2) { //this applies no slip on stress tensor
        dvx=2.f*velrhop1.x; dvy=2.f*velrhop1.y; dvz=2.f*velrhop1.z;  //fomraly I should use the moving BC vel as ug=2ub-uf
      }
      const float cbar=max(PHASEARRAY[pp2].Cs0,PHASEARRAY[pp2].Cs0); //get max Cs0 of phases

      //===== Viscosity ===== 
      if(compute) {
        const float dot=drx*dvx+dry*dvy+drz*dvz;
        const float dot_rr2=dot/(rr2+CTE.eta2);
        visc=max(dot_rr2,visc);  //ViscDt=max(dot/(rr2+Eta2),ViscDt);
        //<vs_non-Newtonian>
        const float visco_NN=PHASECTE[pp2].visco;
        if(tvisco==VISCO_Artificial) {//-Artificial viscosity.
          if(dot<0) {
            const float amubar=CTE.kernelh*dot_rr2;  //amubar=CTE.kernelh*dot/(rr2+CTE.eta2);
            const float robar=(velrhop1.w+velrhop2.w)*0.5f;
            const float pi_visc=(-visco_NN*cbar*amubar/robar)*(USE_FLOATING ? ftmassp2 : massp2);
            acep1.x-=pi_visc*frx; acep1.y-=pi_visc*fry; acep1.z-=pi_visc*frz;
          }
        }
        else if(tvisco!=VISCO_Artificial) {//-Laminar viscosity.
          {//-Laminar contribution.
            //vel gradients
            float visco_etap2=visco_eta[p2];
            //Morris Operator
            if(boundp2)visco_etap2=visco_etap1;
            //Morris Operator
            const float temp=(visco_etap1+visco_etap2)/((rr2+CTE.eta2)*velrhop2.w);
            const float vtemp=(USE_FLOATING ? ftmassp2 : massp2)*temp*(drx*frx+dry*fry+drz*frz);
            acep1.x+=vtemp*dvx; acep1.y+=vtemp*dvy; acep1.z+=vtemp*dvz;
          }
          //-SPS turbulence model.
          //-SPS turbulence model is disabled in v5.0 NN version
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
/// Interaction between particles for non-Newtonian models using the SPH approach. Fluid/Float-Fluid/Float or Fluid/Float-Bound.
/// Includes artificial/laminar viscosity and normal/DEM floating bodies.
///
/// Realiza interaccion entre particulas para modelos no-Newtonianos que utilizan el enfoque de la SPH. Fluid/Float-Fluid/Float or Fluid/Float-Bound
/// Incluye visco artificial/laminar y floatings normales/dem.
//------------------------------------------------------------------------------
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,bool symm>
__global__ void KerInteractionForcesFluid_NN_SPH_Morris(unsigned n,unsigned pinit,float viscob,float viscof,float *visco_eta
  ,int scelldiv,int4 nc,int3 cellzero,const int2 *begincell,unsigned cellfluid,const unsigned *dcell
  ,const float *ftomassp,float *auxnn,const float4 *poscell,const float4 *velrhop
  ,const typecode *code,const unsigned *idp
  ,float3 *ace)
{
  const unsigned p=blockIdx.x*blockDim.x+threadIdx.x; //-Number of particle.
  if(p<n) {
    unsigned p1=p+pinit;      //-Number of particle.			
    float3 acep1=make_float3(0,0,0);
    float visc=0;

    //-Obtains data of particle p1 in case there are floating bodies.
    //-Obtiene datos de particula p1 en caso de existir floatings.
    bool ftp1;       //-Indicates if it is floating. | Indica si es floating.		
    const typecode cod=code[p1];
    if(USE_FLOATING) {
      const typecode cod=code[p1];
      ftp1=CODE_IsFloating(cod);
    }

    //-Obtains basic data of particle p1.
    const float4 pscellp1=poscell[p1];
    const float4 velrhop1=velrhop[p1];
    const bool rsymp1=(symm && CEL_GetPartY(__float_as_uint(pscellp1.w))==0); //<vs_syymmetry>

    //<vs_non-Newtonian>
    const typecode pp1=CODE_GetTypeValue(cod);
    float visco_etap1=visco_eta[p1];

    //-Obtains neighborhood search limits.
    int ini1,fin1,ini2,fin2,ini3,fin3;
    cunsearch::InitCte(dcell[p1],scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

    //-Interaction with fluids.
    ini3+=cellfluid; fin3+=cellfluid;
    for(int c3=ini3; c3<fin3; c3+=nc.w)for(int c2=ini2; c2<fin2; c2+=nc.x) {
      unsigned pini,pfin=0; cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
      if(pfin) {
        KerInteractionForcesFluidBox_SPH_Morris<tker,ftmode,tvisco,false>(false,p1,pini,pfin,viscof,visco_eta,ftomassp,poscell,velrhop,code,idp,pp1,ftp1,pscellp1,velrhop1,acep1,visc,visco_etap1);
        if(symm && rsymp1)	KerInteractionForcesFluidBox_SPH_Morris<tker,ftmode,tvisco,true>(false,p1,pini,pfin,viscof,visco_eta,ftomassp,poscell,velrhop,code,idp,pp1,ftp1,pscellp1,velrhop1,acep1,visc,visco_etap1);
      }
    }
    //-Interaction with boundaries.
    ini3-=cellfluid; fin3-=cellfluid;
    for(int c3=ini3; c3<fin3; c3+=nc.w)for(int c2=ini2; c2<fin2; c2+=nc.x) {
      unsigned pini,pfin=0; cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
      if(pfin) {
        KerInteractionForcesFluidBox_SPH_Morris<tker,ftmode,tvisco,false>(true,p1,pini,pfin,viscob,visco_eta,ftomassp,poscell,velrhop,code,idp,pp1,ftp1,pscellp1,velrhop1,acep1,visc,visco_etap1);
        if(symm && rsymp1)	KerInteractionForcesFluidBox_SPH_Morris<tker,ftmode,tvisco,true>(true,p1,pini,pfin,viscob,visco_eta,ftomassp,poscell,velrhop,code,idp,pp1,ftp1,pscellp1,velrhop1,acep1,visc,visco_etap1);
      }
    }
    //-Stores results.
    if(acep1.x||acep1.y||acep1.z) {
      float3 r=ace[p1]; r.x+=acep1.x; r.y+=acep1.y; r.z+=acep1.z; ace[p1]=r;
      //auxnn[p1] = visco_etap1; // to be used if an auxilary is needed.
    }
  }
}

//==============================================================================
/// Calculates the strain rate tensor and effective viscocity for each particle
/// Calcula el tensor de la velocidad de deformacion y la viscosidad efectiva para cada particula.
//==============================================================================
template<TpFtMode ftmode,TpVisco tvisco,bool symm>
__global__ void KerInteractionForcesFluid_NN_SPH_Visco_eta(unsigned n,unsigned pinit,float viscob,float *visco_eta,const float4 *velrhop
  ,int scelldiv,int4 nc,int3 cellzero,const int2 *begincell,unsigned cellfluid,const unsigned *dcell
  ,float2 *d_tensorff,float2 *gradvelff
  ,const typecode *code,const unsigned *idp
  ,float *viscetadt)
{
  const unsigned p=blockIdx.x*blockDim.x+threadIdx.x; //-Number of particle.
  if(p<n) {
    unsigned p1=p+pinit;      //-Number of particle.
    //-Obtains basic data of particle p1.
    //const float4 pscellp1 = poscell[p1];
    //const float4 velrhop1 = velrhop[p1];

    //<vs_non-Newtonian>
    const typecode cod=code[p1];
    const typecode pp1=CODE_GetTypeValue(cod);
    float visco_etap1=0;

    //-Variables for gradients.
    float2 grap1_xx_xy,grap1_xz_yy,grap1_yz_zz;
    grap1_xx_xy=gradvelff[p1*3];
    grap1_xz_yy=gradvelff[p1*3+1];
    grap1_yz_zz=gradvelff[p1*3+2];

    //Strain rate tensor 
    float2 dtsrp1_xx_xy=make_float2(0,0);
    float2 dtsrp1_xz_yy=make_float2(0,0);
    float2 dtsrp1_yz_zz=make_float2(0,0);
    float div_D_tensor=0; float D_tensor_magn=0;
    float I_D,II_D; float J1_D,J2_D;
    GetStrainRateTensor_tsym(grap1_xx_xy,grap1_xz_yy,grap1_yz_zz,I_D,II_D,J1_D,J2_D,div_D_tensor,D_tensor_magn,dtsrp1_xx_xy,dtsrp1_xz_yy,dtsrp1_yz_zz);

    //Effective viscosity
    float m_NN=PHASECTE[pp1].m_NN; float n_NN=PHASECTE[pp1].n_NN; float tau_yield=PHASECTE[pp1].tau_yield; float visco_NN=PHASECTE[pp1].visco;
    KerGetEta_Effective(pp1,tau_yield,D_tensor_magn,visco_NN,m_NN,n_NN,visco_etap1);

    //-Stores results.
    if(tvisco!=VISCO_Artificial) {
      //time step restriction
      if(visco_etap1>viscetadt[p1])viscetadt[p1]=visco_etap1; //no visceta necessary here
      //save deformation tensor
      float2 rg;
      rg=d_tensorff[p1*3];  rg=make_float2(rg.x+dtsrp1_xx_xy.x,rg.y+dtsrp1_xx_xy.y);  d_tensorff[p1*3]=rg;
      rg=d_tensorff[p1*3+1];  rg=make_float2(rg.x+dtsrp1_xz_yy.x,rg.y+dtsrp1_xz_yy.y);  d_tensorff[p1*3+1]=rg;
      rg=d_tensorff[p1*3+2];  rg=make_float2(rg.x+dtsrp1_yz_zz.x,rg.y+dtsrp1_yz_zz.y);  d_tensorff[p1*3+2]=rg;
      visco_eta[p1]=visco_etap1;
    }
    //auxnn[p1] = visco_etap1; // to be used if an auxilary is needed.
  }
}

//------------------------------------------------------------------------------
/// Interaction of a particle with a set of particles. (Fluid/Float-Fluid/Float/Bound) Not for multilayer
/// Realiza la interaccion de una particula con un conjunto de ellas. (Fluid/Float-Fluid/Float/Bound)
//------------------------------------------------------------------------------
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,TpDensity tdensity,bool shift,bool symm>
__device__ void KerInteractionForcesFluidBox_NN_SPH_PressGrad(bool boundp2,unsigned p1
  ,const unsigned &pini,const unsigned &pfin
  ,const float *ftomassp
  ,const float4 *poscell
  ,const float4 *velrhop,const typecode *code,const unsigned *idp
  ,float massp2,const typecode pp1,bool ftp1
  ,const float4 &pscellp1,const float4 &velrhop1,float pressp1
  ,float2 &grap1_xx_xy,float2 &grap1_xz_yy,float2 &grap1_yz_zz
  ,float3 &acep1,float &arp1,float &visc,float &deltap1
  ,TpShifting shiftmode,float4 &shiftposfsp1)
{
  for(int p2=pini; p2<pfin; p2++) {
    const float4 pscellp2=poscell[p2];
    float drx=pscellp1.x-pscellp2.x+CTE.poscellsize*(CEL_GetX(__float_as_int(pscellp1.w))-CEL_GetX(__float_as_int(pscellp2.w)));
    float dry=pscellp1.y-pscellp2.y+CTE.poscellsize*(CEL_GetY(__float_as_int(pscellp1.w))-CEL_GetY(__float_as_int(pscellp2.w)));
    float drz=pscellp1.z-pscellp2.z+CTE.poscellsize*(CEL_GetZ(__float_as_int(pscellp1.w))-CEL_GetZ(__float_as_int(pscellp2.w)));
    if(symm)dry=pscellp1.y+pscellp2.y+CTE.poscellsize*CEL_GetY(__float_as_int(pscellp2.w)); //<vs_syymmetry>
    const float rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO) {
      //-Computes kernel.
      const float fac=cufsph::GetKernel_Fac<tker>(rr2);
      const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.

      //-Obtains mass of particle p2 for NN and if any floating bodies exist.
      const typecode cod=code[p2];
      const typecode pp2=(boundp2 ? pp1 : CODE_GetTypeValue(cod)); //<vs_non-Newtonian>
      float massp2=(boundp2 ? CTE.massb : PHASEARRAY[pp2].mass); //massp2 not neccesary to go in _Box function
      //Note if you masses are very different more than a ratio of 1.3 then: massp2 = (boundp2 ? PHASEARRAY[pp1].mass : PHASEARRAY[pp2].mass);

      //-Obtiene masa de particula p2 en caso de existir floatings.
      bool ftp2=false;        //-Indicates if it is floating. | Indica si es floating.
      float ftmassp2;						//-Contains mass of floating body or massf if fluid. | Contiene masa de particula floating o massp2 si es bound o fluid.
      bool compute=true;			//-Deactivated when DEM is used and is float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
      if(USE_FLOATING) {
        const typecode cod=code[p2];
        ftp2=CODE_IsFloating(cod);
        ftmassp2=(ftp2 ? ftomassp[CODE_GetTypeValue(cod)] : massp2);
#ifdef DELTA_HEAVYFLOATING
        if(ftp2 && tdensity==DDT_DDT && ftmassp2<=(massp2*1.2f))deltap1=FLT_MAX;
#else
        if(ftp2 && tdensity==DDT_DDT)deltap1=FLT_MAX;
#endif
        if(ftp2 && shift && shiftmode==SHIFT_NoBound)shiftposfsp1.x=FLT_MAX; //-Cancels shifting with floating bodies. | Con floatings anula shifting.
        compute=!(USE_FTEXTERNAL && ftp1&&(boundp2||ftp2)); //-Deactivated when DEM or Chrono is used and is float-float or float-bound. | Se desactiva cuando se usa DEM o Chrono y es float-float o float-bound.
      }
      float4 velrhop2=velrhop[p2];
      if(symm)velrhop2.y=-velrhop2.y; //<vs_syymmetry>

      //===== Aceleration ===== 
      if(compute) {
        const float pressp2=cufsph::ComputePressCte_NN(velrhop2.w,PHASEARRAY[pp2].rho,PHASEARRAY[pp2].CteB,PHASEARRAY[pp2].Gamma,PHASEARRAY[pp2].Cs0,cod);
        const float prs=(pressp1+pressp2)/(velrhop1.w*velrhop2.w)+(tker==KERNEL_Cubic ? cufsph::GetKernelCubic_Tensil(rr2,velrhop1.w,pressp1,velrhop2.w,pressp2) : 0);
        const float p_vpm=-prs*(USE_FLOATING ? ftmassp2 : massp2);
        acep1.x+=p_vpm*frx; acep1.y+=p_vpm*fry; acep1.z+=p_vpm*frz;
      }

      //-Density derivative.
      float dvx=velrhop1.x-velrhop2.x,dvy=velrhop1.y-velrhop2.y,dvz=velrhop1.z-velrhop2.z;
      if(compute)arp1+=(USE_FLOATING ? ftmassp2 : massp2)*(dvx*frx+dvy*fry+dvz*frz)*(velrhop1.w/velrhop2.w);

      const float cbar=max(PHASEARRAY[pp1].Cs0,PHASEARRAY[pp2].Cs0);
      const float dot3=(tdensity!=DDT_None||shift ? drx*frx+dry*fry+drz*frz : 0);
      //-Density derivative (DeltaSPH Molteni).
      if(tdensity==DDT_DDT && deltap1!=FLT_MAX) {
        const float rhop1over2=velrhop1.w/velrhop2.w;
        const float visc_densi=CTE.ddtkh*cbar*(rhop1over2-1.f)/(rr2+CTE.eta2);
        const float delta=(pp1==pp2 ? visc_densi*dot3*(USE_FLOATING ? ftmassp2 : massp2) : 0); //<vs_non-Newtonian>
        //deltap1=(boundp2? FLT_MAX: deltap1+delta);
        deltap1=(boundp2 && CTE.tboundary==BC_DBC ? FLT_MAX : deltap1+delta);
      }
      //-Density Diffusion Term (Fourtakas et al 2019). //<vs_dtt2_ini>
      if((tdensity==DDT_DDT2||(tdensity==DDT_DDT2Full&&!boundp2))&&deltap1!=FLT_MAX&&!ftp2) {
        const float rh=1.f+CTE.ddtgz*drz;
        const float drhop=CTE.rhopzero*pow(rh,1.f/CTE.gamma)-CTE.rhopzero;
        const float visc_densi=CTE.ddtkh*cbar*((velrhop2.w-velrhop1.w)-drhop)/(rr2+CTE.eta2);
        const float delta=(pp1==pp2 ? visc_densi*dot3*massp2/velrhop2.w : 0); //<vs_non-Newtonian>
        deltap1=(boundp2 ? FLT_MAX : deltap1-delta);
      } //<vs_dtt2_end>		

      //-Shifting correction.
      if(shift && shiftposfsp1.x!=FLT_MAX) {
        bool heavyphase=(PHASEARRAY[pp1].mass>PHASEARRAY[pp2].mass && pp1!=pp2 ? true : false); //<vs_non-Newtonian>
        const float massrhop=(USE_FLOATING ? ftmassp2 : massp2)/velrhop2.w;
        const bool noshift=(boundp2&&(shiftmode==SHIFT_NoBound||(shiftmode==SHIFT_NoFixed && CODE_IsFixed(code[p2]))));
        shiftposfsp1.x=(noshift ? FLT_MAX : (heavyphase ? 0 : shiftposfsp1.x+massrhop*frx)); //-Removes shifting for the boundaries. | Con boundary anula shifting.
        shiftposfsp1.y+=(heavyphase ? 0 : massrhop*fry);
        shiftposfsp1.z+=(heavyphase ? 0 : massrhop*frz);
        shiftposfsp1.w-=(heavyphase ? 0 : massrhop*dot3);
      }

      //===== Viscosity ===== 
      if(compute) {
        const float dot=drx*dvx+dry*dvy+drz*dvz;
        const float dot_rr2=dot/(rr2+CTE.eta2);
        visc=max(dot_rr2,visc);  //ViscDt=max(dot/(rr2+Eta2),ViscDt);

        if(tvisco!=VISCO_Artificial) { //&& !boundp2
          //vel gradients
          if(boundp2) {
            dvx=2.f*velrhop1.x; dvy=2.f*velrhop1.y; dvz=2.f*velrhop1.z;  //fomraly I should use the moving BC vel as ug=2ub-uf
          }
          GetVelocityGradients_SPH_tsym(massp2,velrhop2,dvx,dvy,dvz,frx,fry,frz,grap1_xx_xy,grap1_xz_yy,grap1_yz_zz);
        }
      }
    }
  }
}

///  Interploation of soil particle velocity at fluid particle position (Multilayer)
template<TpKernel tker,  bool symm>
__device__ void KerInterpolationVelGranularToFluid(const unsigned& pini, const unsigned& pfin, const float4* poscell, const float4* velrhop
      , const typecode* code, const float4& pscellp1, float& usp1)
  {
      float usp1_denominator;   float3 usp1_numerator;   float wac;   float4 velrhop2;
      usp1_denominator = 0; usp1_numerator.x = 0; usp1_numerator.y = 0; usp1_numerator.z = 0; 
      for (int p2 = pini; p2 < pfin; p2++) {
          const float4 pscellp2 = poscell[p2];
          float drx = pscellp1.x - pscellp2.x + CTE.poscellsize * (CEL_GetX(__float_as_int(pscellp1.w)) - CEL_GetX(__float_as_int(pscellp2.w)));
          float dry = pscellp1.y - pscellp2.y + CTE.poscellsize * (CEL_GetY(__float_as_int(pscellp1.w)) - CEL_GetY(__float_as_int(pscellp2.w)));
          float drz = pscellp1.z - pscellp2.z + CTE.poscellsize * (CEL_GetZ(__float_as_int(pscellp1.w)) - CEL_GetZ(__float_as_int(pscellp2.w)));
          if (symm)dry = pscellp1.y + pscellp2.y + CTE.poscellsize * CEL_GetY(__float_as_int(pscellp2.w)); //<vs_syymmetry>
          const float rr2 = drx * drx + dry * dry + drz * drz;
          if (rr2 <= CTE.kernelsize2 && rr2 >= ALMOSTZERO) {
              const typecode cod = code[p2];
              const typecode pp2 = CODE_GetTypeValue(cod); //<vs_non-Newtonian>
              if (pp2 == 1) {// p1 is fluid, p2 is soil
                  //-Computes kernel.
                  wac = cufsph::GetKernel_Wab<tker>(rr2);
                  velrhop2 = velrhop[p2];
                  usp1_denominator += wac / velrhop2.w;   
                  usp1_numerator.x += velrhop2.x * usp1_denominator; 
                  usp1_numerator.y += velrhop2.y * usp1_denominator; 
                  usp1_numerator.z += velrhop2.z * usp1_denominator;
              }
          }
      }
      usp1 = usp1_numerator / usp1_denominator;
  }

///  Interploation of fluid particle pressure at soil particle position (Multilayer)
template<TpKernel tker, TpVisco tvisco, bool symm>
__device__ void KerInterpolationPFluidToGranular(const unsigned& pini, const unsigned& pfin, const float4* poscell, const float4* velrhop
      , const typecode* code, const float4& pscellp1, float& prep1, float* volfrac)
  {
      float prep1_denominator;    float prep1_numerator;    float4 velrhop2;    float volfracp2;    float wac;   float pressp2;
      prep1_denominator = 0; prep1_numerator = 0; 
      for (int p2 = pini; p2 < pfin; p2++) {
          const float4 pscellp2 = poscell[p2];
          float drx = pscellp1.x - pscellp2.x + CTE.poscellsize * (CEL_GetX(__float_as_int(pscellp1.w)) - CEL_GetX(__float_as_int(pscellp2.w)));
          float dry = pscellp1.y - pscellp2.y + CTE.poscellsize * (CEL_GetY(__float_as_int(pscellp1.w)) - CEL_GetY(__float_as_int(pscellp2.w)));
          float drz = pscellp1.z - pscellp2.z + CTE.poscellsize * (CEL_GetZ(__float_as_int(pscellp1.w)) - CEL_GetZ(__float_as_int(pscellp2.w)));
          if (symm)dry = pscellp1.y + pscellp2.y + CTE.poscellsize * CEL_GetY(__float_as_int(pscellp2.w)); //<vs_syymmetry>
          const float rr2 = drx * drx + dry * dry + drz * drz;
          if (rr2 <= CTE.kernelsize2 && rr2 >= ALMOSTZERO) {
              const typecode cod = code[p2];
              const typecode pp2 = CODE_GetTypeValue(cod); //<vs_non-Newtonian>
              if (pp2 == 0) {// p1 is soil, p2 is fluid
                  //-Computes kernel.
                  wac = cufsph::GetKernel_Wab<tker>(rr2);
                  velrhop2 = velrhop[p2];
                  volfracp2 = volfrac[p2];
                  pressp2 = cufsph::ComputePressCte_NN(velrhop2.w, PHASEARRAY[pp2].rho, PHASEARRAY[pp2].CteB, PHASEARRAY[pp2].Gamma, PHASEARRAY[pp2].Cs0, cod);
                  prep1_denominator += wac / (volfracp2 * velrhop2.w);   
                  prep1_numerator.x += pressp2 * prep1_denominator; 
              }
          }
      }
      prep1 = prep1_numerator / prep1_denominator;
  }

///  Interaction of a soil particle (p1) with a set of particles (p2) either of fluid or granular. (Multilayer)
template<TpKernel tker, TpFtMode ftmode, TpVisco tvisco, TpDensity tdensity, bool shift, bool symm>
__device__ void KerInteractionForcesMultilayerGranularBox_NN_SPH_PressGrad(bool boundp2, unsigned p1
      , const unsigned& pini, const unsigned& pfin
      , const float* ftomassp
      , const float4* poscell
      , const float4* velrhop, const typecode* code, const unsigned* idp
      , float massp2, const typecode pp1, bool ftp1
      , const float4& pscellp1, const float4& velrhop1, float pressp1
      , float2& grap1_xx_xy, float2& grap1_xz_yy, float2& grap1_yz_zz
      , float3& acep1, float& arp1, float& visc, float& deltap1
      , TpShifting shiftmode, float4& shiftposfsp1, float prep1, float& volfracp1, float* volfrac)
  {
      for (int p2 = pini; p2 < pfin; p2++) {
          const float4 pscellp2 = poscell[p2];
          float drx = pscellp1.x - pscellp2.x + CTE.poscellsize * (CEL_GetX(__float_as_int(pscellp1.w)) - CEL_GetX(__float_as_int(pscellp2.w)));
          float dry = pscellp1.y - pscellp2.y + CTE.poscellsize * (CEL_GetY(__float_as_int(pscellp1.w)) - CEL_GetY(__float_as_int(pscellp2.w)));
          float drz = pscellp1.z - pscellp2.z + CTE.poscellsize * (CEL_GetZ(__float_as_int(pscellp1.w)) - CEL_GetZ(__float_as_int(pscellp2.w)));
          if (symm)dry = pscellp1.y + pscellp2.y + CTE.poscellsize * CEL_GetY(__float_as_int(pscellp2.w)); //<vs_syymmetry>
          const float rr2 = drx * drx + dry * dry + drz * drz;
          if (rr2 <= CTE.kernelsize2 && rr2 >= ALMOSTZERO) {
              //-Computes kernel.
              const float fac = cufsph::GetKernel_Fac<tker>(rr2);
              const float frx = fac * drx, fry = fac * dry, frz = fac * drz; //-Gradients.
              const float wac = cufsph::GetKernel_Wab<tker>(rr2);

              //-Obtains mass of particle p2 for NN and if any floating bodies exist.
              const typecode cod = code[p2];
              const typecode pp2 = (boundp2 ? pp1 : CODE_GetTypeValue(cod)); //<vs_non-Newtonian>
              float massp2; //massp2 not neccesary to go in _Box function
              if(pp2 == 0) massp2 = (boundp2 ? CTE.massb : PHASEARRAY[pp2].mass); // p2 is fluid
              if(pp2 == 1) massp2 = (boundp2 ? CTE.massb : PHASEDRUCKERPRAGER[pp2].mass); // p2 is granular
              //Note if you masses are very different more than a ratio of 1.3 then: massp2 = (boundp2 ? PHASEARRAY[pp1].mass : PHASEARRAY[pp2].mass);

              //-Obtiene masa de particula p2 en caso de existir floatings.
              bool ftp2 = false;        //-Indicates if it is floating. | Indica si es floating.
              float ftmassp2;						//-Contains mass of floating body or massf if fluid. | Contiene masa de particula floating o massp2 si es bound o fluid.
              bool compute = true;			//-Deactivated when DEM is used and is float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
              if (USE_FLOATING) {
                  const typecode cod = code[p2];
                  ftp2 = CODE_IsFloating(cod);
                  ftmassp2 = (ftp2 ? ftomassp[CODE_GetTypeValue(cod)] : massp2);
#ifdef DELTA_HEAVYFLOATING
                  if (ftp2 && tdensity == DDT_DDT && ftmassp2 <= (massp2 * 1.2f))deltap1 = FLT_MAX;
#else
                  if (ftp2 && tdensity == DDT_DDT)deltap1 = FLT_MAX;
#endif
                  if (ftp2 && shift && shiftmode == SHIFT_NoBound)shiftposfsp1.x = FLT_MAX; //-Cancels shifting with floating bodies. | Con floatings anula shifting.
                  compute = !(USE_FTEXTERNAL && ftp1 && (boundp2 || ftp2)); //-Deactivated when DEM or Chrono is used and is float-float or float-bound. | Se desactiva cuando se usa DEM o Chrono y es float-float o float-bound.
              }
              float4 velrhop2 = velrhop[p2];
              if (symm)velrhop2.y = -velrhop2.y; //<vs_syymmetry>
              float dvx=velrhop1.x-velrhop2.x,dvy=velrhop1.y-velrhop2.y,dvz=velrhop1.z-velrhop2.z;
              float dv = sqrt(dvx*dvx + dvy*dvy + dvz*dvz);

              //===== Aceleration ===== 
              if (compute) {
                 if (pp2 == 0) { // fluid
                   const float pressp2 = cufsph::ComputePressCte_NN(velrhop2.w, PHASEARRAY[pp2].rho, PHASEARRAY[pp2].CteB, PHASEARRAY[pp2].Gamma, PHASEARRAY[pp2].Cs0, cod);
                   // prep1 is the interpolated water pressure at soil particle p1
                   const float volfracp2 = volfrac[p2];
                   const float prs = volfracp1 * (pressp2 - prep1) / (volfracp2 * velrhop2.w * velrhop1.w) + (tker == KERNEL_Cubic ? cufsph::GetKernelCubic_Tensil(rr2, velrhop1.w, pressp1, velrhop2.w, pressp2) : 0);
                   const float p_vpm = -prs * (USE_FLOATING ? ftmassp2 : massp2);

                   const float fd_x, fd_y, fd_z;
                   const float Drag_alphad = PHASEDRUCKERPRAGER[pp1].Drag_alphad;
                   const float visco = PhaseCte[pp2].visco;
                   const float DP_Dc = PHASEDRUCKERPRAGER[pp1].DP_Dc;
                   const float Drag_betad = PHASEDRUCKERPRAGER[pp1].Drag_betad;
                   fd_x = Drag_alpha * visco * (1 - volfracp2)*(1 - volfracp2) / volfracp2 / DP_Dc / DP_Dc * (-dvx) + Drag_betad * velrhop2.w * (1 - volfracp2) / DP_Dc * dv * (-dvx); 
                   fd_y = Drag_alpha * visco * (1 - volfracp2)*(1 - volfracp2) / volfracp2 / DP_Dc / DP_Dc * (-dvy) + Drag_betad * velrhop2.w * (1 - volfracp2) / DP_Dc * dv * (-dvy);
                   fd_z = Drag_alpha * visco * (1 - volfracp2)*(1 - volfracp2) / volfracp2 / DP_Dc / DP_Dc * (-dvz) + Drag_betad * velrhop2.w * (1 - volfracp2) / DP_Dc * dv * (-dvz);
                   acep1.x += p_vpm * frx + fd_x*massp2/(velrhop1.w * velrhop2.w * volfracp2)*wac; acep1.y += p_vpm * fry + fd_y*massp2/(velrhop1.w * velrhop2.w * volfracp2)*wac; acep1.z += p_vpm * frz + fd_z*massp2/(velrhop1.w * velrhop2.w * volfracp2)*wac;
                 }
                 // Skip if pp2 == 1 is soil, will be calcuated in conseq
              }

              //-Density derivative.
              if (compute){
                if (pp2 == 1) {
                  float dvx = velrhop1.x - velrhop2.x, dvy = velrhop1.y - velrhop2.y, dvz = velrhop1.z - velrhop2.z;
                  arp1 += velrhop1.w * (USE_FLOATING ? ftmassp2 : massp2) / velrhop2.w * (dvx * frx + dvy * fry + dvz * frz);
                }
              }

              if (TVISCO != VISCO_SoilWater || pp2 == 0) const float cbar = max(PHASEARRAY[pp1].Cs0, PHASEARRAY[pp2].Cs0);
              else if (pp2 == 1) const float cbar = PHASEDRUCKERPRAGER[pp1].Cs0;
              const float dot3 = (tdensity != DDT_None || shift ? drx * frx + dry * fry + drz * frz : 0);
              //-Density derivative (DeltaSPH Molteni).
              if (tdensity == DDT_DDT && deltap1 != FLT_MAX) {
                  const float rhop1over2 = velrhop1.w / velrhop2.w;
                  const float visc_densi = CTE.ddtkh * cbar * (rhop1over2 - 1.f) / (rr2 + CTE.eta2);
                  const float delta = (pp1 == pp2 ? visc_densi * dot3 * (USE_FLOATING ? ftmassp2 : massp2) : 0); //<vs_non-Newtonian>
                  //deltap1=(boundp2? FLT_MAX: deltap1+delta);
                  deltap1 = (boundp2 && CTE.tboundary == BC_DBC ? FLT_MAX : deltap1 + delta);
              }
              //-Density Diffusion Term (Fourtakas et al 2019). //<vs_dtt2_ini>
              if ((tdensity == DDT_DDT2 || (tdensity == DDT_DDT2Full && !boundp2)) && deltap1 != FLT_MAX && !ftp2) {
                  const float rh = 1.f + CTE.ddtgz * drz;
                  const float drhop = CTE.rhopzero * pow(rh, 1.f / CTE.gamma) - CTE.rhopzero;
                  const float visc_densi = CTE.ddtkh * cbar * ((velrhop2.w - velrhop1.w) - drhop) / (rr2 + CTE.eta2);
                  const float delta = (pp1 == pp2 ? visc_densi * dot3 * massp2 / velrhop2.w : 0); //<vs_non-Newtonian>
                  deltap1 = (boundp2 ? FLT_MAX : deltap1 - delta);
              } //<vs_dtt2_end>		

              //-Shifting correction.
              if (shift && shiftposfsp1.x != FLT_MAX) {
                  bool heavyphase = (PHASEARRAY[pp1].mass > PHASEARRAY[pp2].mass && pp1 != pp2 ? true : false); //<vs_non-Newtonian>
                  const float massrhop = (USE_FLOATING ? ftmassp2 : massp2) / velrhop2.w;
                  const bool noshift = (boundp2 && (shiftmode == SHIFT_NoBound || (shiftmode == SHIFT_NoFixed && CODE_IsFixed(code[p2]))));
                  shiftposfsp1.x = (noshift ? FLT_MAX : (heavyphase ? 0 : shiftposfsp1.x + massrhop * frx)); //-Removes shifting for the boundaries. | Con boundary anula shifting.
                  shiftposfsp1.y += (heavyphase ? 0 : massrhop * fry);
                  shiftposfsp1.z += (heavyphase ? 0 : massrhop * frz);
                  shiftposfsp1.w -= (heavyphase ? 0 : massrhop * dot3);
              }

              //===== Viscosity ===== 
              if (compute) {
                  const float dot = drx * dvx + dry * dvy + drz * dvz;
                  const float dot_rr2 = dot / (rr2 + CTE.eta2);
                  visc = max(dot_rr2, visc);  //ViscDt=max(dot/(rr2+Eta2),ViscDt);
                  if((dot<0) && (pp2==1) && tvisco == VISCO_SoilWater) {
                      const float amubar=CTE.kernelh*dot_rr2;  //amubar=CTE.h*dot/(rr2+CTE.eta2);
                      const float robar=(velrhop1.w+velrhop2.w)*0.5f;
                      const float visco_NN;
                      visco_NN = PHASEDRUCKERPRAGER[pp2].Cs0;
                      const float pi_visc=(visco_NN*cbar*amubar/robar)*massp2;
                      acep1.x +=pi_visc*frx; acep1.y +=pi_visc*fry; acep1.z +=pi_visc*frz;
                  }

                  if (tvisco != VISCO_Artificial) { //&& !boundp2
                    //vel gradients
                      if (boundp2) {
                          dvx = 2.f * velrhop1.x; dvy = 2.f * velrhop1.y; dvz = 2.f * velrhop1.z;  //fomraly I should use the moving BC vel as ug=2ub-uf
                      }
                      GetVelocityGradients_SPH_tsym(massp2, velrhop2, dvx, dvy, dvz, frx, fry, frz, grap1_xx_xy, grap1_xz_yy, grap1_yz_zz);
                  }
              }
          }
      }
  }

///  Interaction of a fluid particle (p1) with a set of particles (p2) either of fluid or granular. (Multilayer)
template<TpKernel tker, TpFtMode ftmode, TpVisco tvisco, TpDensity tdensity, bool shift, bool symm>
__device__ void KerInteractionForcesMultilayerFluidBox_NN_SPH_PressGrad(bool boundp2, unsigned p1
      , const unsigned& pini, const unsigned& pfin
      , const float* ftomassp
      , const float4* poscell
      , const float4* velrhop, const typecode* code, const unsigned* idp
      , float massp2, const typecode pp1, bool ftp1
      , const float4& pscellp1, const float4& velrhop1, float pressp1
      , float2& grap1_xx_xy, float2& grap1_xz_yy, float2& grap1_yz_zz
      , float3& acep1, float& arp1, float& visc, float& deltap1
      , TpShifting shiftmode, float4& shiftposfsp1, float velsp1, float& volfracp1, float* volfrac)
  {
      for (int p2 = pini; p2 < pfin; p2++) {
          const float4 pscellp2 = poscell[p2];
          float drx = pscellp1.x - pscellp2.x + CTE.poscellsize * (CEL_GetX(__float_as_int(pscellp1.w)) - CEL_GetX(__float_as_int(pscellp2.w)));
          float dry = pscellp1.y - pscellp2.y + CTE.poscellsize * (CEL_GetY(__float_as_int(pscellp1.w)) - CEL_GetY(__float_as_int(pscellp2.w)));
          float drz = pscellp1.z - pscellp2.z + CTE.poscellsize * (CEL_GetZ(__float_as_int(pscellp1.w)) - CEL_GetZ(__float_as_int(pscellp2.w)));
          if (symm)dry = pscellp1.y + pscellp2.y + CTE.poscellsize * CEL_GetY(__float_as_int(pscellp2.w)); //<vs_syymmetry>
          const float rr2 = drx * drx + dry * dry + drz * drz;
          if (rr2 <= CTE.kernelsize2 && rr2 >= ALMOSTZERO) {
              //-Computes kernel.
              const float fac = cufsph::GetKernel_Fac<tker>(rr2);
              const float frx = fac * drx, fry = fac * dry, frz = fac * drz; //-Gradients.
              const float wac = cufsph::GetKernel_Wab<tker>(rr2);

              //-Obtains mass of particle p2 for NN and if any floating bodies exist.
              const typecode cod = code[p2];
              const typecode pp2 = (boundp2 ? pp1 : CODE_GetTypeValue(cod)); //<vs_non-Newtonian>
              float massp2; //massp2 not neccesary to go in _Box function
              if(pp2 == 0) massp2 = (boundp2 ? CTE.massb : PHASEARRAY[pp2].mass); // p2 is fluid
              if(pp2 == 1) massp2 = (boundp2 ? CTE.massb : PHASEDRUCKERPRAGER[pp2].mass); // p2 is granular
              //Note if you masses are very different more than a ratio of 1.3 then: massp2 = (boundp2 ? PHASEARRAY[pp1].mass : PHASEARRAY[pp2].mass);

              //-Obtiene masa de particula p2 en caso de existir floatings.
              bool ftp2 = false;        //-Indicates if it is floating. | Indica si es floating.
              float ftmassp2;						//-Contains mass of floating body or massf if fluid. | Contiene masa de particula floating o massp2 si es bound o fluid.
              bool compute = true;			//-Deactivated when DEM is used and is float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
              if (USE_FLOATING) {
                  const typecode cod = code[p2];
                  ftp2 = CODE_IsFloating(cod);
                  ftmassp2 = (ftp2 ? ftomassp[CODE_GetTypeValue(cod)] : massp2);
#ifdef DELTA_HEAVYFLOATING
                  if (ftp2 && tdensity == DDT_DDT && ftmassp2 <= (massp2 * 1.2f))deltap1 = FLT_MAX;
#else
                  if (ftp2 && tdensity == DDT_DDT)deltap1 = FLT_MAX;
#endif
                  if (ftp2 && shift && shiftmode == SHIFT_NoBound)shiftposfsp1.x = FLT_MAX; //-Cancels shifting with floating bodies. | Con floatings anula shifting.
                  compute = !(USE_FTEXTERNAL && ftp1 && (boundp2 || ftp2)); //-Deactivated when DEM or Chrono is used and is float-float or float-bound. | Se desactiva cuando se usa DEM o Chrono y es float-float o float-bound.
              }
              float4 velrhop2 = velrhop[p2];
              if (symm)velrhop2.y = -velrhop2.y; //<vs_syymmetry>
              float dvx=velrhop1.x-velrhop2.x,dvy=velrhop1.y-velrhop2.y,dvz=velrhop1.z-velrhop2.z;
              float dv = sqrt(dvx*dvx + dvy*dvy + dvz*dvz);

              //===== Aceleration ===== 
              if (compute) {
                 if (pp2 == 0) { // fluid
                   const float pressp2 = cufsph::ComputePressCte_NN(velrhop2.w, PHASEARRAY[pp2].rho, PHASEARRAY[pp2].CteB, PHASEARRAY[pp2].Gamma, PHASEARRAY[pp2].Cs0, cod);
                   const float prs = (pressp1 + pressp2) / (velrhop1.w * velrhop2.w) + (tker == KERNEL_Cubic ? cufsph::GetKernelCubic_Tensil(rr2, velrhop1.w, pressp1, velrhop2.w, pressp2) : 0);
                   const float p_vpm = -prs * (USE_FLOATING ? ftmassp2 : massp2);
                   acep1.x += p_vpm * frx; acep1.y += p_vpm * fry; acep1.z += p_vpm * frz;
                 }
                 if (pp2 == 1) { // soil
                   //const float pressp2 = cufsph::ComputePressCte_NN(velrhop2.w, PHASEARRAY[pp2].rho, PHASEARRAY[pp2].CteB, PHASEARRAY[pp2].Gamma, PHASEARRAY[pp2].Cs0, cod);
                   const float fd_x, fd_y, fd_z;
                   const float Drag_alphad = PHASEDRUCKERPRAGER[pp2].Drag_alphad;
                   const float visco = PhaseCte[pp1].visco;
                   const float DP_Dc = PHASEDRUCKERPRAGER[pp2].DP_Dc;
                   const float Drag_betad = PHASEDRUCKERPRAGER[pp2].Drag_betad;
                   fd_x = Drag_alpha * visco * (1 - volfracp1)*(1 - volfracp1) / volfracp1 / DP_Dc / DP_Dc * dvx + Drag_betad * velrhop1.w * (1 - volfracp1) / DP_Dc * dv * dvx; 
                   fd_y = Drag_alpha * visco * (1 - volfracp1)*(1 - volfracp1) / volfracp1 / DP_Dc / DP_Dc * dvy + Drag_betad * velrhop1.w * (1 - volfracp1) / DP_Dc * dv * dvy;
                   fd_z = Drag_alpha * visco * (1 - Vvolfracp1)*(1 - volfracp1) / volfracp1 / DP_Dc / DP_Dc * dvz + Drag_betad * velrhop1.w * (1 - volfracp1) / DP_Dc * dv * dvz;
                   const float p_vpm = (USE_FLOATING ? ftmassp2 : massp2) / (volfracp1*velrhop1.w*velrhop2.w);
                   acep1.x -= p_vpm * wac * fd_x ; acep1.y -= p_vpm * wac * fd_y ; acep1.z -= p_vpm * wac * fd_z;
                 }
              }

              //-Density derivative.
              if (compute){
                if (pp2 == 0) {
                  float volfracp2 = volfrac[p2]; 
                  float dvx_VolFrac = volfracp2 * velrhop2.x - volfracp1 * velrhop1.x, dvy_VolFrac = volfracp2 * velrhop2.y - volfracp1 * velrhop1.y, dvz_VolFrac = volfracp2 * velrhop2.z - volfracp1 * velrhop1.z;
                  arp1 -= velrhop1.w / volfracp1 * (USE_FLOATING ? ftmassp2 : massp2) / (volfracp2 * velrhop2.w) * (dvx_VolFrac * frx + dvy_VolFrac * fry + dvz_VolFrac * frz);
                }
                if (pp2 == 1) {//velsp1, velocity of solid particle at position p1
                    float dvx_VolFrac = volfracp2 * velrhop2.x - (1 - volfracp1) * velsp1.x, dvy_VolFrac = volfracp2 * velrhop2.y - (1 - volfracp1) * velsp1.y, dvz_VolFrac = volfracp2 * velrhop2.z - (1 - volfracp1) * velsp1.z;
                  arp1 += velrhop1.w / volfracp1 * (USE_FLOATING ? ftmassp2 : massp2) / velrhop2.w * (dvx_VolFrac * frx + dvy_VolFrac * fry + dvz_VolFrac * frz);
                }
              }

              if (TVISCO != VISCO_SoilWater) const float cbar = max(PHASEARRAY[pp1].Cs0, PHASEARRAY[pp2].Cs0);
              else if (pp2 == 0) const float cbar = max(PHASEARRAY[pp1].Cs0, PHASEARRAY[pp2].Cs0);
              else const float cbar = 0;
              const float dot3 = (tdensity != DDT_None || shift ? drx * frx + dry * fry + drz * frz : 0);
              //-Density derivative (DeltaSPH Molteni).
              if (tdensity == DDT_DDT && deltap1 != FLT_MAX) {
                  const float rhop1over2 = velrhop1.w / velrhop2.w;
                  const float visc_densi = CTE.ddtkh * cbar * (rhop1over2 - 1.f) / (rr2 + CTE.eta2);
                  const float delta = (pp1 == pp2 ? visc_densi * dot3 * (USE_FLOATING ? ftmassp2 : massp2) : 0); //<vs_non-Newtonian>
                  //deltap1=(boundp2? FLT_MAX: deltap1+delta);
                  deltap1 = (boundp2 && CTE.tboundary == BC_DBC ? FLT_MAX : deltap1 + delta);
              }
              //-Density Diffusion Term (Fourtakas et al 2019). //<vs_dtt2_ini>
              if ((tdensity == DDT_DDT2 || (tdensity == DDT_DDT2Full && !boundp2)) && deltap1 != FLT_MAX && !ftp2) {
                  const float rh = 1.f + CTE.ddtgz * drz;
                  const float drhop = CTE.rhopzero * pow(rh, 1.f / CTE.gamma) - CTE.rhopzero;
                  const float visc_densi = CTE.ddtkh * cbar * ((velrhop2.w - velrhop1.w) - drhop) / (rr2 + CTE.eta2);
                  const float delta = (pp1 == pp2 ? visc_densi * dot3 * massp2 / velrhop2.w : 0); //<vs_non-Newtonian>
                  deltap1 = (boundp2 ? FLT_MAX : deltap1 - delta);
              } //<vs_dtt2_end>		

              //-Shifting correction.
              if (shift && shiftposfsp1.x != FLT_MAX) {
                  bool heavyphase = (PHASEARRAY[pp1].mass > PHASEARRAY[pp2].mass && pp1 != pp2 ? true : false); //<vs_non-Newtonian>
                  const float massrhop = (USE_FLOATING ? ftmassp2 : massp2) / velrhop2.w;
                  const bool noshift = (boundp2 && (shiftmode == SHIFT_NoBound || (shiftmode == SHIFT_NoFixed && CODE_IsFixed(code[p2]))));
                  shiftposfsp1.x = (noshift ? FLT_MAX : (heavyphase ? 0 : shiftposfsp1.x + massrhop * frx)); //-Removes shifting for the boundaries. | Con boundary anula shifting.
                  shiftposfsp1.y += (heavyphase ? 0 : massrhop * fry);
                  shiftposfsp1.z += (heavyphase ? 0 : massrhop * frz);
                  shiftposfsp1.w -= (heavyphase ? 0 : massrhop * dot3);
              }

              //===== Viscosity ===== 
              if (compute) {
                  const float dot = drx * dvx + dry * dvy + drz * dvz;
                  const float dot_rr2 = dot / (rr2 + CTE.eta2);
                  visc = max(dot_rr2, visc);  //ViscDt=max(dot/(rr2+Eta2),ViscDt);
                  if((dot<0) && (pp2==0) && tvisco == VISCO_SoilWater) {
                      const float amubar=CTE.kernelh*dot_rr2;  //amubar=CTE.h*dot/(rr2+CTE.eta2);
                      const float robar=(velrhop1.w+velrhop2.w)*0.5f;
                      const float visco_NN;
                      visco_NN = PHASEARRAY[pp2].Cs0;
                      const float pi_visc=(visco_NN*cbar*amubar/robar)*massp2;
                      acep1.x +=pi_visc*frx; acep1.y +=pi_visc*fry; acep1.z +=pi_visc*frz;
                  }

                  if (tvisco != VISCO_Artificial) { //&& !boundp2
                    //vel gradients
                      if (boundp2) {
                          dvx = 2.f * velrhop1.x; dvy = 2.f * velrhop1.y; dvz = 2.f * velrhop1.z;  //fomraly I should use the moving BC vel as ug=2ub-uf
                      }
                      // When calculate the fluid phase viscosity use artificial method for the multilayer setup, no need the vel gradient for fluids
                      //GetVelocityGradients_SPH_tsym(massp2, velrhop2, dvx, dvy, dvz, frx, fry, frz, grap1_xx_xy, grap1_xz_yy, grap1_yz_zz);
                  }
              }
          }
      }
  }

///  Interaction of a soil particle (p1) with a set of particles (p2) either of fluid or granular. (Multilayer)
/*template<TpKernel tker, TpFtMode ftmode, TpVisco tvisco, TpDensity tdensity, bool shift, bool symm>
__device__ void KerInteractionForcesMultilayerGranularBox_NN_SPH_PressGrad(bool boundp2, unsigned p1
      , const unsigned& pini, const unsigned& pfin
      , const float* ftomassp
      , const float4* poscell
      , const float4* velrhop, const typecode* code, const unsigned* idp
      , float massp2, const typecode pp1, bool ftp1
      , const float4& pscellp1, const float4& velrhop1, float pressp1
      , float2& grap1_xx_xy, float2& grap1_xz_yy, float2& grap1_yz_zz
      , float3& acep1, float& arp1, float& visc, float& deltap1
      , TpShifting shiftmode, float4& shiftposfsp1)
  {
      for (int p2 = pini; p2 < pfin; p2++) {
          const float4 pscellp2 = poscell[p2];
          float drx = pscellp1.x - pscellp2.x + CTE.poscellsize * (CEL_GetX(__float_as_int(pscellp1.w)) - CEL_GetX(__float_as_int(pscellp2.w)));
          float dry = pscellp1.y - pscellp2.y + CTE.poscellsize * (CEL_GetY(__float_as_int(pscellp1.w)) - CEL_GetY(__float_as_int(pscellp2.w)));
          float drz = pscellp1.z - pscellp2.z + CTE.poscellsize * (CEL_GetZ(__float_as_int(pscellp1.w)) - CEL_GetZ(__float_as_int(pscellp2.w)));
          if (symm)dry = pscellp1.y + pscellp2.y + CTE.poscellsize * CEL_GetY(__float_as_int(pscellp2.w)); //<vs_syymmetry>
          const float rr2 = drx * drx + dry * dry + drz * drz;
          if (rr2 <= CTE.kernelsize2 && rr2 >= ALMOSTZERO) {
              //-Computes kernel.
              const float fac = cufsph::GetKernel_Fac<tker>(rr2);
              const float frx = fac * drx, fry = fac * dry, frz = fac * drz; //-Gradients.
              const float wac = cufsph::GetKernel_Wab<tker>(rr2);

              //-Obtains mass of particle p2 for NN and if any floating bodies exist.
              const typecode cod = code[p2];
              const typecode pp2 = (boundp2 ? pp1 : CODE_GetTypeValue(cod)); //<vs_non-Newtonian>
              float massp2; //massp2 not neccesary to go in _Box function
              if(pp2 == 0) massp2 = (boundp2 ? CTE.massb : PHASEARRAY[pp2].mass); // p2 is fluid
              if(pp2 == 1) massp2 = (boundp2 ? CTE.massb : PHASEDRUCKERPRAGER[pp2].mass); // p2 is granular
              //Note if you masses are very different more than a ratio of 1.3 then: massp2 = (boundp2 ? PHASEARRAY[pp1].mass : PHASEARRAY[pp2].mass);

              //-Obtiene masa de particula p2 en caso de existir floatings.
              bool ftp2 = false;        //-Indicates if it is floating. | Indica si es floating.
              float ftmassp2;						//-Contains mass of floating body or massf if fluid. | Contiene masa de particula floating o massp2 si es bound o fluid.
              bool compute = true;			//-Deactivated when DEM is used and is float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
              if (USE_FLOATING) {
                  const typecode cod = code[p2];
                  ftp2 = CODE_IsFloating(cod);
                  ftmassp2 = (ftp2 ? ftomassp[CODE_GetTypeValue(cod)] : massp2);
#ifdef DELTA_HEAVYFLOATING
                  if (ftp2 && tdensity == DDT_DDT && ftmassp2 <= (massp2 * 1.2f))deltap1 = FLT_MAX;
#else
                  if (ftp2 && tdensity == DDT_DDT)deltap1 = FLT_MAX;
#endif
                  if (ftp2 && shift && shiftmode == SHIFT_NoBound)shiftposfsp1.x = FLT_MAX; //-Cancels shifting with floating bodies. | Con floatings anula shifting.
                  compute = !(USE_FTEXTERNAL && ftp1 && (boundp2 || ftp2)); //-Deactivated when DEM or Chrono is used and is float-float or float-bound. | Se desactiva cuando se usa DEM o Chrono y es float-float o float-bound.
              }
              float4 velrhop2 = velrhop[p2];
              if (symm)velrhop2.y = -velrhop2.y; //<vs_syymmetry>
              float dvx=velrhop1.x-velrhop2.x,dvy=velrhop1.y-velrhop2.y,dvz=velrhop1.z-velrhop2.z;
              float dv = sqrt(dvx*dvx + dvy*dvy + dvz*dvz);

              //===== Aceleration ===== 
              if (compute) {
                 if (pp2 == 0) { // fluid
                   const float pressp2 = cufsph::ComputePressCte_NN(velrhop2.w, PHASEARRAY[pp2].rho, PHASEARRAY[pp2].CteB, PHASEARRAY[pp2].Gamma, PHASEARRAY[pp2].Cs0, cod);
                   const float prs = VolFracp1 * (pressp2 - pressp1) / (VolFracp2 * velrhop2.w * velrhop1.w) + (tker == KERNEL_Cubic ? cufsph::GetKernelCubic_Tensil(rr2, velrhop1.w, pressp1, velrhop2.w, pressp2) : 0);
                   const float p_vpm = -prs * (USE_FLOATING ? ftmassp2 : massp2);

                   const float fd_x, fd_y, fd_z;
                   const float Drag_alphad = PHASEDRUCKERPRAGER[pp1].Drag_alphad;
                   const float visco = PhaseCte[pp2].visco;
                   const float DP_Dc = PHASEDRUCKERPRAGER[pp1].DP_Dc;
                   const float Drag_betad = PHASEDRUCKERPRAGER[pp1].Drag_betad;
                   fd_x = Drag_alpha * visco * (1 - VolFracp2)*(1 - VolFracp2) / VolFracp2 / DP_Dc / DP_Dc * (-dvx) + Drag_betad * velrhop2.w * (1 - VolFracp2) / DP_Dc * dv * (-dvx); 
                   fd_y = Drag_alpha * visco * (1 - VolFracp2)*(1 - VolFracp2) / VolFracp2 / DP_Dc / DP_Dc * (-dvy) + Drag_betad * velrhop2.w * (1 - VolFracp2) / DP_Dc * dv * (-dvy);
                   fd_z = Drag_alpha * visco * (1 - VolFracp2)*(1 - VolFracp2) / VolFracp2 / DP_Dc / DP_Dc * (-dvz) + Drag_betad * velrhop2.w * (1 - VolFracp2) / DP_Dc * dv * (-dvz);
                   acep1.x += p_vpm * frx + fd_x*massp2/(velrhop1.w * velrhop2.w * VolFracp2)*wac; acep1.y += p_vpm * fry + fd_y*massp2/(velrhop1.w * velrhop2.w * VolFracp2)*wac; acep1.z += p_vpm * frz + fd_z*massp2/(velrhop1.w * velrhop2.w * VolFracp2)*wac;
                 }
                 // Skip if pp2 == 1 is soil, will be calcuated in conseq
              }

              //-Density derivative.
              if (compute){
                if (pp2 == 1) {
                  float dvx = velrhop1.x - velrhop2.x, dvy = velrhop1.y - velrhop2.y, dvz = velrhop1.z - velrhop2.z;
                  arp1 += velrhop1.w * (USE_FLOATING ? ftmassp2 : massp2) / velrhop2.w * (dvx * frx + dvy * fry + dvz * frz);
                }
              }

              if (TVISCO != VISCO_SoilWater) const float cbar = max(PHASEARRAY[pp1].Cs0, PHASEARRAY[pp2].Cs0);
              else if (pp2 == 1) const float cbar = PHASEDRUCKERPRAGER[pp1].Cs0;
              else const float cbar = 0;
              const float dot3 = (tdensity != DDT_None || shift ? drx * frx + dry * fry + drz * frz : 0);
              //-Density derivative (DeltaSPH Molteni).
              if (tdensity == DDT_DDT && deltap1 != FLT_MAX) {
                  const float rhop1over2 = velrhop1.w / velrhop2.w;
                  const float visc_densi = CTE.ddtkh * cbar * (rhop1over2 - 1.f) / (rr2 + CTE.eta2);
                  const float delta = (pp1 == pp2 ? visc_densi * dot3 * (USE_FLOATING ? ftmassp2 : massp2) : 0); //<vs_non-Newtonian>
                  //deltap1=(boundp2? FLT_MAX: deltap1+delta);
                  deltap1 = (boundp2 && CTE.tboundary == BC_DBC ? FLT_MAX : deltap1 + delta);
              }
              //-Density Diffusion Term (Fourtakas et al 2019). //<vs_dtt2_ini>
              if ((tdensity == DDT_DDT2 || (tdensity == DDT_DDT2Full && !boundp2)) && deltap1 != FLT_MAX && !ftp2) {
                  const float rh = 1.f + CTE.ddtgz * drz;
                  const float drhop = CTE.rhopzero * pow(rh, 1.f / CTE.gamma) - CTE.rhopzero;
                  const float visc_densi = CTE.ddtkh * cbar * ((velrhop2.w - velrhop1.w) - drhop) / (rr2 + CTE.eta2);
                  const float delta = (pp1 == pp2 ? visc_densi * dot3 * massp2 / velrhop2.w : 0); //<vs_non-Newtonian>
                  deltap1 = (boundp2 ? FLT_MAX : deltap1 - delta);
              } //<vs_dtt2_end>		

              //-Shifting correction.
              if (shift && shiftposfsp1.x != FLT_MAX) {
                  bool heavyphase = (PHASEARRAY[pp1].mass > PHASEARRAY[pp2].mass && pp1 != pp2 ? true : false); //<vs_non-Newtonian>
                  const float massrhop = (USE_FLOATING ? ftmassp2 : massp2) / velrhop2.w;
                  const bool noshift = (boundp2 && (shiftmode == SHIFT_NoBound || (shiftmode == SHIFT_NoFixed && CODE_IsFixed(code[p2]))));
                  shiftposfsp1.x = (noshift ? FLT_MAX : (heavyphase ? 0 : shiftposfsp1.x + massrhop * frx)); //-Removes shifting for the boundaries. | Con boundary anula shifting.
                  shiftposfsp1.y += (heavyphase ? 0 : massrhop * fry);
                  shiftposfsp1.z += (heavyphase ? 0 : massrhop * frz);
                  shiftposfsp1.w -= (heavyphase ? 0 : massrhop * dot3);
              }

              //===== Viscosity ===== 
              if (compute) {
                  const float dot = drx * dvx + dry * dvy + drz * dvz;
                  const float dot_rr2 = dot / (rr2 + CTE.eta2);
                  visc = max(dot_rr2, visc);  //ViscDt=max(dot/(rr2+Eta2),ViscDt);
                  if((dot<0) && (pp2==1) && tvisco == VISCO_SoilWater) {
                      const float amubar=CTE.kernelh*dot_rr2;  //amubar=CTE.h*dot/(rr2+CTE.eta2);
                      const float robar=(velrhop1.w+velrhop2.w)*0.5f;
                      const float visco_NN;
                      visco_NN = PHASEDRUCKERPRAGER[pp2].Cs0;
                      const float pi_visc=(visco_NN*cbar*amubar/robar)*massp2;
                      acep1.x +=pi_visc*frx; acep1.y +=pi_visc*fry; acep1.z +=pi_visc*frz;
                  }

                  if (tvisco != VISCO_Artificial) { //&& !boundp2
                    //vel gradients
                      if (boundp2) {
                          dvx = 2.f * velrhop1.x; dvy = 2.f * velrhop1.y; dvz = 2.f * velrhop1.z;  //fomraly I should use the moving BC vel as ug=2ub-uf
                      }
                      GetVelocityGradients_SPH_tsym(massp2, velrhop2, dvx, dvy, dvz, frx, fry, frz, grap1_xx_xy, grap1_xz_yy, grap1_yz_zz);
                  }
              }
          }
      }
  }
*/
//------------------------------------------------------------------------------
/// Interaction between particles for non-Newtonian models using the SPH approach. Fluid/Float-Fluid/Float or Fluid/Float-Bound.
/// Includes pressure calculations, velocity gradients and normal/DEM floating bodies.
///
/// Realiza interaccion entre particulas para modelos no-Newtonianos que utilizan el enfoque de la SPH. Fluid/Float-Fluid/Float or Fluid/Float-Bound
/// Incluye visco artificial/laminar y floatings normales/dem.
//------------------------------------------------------------------------------
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,TpDensity tdensity,bool shift,bool symm>
__global__ void KerInteractionForcesFluid_NN_SPH_PressGrad(unsigned n,unsigned pinit
  ,int scelldiv,int4 nc,int3 cellzero,const int2 *begincell,unsigned cellfluid,const unsigned *dcell
  ,const float *ftomassp,float2 *gradvelff
  ,const float4 *poscell
  ,const float4 *velrhop,const typecode *code,const unsigned *idp
  ,float *viscdt,float *ar,float3 *ace,float *delta
  ,TpShifting shiftmode,float4 *shiftposfs, float *volfrac)
{
  const unsigned p=blockIdx.x*blockDim.x+threadIdx.x; //-Number of particle.
  float usp1 = 0.0f; // soil interpolation velocity at fluid particle
  float prep1 = 0.0f; // water interpolation pressure at soil particle

  if(p<n) {
    unsigned p1=p+pinit;      //-Number of particle.
    float visc=0,arp1=0,deltap1=0;
    float3 acep1=make_float3(0,0,0);

    //-Variables for Shifting.
    float4 shiftposfsp1;
    if(shift)shiftposfsp1=shiftposfs[p1];

    //-Obtains data of particle p1 in case there are floating bodies.
    //-Obtiene datos de particula p1 en caso de existir floatings.
    bool ftp1;       //-Indicates if it is floating. | Indica si es floating.
    const typecode cod=code[p1];
    if(USE_FLOATING) {
      ftp1=CODE_IsFloating(cod);
      if(ftp1 && tdensity!=DDT_None)deltap1=FLT_MAX; //-DDT is not applied to floating particles.
      if(ftp1 && shift)shiftposfsp1.x=FLT_MAX; //-Shifting is not calculated for floating bodies. | Para floatings no se calcula shifting.
    }

    //-Obtains basic data of particle p1.		
    const float4 pscellp1=poscell[p1];
    const float4 velrhop1=velrhop[p1];
    const float volfracp1 = volfrac[p1];
    //<vs_non-Newtonian>
    const typecode pp1=CODE_GetTypeValue(cod);

    //Obtain pressure
    //Let typecode = 0 refers to the fluid phase (mkfluid = 0) 
    if (pp1 == 0)const float pressp1=cufsph::ComputePressCte_NN(velrhop1.w,PHASEARRAY[pp1].rho,PHASEARRAY[pp1].CteB,PHASEARRAY[pp1].Gamma,PHASEARRAY[pp1].Cs0,cod);
    const bool rsymp1=(symm && CEL_GetPartY(__float_as_uint(pscellp1.w))==0); //<vs_syymmetry>

    //-Variables for vel gradients
    float2 grap1_xx_xy,grap1_xz_yy,grap1_yz_zz;
    if(tvisco!=VISCO_Artificial) {
      grap1_xx_xy=make_float2(0,0);
      grap1_xz_yy=make_float2(0,0);
      grap1_yz_zz=make_float2(0,0);
    }

    //-Obtains neighborhood search limits.
    int ini1,fin1,ini2,fin2,ini3,fin3;
    cunsearch::InitCte(dcell[p1],scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

    //-Interaction with fluids/granulars.
    ini3+=cellfluid; fin3+=cellfluid;
    for(int c3=ini3; c3<fin3; c3+=nc.w)for(int c2=ini2; c2<fin2; c2+=nc.x) {
      unsigned pini,pfin=0; cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
      if(pfin) {
          if (TVisco != VISCO_SoilWater){
              KerInteractionForcesFluidBox_NN_SPH_PressGrad<tker, ftmode, tvisco, tdensity, shift, false>(false, p1, pini, pfin, ftomassp, poscell, velrhop, code, idp, CTE.massf, pp1, ftp1, pscellp1, velrhop1, pressp1, grap1_xx_xy, grap1_xz_yy, grap1_yz_zz, acep1, arp1, visc, deltap1, shiftmode, shiftposfsp1);
              if (symm && rsymp1)	KerInteractionForcesFluidBox_NN_SPH_PressGrad<tker, ftmode, tvisco, tdensity, shift, true >(false, p1, pini, pfin, ftomassp, poscell, velrhop, code, idp, CTE.massf, pp1, ftp1, pscellp1, velrhop1, pressp1, grap1_xx_xy, grap1_xz_yy, grap1_yz_zz, acep1, arp1, visc, deltap1, shiftmode, shiftposfsp1); //<vs_syymmetry>
          }
          if (TVisco == VISCO_SoilWater && pp1 == 0) {         //-If p1 is fluid
              usp1 = 0;
              KerInterpolationVelGranularToFluid<tker, false>(pini, pfin, poscell, velrhop, code, pscellp1, usp1);
              // if (symm && rsymp1) KerInterpolationVelGranularToFluid<tker, true>(pini, pfin, poscell, velrhop, code, pscellp1, usp1);
              KerInteractionForcesMultilayerFluidBox_NN_SPH_PressGrad<tker, ftmode, tvisco, tdensity, shift, false>(false, p1, pini, pfin, ftomassp, poscell, velrhop, code, idp, CTE.massf, pp1, ftp1, pscellp1, velrhop1, pressp1, grap1_xx_xy, grap1_xz_yy, grap1_yz_zz, acep1, arp1, visc, deltap1, shiftmode, shiftposfsp1, usp1, VolFracp1, VolFrac);
              if (symm && rsymp1)	KerInteractionForcesMultilayerFluidBox_NN_SPH_PressGrad<tker, ftmode, tvisco, tdensity, shift, true >(false, p1, pini, pfin, ftomassp, poscell, velrhop, code, idp, CTE.massf, pp1, ftp1, pscellp1, velrhop1, pressp1, grap1_xx_xy, grap1_xz_yy, grap1_yz_zz, acep1, arp1, visc, deltap1, shiftmode, shiftposfsp1, usp1, VolFracp1, VolFrac); //<vs_syymmetry>
          }
          if (TVisco == VISCO_SoilWater && pp1 == 1) {         //-If p1 is granular 
              prep1 = 0;
              KerInterpolationPFluidToGranular<tker, tvisco, false>(pini, pfin, poscell, velrhop, code, pscellp1, pressp1, prep1, volfrac);
              // if (symm && rsymp1) KerInterpolationPFluidToGranular<tker, tvisco, true>(pini, pfin, poscell, velrhop, code, pscellp1, pressp1, prep1, volfrac);
              KerInteractionForcesMultilayerGranularBox_NN_SPH_PressGrad<tker, ftmode, tvisco, tdensity, shift, false>(false, p1, pini, pfin, ftomassp, poscell, velrhop, code, idp, CTE.massf, pp1, ftp1, pscellp1, velrhop1, pressp1, grap1_xx_xy, grap1_xz_yy, grap1_yz_zz, acep1, arp1, visc, deltap1, shiftmode, shiftposfsp1, prep1, VolFracp1, VolFrac);
              if (symm && rsymp1)	KerInteractionForcesMultilayerGranularBox_NN_SPH_PressGrad<tker, ftmode, tvisco, tdensity, shift, true >(false, p1, pini, pfin, ftomassp, poscell, velrhop, code, idp, CTE.massf, pp1, ftp1, pscellp1, velrhop1, pressp1, grap1_xx_xy, grap1_xz_yy, grap1_yz_zz, acep1, arp1, visc, deltap1, shiftmode, shiftposfsp1, prep1, VolFracp1, VolFrac); //<vs_syymmetry>
          }
    }
    //-Interaction with boundaries.
    ini3-=cellfluid; fin3-=cellfluid;
    for(int c3=ini3; c3<fin3; c3+=nc.w)for(int c2=ini2; c2<fin2; c2+=nc.x) {
      unsigned pini,pfin=0; cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
      if(pfin) {
        KerInteractionForcesFluidBox_NN_SPH_PressGrad<tker,ftmode,tvisco,tdensity,shift,false>(true,p1,pini,pfin,ftomassp,poscell,velrhop,code,idp,CTE.massb,pp1,ftp1,pscellp1,velrhop1,pressp1,grap1_xx_xy,grap1_xz_yy,grap1_yz_zz,acep1,arp1,visc,deltap1,shiftmode,shiftposfsp1);
        if(symm && rsymp1)	KerInteractionForcesFluidBox_NN_SPH_PressGrad<tker,ftmode,tvisco,tdensity,shift,true >(true,p1,pini,pfin,ftomassp,poscell,velrhop,code,idp,CTE.massb,pp1,ftp1,pscellp1,velrhop1,pressp1,grap1_xx_xy,grap1_xz_yy,grap1_yz_zz,acep1,arp1,visc,deltap1,shiftmode,shiftposfsp1); //<vs_syymmetry>
      }
    }
    //-Stores results.
    if(shift||arp1||acep1.x||acep1.y||acep1.z||visc) {
      if(tdensity!=DDT_None) {
        if(delta) {
          const float rdelta=delta[p1];
          delta[p1]=(rdelta==FLT_MAX||deltap1==FLT_MAX ? FLT_MAX : rdelta+deltap1);
        }
        else if(deltap1!=FLT_MAX)arp1+=deltap1;
      }
      ar[p1]+=arp1;
      float3 r=ace[p1]; r.x+=acep1.x; r.y+=acep1.y; r.z+=acep1.z; ace[p1]=r;
      if(visc>viscdt[p1])viscdt[p1]=visc;
      if(tvisco!=VISCO_Artificial) {
        float2 rg;
        rg=gradvelff[p1*3];		 rg=make_float2(rg.x+grap1_xx_xy.x,rg.y+grap1_xx_xy.y);  gradvelff[p1*3]=rg;
        rg=gradvelff[p1*3+1];  rg=make_float2(rg.x+grap1_xz_yy.x,rg.y+grap1_xz_yy.y);  gradvelff[p1*3+1]=rg;
        rg=gradvelff[p1*3+2];  rg=make_float2(rg.x+grap1_yz_zz.x,rg.y+grap1_yz_zz.y);  gradvelff[p1*3+2]=rg;
      }
      if(shift)shiftposfs[p1]=shiftposfsp1;
      //auxnn[p1] = visco_etap1; // to be used if an auxilary is needed.
    }
  }
}
}
//==============================================================================
/// Interaction for the force computation using the SPH approach.
/// Interaccion para el calculo de fuerzas que utilizan el enfoque de la SPH .
//==============================================================================
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,TpDensity tdensity,bool shift>
void Interaction_ForcesGpuT_NN_SPH(const StInterParmsg &t)
{
  //-Collects kernel information.
#ifndef DISABLE_BSMODES
  if(t.kerinfo) {
    cusph::Interaction_ForcesT_KerInfo<tker,ftmode,true,tdensity,shift,false>(t.kerinfo);
    return;
  }
#endif
  const StDivDataGpu &dvd=t.divdatag;
  const int2* beginendcell=dvd.beginendcell;
  //-Interaction Fluid-Fluid & Fluid-Bound.
  if(t.fluidnum) {
    dim3 sgridf=GetSimpleGridSize(t.fluidnum,t.bsfluid);
    if(t.symmetry){ //<vs_syymmetry_ini>
      KerInteractionForcesFluid_NN_SPH_PressGrad<tker,ftmode,tvisco,tdensity,shift,true ><<<sgridf,t.bsfluid,0,t.stm>>>
        (t.fluidnum,t.fluidini,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell,dvd.cellfluid,t.dcell
          ,t.ftomassp,(float2*)t.gradvel,t.poscell,t.velrhop,t.code,t.idp
          ,t.viscdt,t.ar,t.ace,t.delta,t.shiftmode,t.shiftposfs,t.volfrac);

      if(tvisco !=VISCO_SoilWater)KerInteractionForcesFluid_NN_SPH_Visco_eta<ftmode,tvisco,true ><<<sgridf,t.bsfluid,0,t.stm>>>
        (t.fluidnum,t.fluidini,t.viscob,t.visco_eta,t.velrhop,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell,dvd.cellfluid,t.dcell
          ,(float2*)t.d_tensor,(float2*)t.gradvel,t.code,t.idp
          ,t.viscetadt);
      //choice of visc formulation
      if(tvisco!=VISCO_ConstEq || tvisco !=VISCO_SoilWater) KerInteractionForcesFluid_NN_SPH_Morris<tker,ftmode,tvisco,true ><<<sgridf,t.bsfluid,0,t.stm>>>
        (t.fluidnum,t.fluidini,t.viscob,t.viscof,t.visco_eta,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell,dvd.cellfluid,t.dcell
          ,t.ftomassp,t.auxnn,t.poscell,t.velrhop,t.code,t.idp
          ,t.ace);
      if (tvisco==VISCO_ConstEq || tvisco ==VISCO_SoilWater) {
        // Build stress tensor
        KerInteractionForcesFluid_NN_SPH_Visco_Stress_tensor<ftmode,tvisco,true ><<<sgridf,t.bsfluid,0,t.stm>>>
          (t.fluidnum,t.fluidini,t.visco_eta,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell,dvd.cellfluid,t.dcell
            ,t.ftomassp,(float2*)t.tau,(float2*)t.pstrain,(float2*)t.d_tensor, (float2*)t.gradvel,t.auxnn,t.poscell,t.velrhop,t.code,t.idp,dt);
        //Get stresses
        KerInteractionForcesFluid_NN_SPH_ConsEq<tker,ftmode,tvisco,true ><<<sgridf,t.bsfluid,0,t.stm>>>
          (t.fluidnum,t.fluidini,t.viscob,t.viscof,t.visco_eta,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell,dvd.cellfluid,t.dcell
            ,t.ftomassp,(float2*)t.tau,t.auxnn,t.poscell,t.velrhop,t.code,t.idp
            ,t.ace);
      }

    }
    else {//<vs_syymmetry_end>			
      KerInteractionForcesFluid_NN_SPH_PressGrad<tker,ftmode,tvisco,tdensity,shift,false ><<<sgridf,t.bsfluid,0,t.stm>>>
        (t.fluidnum,t.fluidini,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell,dvd.cellfluid,t.dcell
          ,t.ftomassp,(float2*)t.gradvel,t.poscell,t.velrhop,t.code,t.idp
          ,t.viscdt,t.ar,t.ace,t.delta,t.shiftmode,t.shiftposfs,t.volfrac);

      if(tvisco !=VISCO_SoilWater)KerInteractionForcesFluid_NN_SPH_Visco_eta<ftmode,tvisco,false ><<<sgridf,t.bsfluid,0,t.stm>>>
        (t.fluidnum,t.fluidini,t.viscob,t.visco_eta,t.velrhop,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell,dvd.cellfluid,t.dcell
          ,(float2*)t.d_tensor,(float2*)t.gradvel,t.code,t.idp
          ,t.viscetadt);
      //choice of visc formulation
      if(tvisco!=VISCO_ConstEq tvisco !=VISCO_SoilWater) KerInteractionForcesFluid_NN_SPH_Morris<tker,ftmode,tvisco,false ><<<sgridf,t.bsfluid,0,t.stm>>>
        (t.fluidnum,t.fluidini,t.viscob,t.viscof,t.visco_eta,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell,dvd.cellfluid,t.dcell
          ,t.ftomassp,t.auxnn,t.poscell,t.velrhop,t.code,t.idp
          ,t.ace);
      if (tvisco==VISCO_ConstEq || tvisco ==VISCO_SoilWater) {
        // Build stress tensor				
        KerInteractionForcesFluid_NN_SPH_Visco_Stress_tensor<ftmode,tvisco,false ><<<sgridf,t.bsfluid,0,t.stm>>>
          (t.fluidnum,t.fluidini,t.visco_eta,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell,dvd.cellfluid,t.dcell
            ,t.ftomassp,(float2*)t.tau,(float2*)t.pstrain,(float2*)t.d_tensor,t.auxnn,t.poscell,t.velrhop,t.code,t.idp);
        //Get stresses
        KerInteractionForcesFluid_NN_SPH_ConsEq<tker,ftmode,tvisco,false ><<<sgridf,t.bsfluid,0,t.stm>>>
          (t.fluidnum,t.fluidini,t.viscob,t.viscof,t.visco_eta,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell,dvd.cellfluid,t.dcell
            ,t.ftomassp,(float2*)t.tau,t.auxnn,t.poscell,t.velrhop,t.code,t.idp
            ,t.ace);
      }
    }
  }

  //-Interaction Boundary-Fluid.
  if(t.boundnum) {
    const int2* beginendcellfluid=dvd.beginendcell+dvd.cellfluid;
    dim3 sgridb=GetSimpleGridSize(t.boundnum,t.bsbound);
    //printf("bsbound:%u\n",bsbound);
    if(t.symmetry) //<vs_syymmetry_ini>
      KerInteractionForcesBound_NN<tker,ftmode,true ><<<sgridb,t.bsbound,0,t.stm>>>
      (t.boundnum,t.boundini,dvd.scelldiv,dvd.nc,dvd.cellzero,beginendcell+dvd.cellfluid,t.dcell
        ,t.ftomassp,t.poscell,t.velrhop,t.code,t.idp,t.viscdt,t.ar);
    else //<vs_syymmetry_end>
      KerInteractionForcesBound_NN<tker,ftmode,false><<<sgridb,t.bsbound,0,t.stm>>>
      (t.boundnum,t.boundini,dvd.scelldiv,dvd.nc,dvd.cellzero,beginendcellfluid,t.dcell
        ,t.ftomassp,t.poscell,t.velrhop,t.code,t.idp,t.viscdt,t.ar);
  }
}
//======================END of SPH==============================================

//======================Start of non-Newtonian Templates=======================================
//Uncomment for fast compile 
//#define FAST_COMPILATION
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,TpDensity tdensity,bool shift> void Interaction_ForcesGpuT_NN(const StInterParmsg &t) {
#ifdef FAST_COMPILATION
  if(t.tvelgrad!=VELGRAD_FDA)throw "Extra SPH Gradients options are disabled for FastCompilation...";
  Interaction_ForcesGpuT_NN_FDA	    < tker,ftmode,tvisco,tdensity,shift>(t);
#else	
  if(t.tvelgrad==VELGRAD_FDA) Interaction_ForcesGpuT_NN_FDA	    < tker,ftmode,tvisco,tdensity,shift>(t);
  else if(t.tvelgrad==VELGRAD_SPH)	Interaction_ForcesGpuT_NN_SPH		< tker,ftmode,tvisco,tdensity,shift>(t);
#endif
}
//==============================================================================
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco> void Interaction_ForcesNN_gt2(const StInterParmsg &t) {
#ifdef FAST_COMPILATION
  if(!t.shiftmode||t.tdensity!=DDT_DDT2Full)throw "Shifting and extra DDT are disabled for FastCompilation...";
  Interaction_ForcesGpuT_NN<tker,ftmode,tvisco,DDT_DDT2Full,true>(t);
#else
  if(t.shiftmode) {
    const bool shift=true;
    if(t.tdensity==DDT_None)    Interaction_ForcesGpuT_NN<tker,ftmode,tvisco,DDT_None,shift>(t);
    if(t.tdensity==DDT_DDT)     Interaction_ForcesGpuT_NN<tker,ftmode,tvisco,DDT_DDT,shift>(t);
    if(t.tdensity==DDT_DDT2)    Interaction_ForcesGpuT_NN<tker,ftmode,tvisco,DDT_DDT2,shift>(t);  //<vs_dtt2>
    if(t.tdensity==DDT_DDT2Full)Interaction_ForcesGpuT_NN<tker,ftmode,tvisco,DDT_DDT2Full,shift>(t);  //<vs_dtt2>
  }
  else {
    const bool shift=false;
    if(t.tdensity==DDT_None)    Interaction_ForcesGpuT_NN<tker,ftmode,tvisco,DDT_None,shift>(t);
    if(t.tdensity==DDT_DDT)     Interaction_ForcesGpuT_NN<tker,ftmode,tvisco,DDT_DDT,shift>(t);
    if(t.tdensity==DDT_DDT2)    Interaction_ForcesGpuT_NN<tker,ftmode,tvisco,DDT_DDT2,shift>(t);  //<vs_dtt2>
    if(t.tdensity==DDT_DDT2Full)Interaction_ForcesGpuT_NN<tker,ftmode,tvisco,DDT_DDT2Full,shift>(t);  //<vs_dtt2>
  }
#endif
}
//==============================================================================
template<TpKernel tker,TpFtMode ftmode> void Interaction_ForcesNN_gt1(const StInterParmsg &t) {
  //GFCheck how to add fast compilation of laminar viscosity
#ifdef FAST_COMPILATION
  if(t.tvisco!=VISCO_LaminarSPS)throw "Extra viscosity options are disabled for FastCompilation...";
  Interaction_ForcesNN_gt2<tker,ftmode,VISCO_LaminarSPS>(t);
#else
  if(t.tvisco==VISCO_ConstEq)		      Interaction_ForcesNN_gt2<tker,ftmode,VISCO_ConstEq>(t);
  else if(t.tvisco==VISCO_LaminarSPS)	Interaction_ForcesNN_gt2<tker,ftmode,VISCO_LaminarSPS>(t);
  else if(t.tvisco==VISCO_Artificial)	Interaction_ForcesNN_gt2<tker,ftmode,VISCO_Artificial>(t);
#endif
}
//==============================================================================
template<TpKernel tker> void Interaction_ForcesNN_gt0(const StInterParmsg &t) {
#ifdef FAST_COMPILATION
  if(t.ftmode!=FTMODE_None)throw "Extra FtMode options are disabled for FastCompilation...";
  Interaction_ForcesNN_gt1<tker,FTMODE_None>(t);
#else
  if(t.ftmode==FTMODE_None)    Interaction_ForcesNN_gt1<tker,FTMODE_None>(t);
  else if(t.ftmode==FTMODE_Sph)Interaction_ForcesNN_gt1<tker,FTMODE_Sph>(t);
  else if(t.ftmode==FTMODE_Ext)Interaction_ForcesNN_gt1<tker,FTMODE_Ext>(t);
#endif
}
//==============================================================================
void Interaction_ForcesNN(const StInterParmsg &t) {
#ifdef FAST_COMPILATION
  if(t.tkernel!=KERNEL_Wendland)throw "Extra kernels are disabled for FastCompilation...";
  Interaction_ForcesNN_gt0<KERNEL_Wendland>(t);
#else
  if(t.tkernel==KERNEL_Wendland)     Interaction_ForcesNN_gt0<KERNEL_Wendland>(t);
#ifndef DISABLE_KERNELS_EXTRA
  else if(t.tkernel==KERNEL_Cubic)   Interaction_ForcesNN_gt0<KERNEL_Cubic   >(t);
#endif
#endif
}
//======================End of NN Templates=======================================

//======================Start of Multi-layer MDBC=====================================
//------------------------------------------------------------------------------
/// Perform interaction between ghost node of selected bondary and domain particle for water phase.
//------------------------------------------------------------------------------
template<TpKernel tker, bool sim2d, TpSlipMode tslip> __global__ void KerInteractionMdbcCorrectionNNFluid_Fast
(unsigned n, unsigned nbound, float determlimit, float mdbcthreshold
	, double3 mapposmin, float poscellsize, const float4 *poscell
	, int scelldiv, int4 nc, int3 cellzero, const int2 *beginendcellfluid
	, const double2 *posxy, const double *posz, const typecode *code, const unsigned *idp
	, const float3 *boundnormal, const float3 *motionvel, float4 *velrhop)
{
	const unsigned p1 = blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
	if (p1<n) {
		const float3 bnormalp1 = boundnormal[p1];
		if (bnormalp1.x != 0 || bnormalp1.y != 0 || bnormalp1.z != 0) {
			float rhopfinal = FLT_MAX;
			float3 velrhopfinal = make_float3(0, 0, 0);
			float sumwab = 0;

			//-Calculates ghost node position.
			double3 gposp1 = make_double3(posxy[p1].x + bnormalp1.x, posxy[p1].y + bnormalp1.y, posz[p1] + bnormalp1.z);
			gposp1 = (CTE.periactive != 0 ? cusph::KerUpdatePeriodicPos(gposp1) : gposp1); //-Corrected interface Position.
			const float4 gpscellp1 = cusph::KerComputePosCell(gposp1, mapposmin, poscellsize);

			//-Initializes variables for calculation.
			float rhopp1 = 0;
			float3 gradrhopp1 = make_float3(0, 0, 0);
			float3 velp1 = make_float3(0, 0, 0);                              // -Only for velocity
			tmatrix3f a_corr2; if (sim2d) cumath::Tmatrix3fReset(a_corr2); //-Only for 2D.
			tmatrix4f a_corr3; if (!sim2d)cumath::Tmatrix4fReset(a_corr3); //-Only for 3D.

			//-Obtains neighborhood search limits.
			int ini1, fin1, ini2, fin2, ini3, fin3;
			cunsearch::InitCte(gposp1.x, gposp1.y, gposp1.z, scelldiv, nc, cellzero, ini1, fin1, ini2, fin2, ini3, fin3);

			//-Boundary-Fluid interaction.
			for (int c3 = ini3; c3<fin3; c3 += nc.w)for (int c2 = ini2; c2<fin2; c2 += nc.x) {
				unsigned pini, pfin = 0;  cunsearch::ParticleRange(c2, c3, ini1, fin1, beginendcellfluid, pini, pfin);
				if (pfin)for (unsigned p2 = pini; p2<pfin; p2++) {
					const float4 pscellp2 = poscell[p2];
					float drx = gpscellp1.x - pscellp2.x + CTE.poscellsize*(CEL_GetX(__float_as_int(gpscellp1.w)) - CEL_GetX(__float_as_int(pscellp2.w)));
					float dry = gpscellp1.y - pscellp2.y + CTE.poscellsize*(CEL_GetY(__float_as_int(gpscellp1.w)) - CEL_GetY(__float_as_int(pscellp2.w)));
					float drz = gpscellp1.z - pscellp2.z + CTE.poscellsize*(CEL_GetZ(__float_as_int(gpscellp1.w)) - CEL_GetZ(__float_as_int(pscellp2.w)));
					const float rr2 = drx*drx + dry*dry + drz*drz;
					const typecode pp2 = CODE_GetTypeValue(code[p2]); //<vs_non-Newtonian>
					if (rr2 <= CTE.kernelsize2 && rr2 >= ALMOSTZERO && CODE_IsFluid(code[p2])&&pp2==0) {//-Only with domain particles in water phase (including inout).
						//-Computes kernel.
						float fac;
						const float wab = cufsph::GetKernel_WabFac<tker>(rr2, fac);
						const float frx = fac*drx, fry = fac*dry, frz = fac*drz; //-Gradients.

						//===== Get mass and volume of particle p2 =====
						const float4 velrhopp2 = velrhop[p2];
						//float massp2 = CTE.massf;

						
						float massp2 = PHASEARRAY[pp2].mass; //-Contiene masa de particula segun sea bound o fluid.

						const float volp2 = massp2 / velrhopp2.w;

						//===== Density and its gradient =====
						rhopp1 += massp2*wab;
						gradrhopp1.x += massp2*frx;
						gradrhopp1.y += massp2*fry;
						gradrhopp1.z += massp2*frz;

						//===== Kernel values multiplied by volume =====
						const float vwab = wab*volp2;
						sumwab += vwab;
						const float vfrx = frx*volp2;
						const float vfry = fry*volp2;
						const float vfrz = frz*volp2;

						//===== Velocity =====
						if (tslip != SLIP_Vel0) {
							velp1.x += vwab*velrhopp2.x;
							velp1.y += vwab*velrhopp2.y;
							velp1.z += vwab*velrhopp2.z;
						}

						//===== Matrix A for correction =====
						if (sim2d) {
							a_corr2.a11 += vwab;  a_corr2.a12 += drx*vwab;  a_corr2.a13 += drz*vwab;
							a_corr2.a21 += vfrx;  a_corr2.a22 += drx*vfrx;  a_corr2.a23 += drz*vfrx;
							a_corr2.a31 += vfrz;  a_corr2.a32 += drx*vfrz;  a_corr2.a33 += drz*vfrz;
						}
						else {
							a_corr3.a11 += vwab;  a_corr3.a12 += drx*vwab;  a_corr3.a13 += dry*vwab;  a_corr3.a14 += drz*vwab;
							a_corr3.a21 += vfrx;  a_corr3.a22 += drx*vfrx;  a_corr3.a23 += dry*vfrx;  a_corr3.a24 += drz*vfrx;
							a_corr3.a31 += vfry;  a_corr3.a32 += drx*vfry;  a_corr3.a33 += dry*vfry;  a_corr3.a34 += drz*vfry;
							a_corr3.a41 += vfrz;  a_corr3.a42 += drx*vfrz;  a_corr3.a43 += dry*vfrz;  a_corr3.a44 += drz*vfrz;
						}
					}
				}
			}

			//-Store the results.
			//--------------------
			if (sumwab >= mdbcthreshold) {
				const float3 dpos = make_float3(-bnormalp1.x, -bnormalp1.y, -bnormalp1.z); //-Boundary particle position - ghost node position.
				if (sim2d) {
					const double determ = cumath::Determinant3x3dbl(a_corr2);
					if (fabs(determ) >= determlimit) {//-Use 1e-3f (first_order) or 1e+3f (zeroth_order).
						const tmatrix3f invacorr2 = cumath::InverseMatrix3x3dbl(a_corr2, determ);
						//-GHOST NODE DENSITY IS MIRRORED BACK TO THE BOUNDARY PARTICLES.
						const float rhoghost = float(invacorr2.a11*rhopp1 + invacorr2.a12*gradrhopp1.x + invacorr2.a13*gradrhopp1.z);
						const float grx = -float(invacorr2.a21*rhopp1 + invacorr2.a22*gradrhopp1.x + invacorr2.a23*gradrhopp1.z);
						const float grz = -float(invacorr2.a31*rhopp1 + invacorr2.a32*gradrhopp1.x + invacorr2.a33*gradrhopp1.z);
						rhopfinal = (rhoghost + grx*dpos.x + grz*dpos.z);
					}
					else if (a_corr2.a11>0) {//-Determinant is small but a11 is nonzero, 0th order ANGELO.
						rhopfinal = float(rhopp1 / a_corr2.a11);
					}
					//-Ghost node velocity (0th order).
					if (tslip != SLIP_Vel0) {
						velrhopfinal.x = float(velp1.x / a_corr2.a11);
						velrhopfinal.z = float(velp1.z / a_corr2.a11);
						velrhopfinal.y = 0;
					}
				}
				else {
					const double determ = cumath::Determinant4x4dbl(a_corr3);
					if (fabs(determ) >= determlimit) {
						const tmatrix4f invacorr3 = cumath::InverseMatrix4x4dbl(a_corr3, determ);
						//-GHOST NODE DENSITY IS MIRRORED BACK TO THE BOUNDARY PARTICLES.
						const float rhoghost = float(invacorr3.a11*rhopp1 + invacorr3.a12*gradrhopp1.x + invacorr3.a13*gradrhopp1.y + invacorr3.a14*gradrhopp1.z);
						const float grx = -float(invacorr3.a21*rhopp1 + invacorr3.a22*gradrhopp1.x + invacorr3.a23*gradrhopp1.y + invacorr3.a24*gradrhopp1.z);
						const float gry = -float(invacorr3.a31*rhopp1 + invacorr3.a32*gradrhopp1.x + invacorr3.a33*gradrhopp1.y + invacorr3.a34*gradrhopp1.z);
						const float grz = -float(invacorr3.a41*rhopp1 + invacorr3.a42*gradrhopp1.x + invacorr3.a43*gradrhopp1.y + invacorr3.a44*gradrhopp1.z);
						rhopfinal = (rhoghost + grx*dpos.x + gry*dpos.y + grz*dpos.z);
					}
					else if (a_corr3.a11>0) {//-Determinant is small but a11 is nonzero, 0th order ANGELO.
						rhopfinal = float(rhopp1 / a_corr3.a11);
					}
					//-Ghost node velocity (0th order).
					if (tslip != SLIP_Vel0) {
						velrhopfinal.x = float(velp1.x / a_corr3.a11);
						velrhopfinal.y = float(velp1.y / a_corr3.a11);
						velrhopfinal.z = float(velp1.z / a_corr3.a11);
					}
				}
				//-Store the results.
				rhopfinal = (rhopfinal != FLT_MAX ? rhopfinal : CTE.rhopzero);
				if (tslip == SLIP_Vel0) {//-DBC vel=0
					velrhop[p1].w = rhopfinal;
				}
				if (tslip == SLIP_NoSlip) {//-No-Slip
					const float3 v = motionvel[p1];
					velrhop[p1] = make_float4(v.x + v.x - velrhopfinal.x, v.y + v.y - velrhopfinal.y, v.z + v.z - velrhopfinal.z, rhopfinal);
				}
				if (tslip == SLIP_FreeSlip) {//-No-Penetration and free slip    SHABA
					float3 FSVelFinal; // final free slip boundary velocity
					const float3 v = motionvel[p1];
					float motion = sqrt(v.x*v.x + v.y*v.y + v.z*v.z); // to check if boundary moving
					float norm = sqrt(bnormalp1.x*bnormalp1.x + bnormalp1.y*bnormalp1.y + bnormalp1.z*bnormalp1.z);
					float3 normal; // creating a normailsed boundary normal
					normal.x = fabs(bnormalp1.x) / norm; normal.y = fabs(bnormalp1.y) / norm; normal.z = fabs(bnormalp1.z) / norm;

					// finding the velocity componants normal and tangential to boundary 
					float3 normvel = make_float3(velrhopfinal.x*normal.x, velrhopfinal.y*normal.y, velrhopfinal.z*normal.z); // velocity in direction of normal pointin ginto fluid)
					float3 tangvel = make_float3(velrhopfinal.x - normvel.x, velrhopfinal.y - normvel.y, velrhopfinal.z - normvel.z); // velocity tangential to normal

					if (motion > 0) { // if moving boundary
						float3 normmot = make_float3(v.x*normal.x, v.y*normal.y, v.z*normal.z); // boundary motion in direction normal to boundary 
						FSVelFinal = make_float3(normmot.x + normmot.x - normvel.x, normmot.y + normmot.y - normvel.y, normmot.z + normmot.z - normvel.z);
						// only velocity in normal direction for no-penetration
						// fluid sees zero velocity in the tangetial direction
					}
					else {
						FSVelFinal = make_float3(tangvel.x - normvel.x, tangvel.y - normvel.y, tangvel.z - normvel.z);
						// tangential velocity equal to fluid velocity for free slip
						// normal velocity reversed for no-penetration
					}

					// Save the velocity and density
					velrhop[p1] = make_float4(FSVelFinal.x, FSVelFinal.y, FSVelFinal.z, rhopfinal);
				}
			}
		}
	}
}
//------------------------------------------------------------------------------
/// Perform interaction between ghost node of selected bondary and domain particle for soil phase.
//------------------------------------------------------------------------------
template<TpKernel tker, bool sim2d, TpSlipMode tslip> __global__ void KerInteractionMdbcCorrectionNNGranular_Fast
(unsigned n, unsigned nbound, float determlimit, float mdbcthreshold
	, double3 mapposmin, float poscellsize, const float4 *poscell
	, int scelldiv, int4 nc, int3 cellzero, const int2 *beginendcellfluid
	, const double2 *posxy, const double *posz, const typecode *code, const unsigned *idp
	, const float3 *boundnormal, const float3 *motionvel, float4 *velrhop,tsymatrix3f *sigma)
{
	const unsigned p1 = blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
	if (p1<n) {
		const float3 bnormalp1 = boundnormal[p1];
		if (bnormalp1.x != 0 || bnormalp1.y != 0 || bnormalp1.z != 0) {
			float rhopfinal = FLT_MAX;
			float3 velrhopfinal = make_float3(0, 0, 0);
			tsymatrix3f sigmafinal = { 0,0,0,0,0,0 };//mdbr
			float sumwab = 0;

			//-Calculates ghost node position.
			double3 gposp1 = make_double3(posxy[p1].x + bnormalp1.x, posxy[p1].y + bnormalp1.y, posz[p1] + bnormalp1.z);
			gposp1 = (CTE.periactive != 0 ? cusph::KerUpdatePeriodicPos(gposp1) : gposp1); //-Corrected interface Position.
			const float4 gpscellp1 = cusph::KerComputePosCell(gposp1, mapposmin, poscellsize);

			//-Initializes variables for calculation.
			float rhopp1 = 0;
			float3 gradrhopp1 = make_float3(0, 0, 0);
			//===== mdbr
			//-Solid stress
			tsymatrix3f sigmap1 = { 0,0,0,0,0,0 };//-First-order Stress value
			float3 gradsigmaxxp1 = make_float3(0, 0, 0);//-Stress gradient
			float3 gradsigmayyp1 = make_float3(0, 0, 0);
			float3 gradsigmazzp1 = make_float3(0, 0, 0);
			float3 gradsigmaxyp1 = make_float3(0, 0, 0);
			float3 gradsigmayzp1 = make_float3(0, 0, 0);
			float3 gradsigmaxzp1 = make_float3(0, 0, 0);
			//=====
			float3 velp1 = make_float3(0, 0, 0);                              // -Only for velocity
			tmatrix3f a_corr2; if (sim2d) cumath::Tmatrix3fReset(a_corr2); //-Only for 2D.
			tmatrix4f a_corr3; if (!sim2d)cumath::Tmatrix4fReset(a_corr3); //-Only for 3D.

			//-Obtains neighborhood search limits.
			int ini1, fin1, ini2, fin2, ini3, fin3;
			cunsearch::InitCte(gposp1.x, gposp1.y, gposp1.z, scelldiv, nc, cellzero, ini1, fin1, ini2, fin2, ini3, fin3);

			//-Boundary-Fluid interaction.
			for (int c3 = ini3; c3<fin3; c3 += nc.w)for (int c2 = ini2; c2<fin2; c2 += nc.x) {
				unsigned pini, pfin = 0;  cunsearch::ParticleRange(c2, c3, ini1, fin1, beginendcellfluid, pini, pfin);
				if (pfin)for (unsigned p2 = pini; p2<pfin; p2++) {
					const float4 pscellp2 = poscell[p2];
					float drx = gpscellp1.x - pscellp2.x + CTE.poscellsize*(CEL_GetX(__float_as_int(gpscellp1.w)) - CEL_GetX(__float_as_int(pscellp2.w)));
					float dry = gpscellp1.y - pscellp2.y + CTE.poscellsize*(CEL_GetY(__float_as_int(gpscellp1.w)) - CEL_GetY(__float_as_int(pscellp2.w)));
					float drz = gpscellp1.z - pscellp2.z + CTE.poscellsize*(CEL_GetZ(__float_as_int(gpscellp1.w)) - CEL_GetZ(__float_as_int(pscellp2.w)));
					const float rr2 = drx*drx + dry*dry + drz*drz;
					const typecode pp2 = CODE_GetTypeValue(code[p2]); //<vs_non-Newtonian>
					if (rr2 <= CTE.kernelsize2 && rr2 >= ALMOSTZERO && CODE_IsFluid(code[p2]) && pp2 == 1) {//-Only with domain particles in soil phase (including inout).
						//-Computes kernel.
						float fac;
						const float wab = cufsph::GetKernel_WabFac<tker>(rr2, fac);
						const float frx = fac*drx, fry = fac*dry, frz = fac*drz; //-Gradients.

						//===== Get mass and volume of particle p2 =====
						const float4 velrhopp2 = velrhop[p2];
						//float massp2 = CTE.massf;


						float massp2 = PHASEARRAY[pp2].mass; //-Contiene masa de particula segun sea bound o fluid.

						const float volp2 = massp2 / velrhopp2.w;

						//===== Density and its gradient =====
						rhopp1 += massp2*wab;
						gradrhopp1.x += massp2*frx;
						gradrhopp1.y += massp2*fry;
						gradrhopp1.z += massp2*frz;

						//===== Kernel values multiplied by volume =====
						const float vwab = wab*volp2;
						sumwab += vwab;
						const float vfrx = frx*volp2;
						const float vfry = fry*volp2;
						const float vfrz = frz*volp2;
						//===== mdbr
						//===== Stress value =====
						sigmap1.xx += vwab*sigmap2.xx;
						sigmap1.yy += vwab*sigmap2.yy;
						sigmap1.zz += vwab*sigmap2.zz;
						sigmap1.xy += vwab*sigmap2.xy;
						sigmap1.yz += vwab*sigmap2.yz;
						sigmap1.xz += vwab*sigmap2.xz;

						//===== Stress gradient =====
						//===== xx
						gradsigmaxxp1.x += vfrx*sigmap2.xx;
						gradsigmaxxp1.y += vfry*sigmap2.xx;
						gradsigmaxxp1.z += vfrz*sigmap2.xx;
						//===== yy
						gradsigmayyp1.x += vfrx*sigmap2.yy;
						gradsigmayyp1.y += vfry*sigmap2.yy;
						gradsigmayyp1.z += vfrz*sigmap2.yy;
						//===== zz
						gradsigmazzp1.x += vfrx*sigmap2.zz;
						gradsigmazzp1.y += vfry*sigmap2.zz;
						gradsigmazzp1.z += vfrz*sigmap2.zz;
						//===== xy
						gradsigmaxyp1.x += vfrx*sigmap2.xy;
						gradsigmaxyp1.y += vfry*sigmap2.xy;
						gradsigmaxyp1.z += vfrz*sigmap2.xy;
						//===== yz
						gradsigmayzp1.x += vfrx*sigmap2.yz;
						gradsigmayzp1.y += vfry*sigmap2.yz;
						gradsigmayzp1.z += vfrz*sigmap2.yz;
						//===== xz
						gradsigmaxzp1.x += vfrx*sigmap2.xz;
						gradsigmaxzp1.y += vfry*sigmap2.xz;
						gradsigmaxzp1.z += vfrz*sigmap2.xz;
						//===== End
						//===== Velocity =====
						if (tslip != SLIP_Vel0) {
							velp1.x += vwab*velrhopp2.x;
							velp1.y += vwab*velrhopp2.y;
							velp1.z += vwab*velrhopp2.z;
						}

						//===== Matrix A for correction =====
						if (sim2d) {
							a_corr2.a11 += vwab;  a_corr2.a12 += drx*vwab;  a_corr2.a13 += drz*vwab;
							a_corr2.a21 += vfrx;  a_corr2.a22 += drx*vfrx;  a_corr2.a23 += drz*vfrx;
							a_corr2.a31 += vfrz;  a_corr2.a32 += drx*vfrz;  a_corr2.a33 += drz*vfrz;
						}
						else {
							a_corr3.a11 += vwab;  a_corr3.a12 += drx*vwab;  a_corr3.a13 += dry*vwab;  a_corr3.a14 += drz*vwab;
							a_corr3.a21 += vfrx;  a_corr3.a22 += drx*vfrx;  a_corr3.a23 += dry*vfrx;  a_corr3.a24 += drz*vfrx;
							a_corr3.a31 += vfry;  a_corr3.a32 += drx*vfry;  a_corr3.a33 += dry*vfry;  a_corr3.a34 += drz*vfry;
							a_corr3.a41 += vfrz;  a_corr3.a42 += drx*vfrz;  a_corr3.a43 += dry*vfrz;  a_corr3.a44 += drz*vfrz;
						}
					}
				}
			}

			//-Store the results.
			//--------------------
			if (sumwab >= mdbcthreshold) {
				const float3 dpos = make_float3(-bnormalp1.x, -bnormalp1.y, -bnormalp1.z); //-Boundary particle position - ghost node position.
				if (sim2d) {
					const double determ = cumath::Determinant3x3dbl(a_corr2);
					if (fabs(determ) >= determlimit) {//-Use 1e-3f (first_order) or 1e+3f (zeroth_order).
						const tmatrix3f invacorr2 = cumath::InverseMatrix3x3dbl(a_corr2, determ);
						//-GHOST NODE DENSITY IS MIRRORED BACK TO THE BOUNDARY PARTICLES.
						const float rhoghost = float(invacorr2.a11*rhopp1 + invacorr2.a12*gradrhopp1.x + invacorr2.a13*gradrhopp1.z);
						const float grx = -float(invacorr2.a21*rhopp1 + invacorr2.a22*gradrhopp1.x + invacorr2.a23*gradrhopp1.z);
						const float grz = -float(invacorr2.a31*rhopp1 + invacorr2.a32*gradrhopp1.x + invacorr2.a33*gradrhopp1.z);
						rhopfinal = (rhoghost + grx*dpos.x + grz*dpos.z);
						//-Ghost stress ==== mdbr
						//-xx
						const float sigmaxxg = float(invacorr2.a11*sigmap1.xx + invacorr2.a12*gradsigmaxxp1.x + invacorr2.a13*gradsigmaxxp1.z);
						const float sixxgrx = -float(invacorr2.a21*sigmap1.xx + invacorr2.a22*gradsigmaxxp1.x + invacorr2.a23*gradsigmaxxp1.z);
						const float sixxgrz = -float(invacorr2.a31*sigmap1.xx + invacorr2.a32*gradsigmaxxp1.x + invacorr2.a33*gradsigmaxxp1.z);
						//-zz
						const float sigmazzg = float(invacorr2.a11*sigmap1.zz + invacorr2.a12*gradsigmazzp1.x + invacorr2.a13*gradsigmazzp1.z);
						const float sizzgrx = -float(invacorr2.a21*sigmap1.zz + invacorr2.a22*gradsigmazzp1.x + invacorr2.a23*gradsigmazzp1.z);
						const float sizzgrz = -float(invacorr2.a31*sigmap1.zz + invacorr2.a32*gradsigmazzp1.x + invacorr2.a33*gradsigmazzp1.z);
						//-xz
						const float sigmaxzg = float(invacorr2.a11*sigmap1.xz + invacorr2.a12*gradsigmaxzp1.x + invacorr2.a13*gradsigmaxzp1.z);
						const float sixzgrx = -float(invacorr2.a21*sigmap1.xz + invacorr2.a22*gradsigmaxzp1.x + invacorr2.a23*gradsigmaxzp1.z);
						const float sixzgrz = -float(invacorr2.a31*sigmap1.xz + invacorr2.a32*gradsigmaxzp1.x + invacorr2.a33*gradsigmaxzp1.z);
						//-Final stress
						sigmafinal.xx = sigmaxxg + sixxgrx*dpos.x + sixxgrz*dpos.z;
						sigmafinal.zz = sigmazzg + sizzgrx*dpos.x + sizzgrz*dpos.z;
						sigmafinal.xz = sigmaxzg + sixzgrx*dpos.x + sixzgrz*dpos.z;
						//=====
					}
					else if (a_corr2.a11>0) {//-Determinant is small but a11 is nonzero, 0th order ANGELO.
						rhopfinal = float(rhopp1 / a_corr2.a11);
						//====mdbr
						sigmafinal.xx = float(sigmap1.xx / a_corr2.a11);
						sigmafinal.zz = float(sigmap1.zz / a_corr2.a11);
						sigmafinal.xz = float(sigmap1.xz / a_corr2.a11);
					}
					//-Ghost node velocity (0th order).
					if (tslip != SLIP_Vel0) {
						velrhopfinal.x = float(velp1.x / a_corr2.a11);
						velrhopfinal.z = float(velp1.z / a_corr2.a11);
						velrhopfinal.y = 0;
					}
				}
				else {
					const double determ = cumath::Determinant4x4dbl(a_corr3);
					if (fabs(determ) >= determlimit) {
						const tmatrix4f invacorr3 = cumath::InverseMatrix4x4dbl(a_corr3, determ);
						//-GHOST NODE DENSITY IS MIRRORED BACK TO THE BOUNDARY PARTICLES.
						const float rhoghost = float(invacorr3.a11*rhopp1 + invacorr3.a12*gradrhopp1.x + invacorr3.a13*gradrhopp1.y + invacorr3.a14*gradrhopp1.z);
						const float grx = -float(invacorr3.a21*rhopp1 + invacorr3.a22*gradrhopp1.x + invacorr3.a23*gradrhopp1.y + invacorr3.a24*gradrhopp1.z);
						const float gry = -float(invacorr3.a31*rhopp1 + invacorr3.a32*gradrhopp1.x + invacorr3.a33*gradrhopp1.y + invacorr3.a34*gradrhopp1.z);
						const float grz = -float(invacorr3.a41*rhopp1 + invacorr3.a42*gradrhopp1.x + invacorr3.a43*gradrhopp1.y + invacorr3.a44*gradrhopp1.z);
						rhopfinal = (rhoghost + grx*dpos.x + gry*dpos.y + grz*dpos.z);
						//-Ghost stress ==== mdbr
						//-xx
						const float sigmaxxg = float(invacorr3.a11*sigmap1.xx + invacorr3.a12*gradsigmaxxp1.x + invacorr3.a13*gradsigmaxxp1.y + invacorr3.a14*gradsigmaxxp1.z);
						const float sixxgrx = -float(invacorr3.a21*sigmap1.xx + invacorr3.a22*gradsigmaxxp1.x + invacorr3.a23*gradsigmaxxp1.y + invacorr3.a24*gradsigmaxxp1.z);
						const float sixxgry = -float(invacorr3.a31*sigmap1.xx + invacorr3.a32*gradsigmaxxp1.x + invacorr3.a33*gradsigmaxxp1.y + invacorr3.a34*gradsigmaxxp1.z);
						const float sixxgrz = -float(invacorr3.a41*sigmap1.xx + invacorr3.a42*gradsigmaxxp1.x + invacorr3.a43*gradsigmaxxp1.y + invacorr3.a44*gradsigmaxxp1.z);
						//-yy
						const float sigmayyg = float(invacorr3.a11*sigmap1.yy + invacorr3.a12*gradsigmayyp1.x + invacorr3.a13*gradsigmayyp1.y + invacorr3.a14*gradsigmayyp1.z);
						const float siyygrx = -float(invacorr3.a21*sigmap1.yy + invacorr3.a22*gradsigmayyp1.x + invacorr3.a23*gradsigmayyp1.y + invacorr3.a24*gradsigmayyp1.z);
						const float siyygry = -float(invacorr3.a31*sigmap1.yy + invacorr3.a32*gradsigmayyp1.x + invacorr3.a33*gradsigmayyp1.y + invacorr3.a34*gradsigmayyp1.z);
						const float siyygrz = -float(invacorr3.a41*sigmap1.yy + invacorr3.a42*gradsigmayyp1.x + invacorr3.a43*gradsigmayyp1.y + invacorr3.a44*gradsigmayyp1.z);
						//-zz
						const float sigmazzg = float(invacorr3.a11*sigmap1.zz + invacorr3.a12*gradsigmazzp1.x + invacorr3.a13*gradsigmazzp1.y + invacorr3.a14*gradsigmazzp1.z);
						const float sizzgrx = -float(invacorr3.a21*sigmap1.zz + invacorr3.a22*gradsigmazzp1.x + invacorr3.a23*gradsigmazzp1.y + invacorr3.a24*gradsigmazzp1.z);
						const float sizzgry = -float(invacorr3.a31*sigmap1.zz + invacorr3.a32*gradsigmazzp1.x + invacorr3.a33*gradsigmazzp1.y + invacorr3.a34*gradsigmazzp1.z);
						const float sizzgrz = -float(invacorr3.a41*sigmap1.zz + invacorr3.a42*gradsigmazzp1.x + invacorr3.a43*gradsigmazzp1.y + invacorr3.a44*gradsigmazzp1.z);
						//-xy
						const float sigmaxyg = float(invacorr3.a11*sigmap1.xy + invacorr3.a12*gradsigmaxyp1.x + invacorr3.a13*gradsigmaxyp1.y + invacorr3.a14*gradsigmaxyp1.z);
						const float sixygrx = -float(invacorr3.a21*sigmap1.xy + invacorr3.a22*gradsigmaxyp1.x + invacorr3.a23*gradsigmaxyp1.y + invacorr3.a24*gradsigmaxyp1.z);
						const float sixygry = -float(invacorr3.a31*sigmap1.xy + invacorr3.a32*gradsigmaxyp1.x + invacorr3.a33*gradsigmaxyp1.y + invacorr3.a34*gradsigmaxyp1.z);
						const float sixygrz = -float(invacorr3.a41*sigmap1.xy + invacorr3.a42*gradsigmaxyp1.x + invacorr3.a43*gradsigmaxyp1.y + invacorr3.a44*gradsigmaxyp1.z);
						//-yz
						const float sigmayzg = float(invacorr3.a11*sigmap1.yz + invacorr3.a12*gradsigmayzp1.x + invacorr3.a13*gradsigmayzp1.y + invacorr3.a14*gradsigmayzp1.z);
						const float siyzgrx = -float(invacorr3.a21*sigmap1.yz + invacorr3.a22*gradsigmayzp1.x + invacorr3.a23*gradsigmayzp1.y + invacorr3.a24*gradsigmayzp1.z);
						const float siyzgry = -float(invacorr3.a31*sigmap1.yz + invacorr3.a32*gradsigmayzp1.x + invacorr3.a33*gradsigmayzp1.y + invacorr3.a34*gradsigmayzp1.z);
						const float siyzgrz = -float(invacorr3.a41*sigmap1.yz + invacorr3.a42*gradsigmayzp1.x + invacorr3.a43*gradsigmayzp1.y + invacorr3.a44*gradsigmayzp1.z);
						//-xz
						const float sigmaxzg = float(invacorr3.a11*sigmap1.xz + invacorr3.a12*gradsigmaxzp1.x + invacorr3.a13*gradsigmaxzp1.y + invacorr3.a14*gradsigmaxzp1.z);
						const float sixzgrx = -float(invacorr3.a21*sigmap1.xz + invacorr3.a22*gradsigmaxzp1.x + invacorr3.a23*gradsigmaxzp1.y + invacorr3.a24*gradsigmaxzp1.z);
						const float sixzgry = -float(invacorr3.a31*sigmap1.xz + invacorr3.a32*gradsigmaxzp1.x + invacorr3.a33*gradsigmaxzp1.y + invacorr3.a34*gradsigmaxzp1.z);
						const float sixzgrz = -float(invacorr3.a41*sigmap1.xz + invacorr3.a42*gradsigmaxzp1.x + invacorr3.a43*gradsigmaxzp1.y + invacorr3.a44*gradsigmaxzp1.z);
						//-Final stress
						sigmafinal.xx = sigmaxxg + sixxgrx*dpos.x + sixxgry*dpos.y + sixxgrz*dpos.z;
						sigmafinal.yy = sigmayyg + siyygrx*dpos.x + siyygry*dpos.y + siyygrz*dpos.z;
						sigmafinal.zz = sigmazzg + sizzgrx*dpos.x + sizzgry*dpos.y + sizzgrz*dpos.z;
						sigmafinal.xy = sigmaxyg + sixygrx*dpos.x + sixygry*dpos.y + sixygrz*dpos.z;
						sigmafinal.yz = sigmayzg + siyzgrx*dpos.x + siyzgry*dpos.y + siyzgrz*dpos.z;
						sigmafinal.xz = sigmaxzg + sixzgrx*dpos.x + sixzgry*dpos.y + sixzgrz*dpos.z;
					}
					else if (a_corr3.a11>0) {//-Determinant is small but a11 is nonzero, 0th order ANGELO.
						rhopfinal = float(rhopp1 / a_corr3.a11);
						//==== mdbr
						sigmafinal.xx = float(sigmap1.xx / a_corr3.a11);
						sigmafinal.yy = float(sigmap1.yy / a_corr3.a11);
						sigmafinal.zz = float(sigmap1.zz / a_corr3.a11);
						sigmafinal.xy = float(sigmap1.xy / a_corr3.a11);
						sigmafinal.yz = float(sigmap1.yz / a_corr3.a11);
						sigmafinal.xz = float(sigmap1.xz / a_corr3.a11);
					}
					//-Ghost node velocity (0th order).
					if (tslip != SLIP_Vel0) {
						velrhopfinal.x = float(velp1.x / a_corr3.a11);
						velrhopfinal.y = float(velp1.y / a_corr3.a11);
						velrhopfinal.z = float(velp1.z / a_corr3.a11);
					}
				}
				//-Store the results.
				rhopfinal = (rhopfinal != FLT_MAX ? rhopfinal : CTE.rhopzero);
				if (tslip == SLIP_Vel0) {//-DBC vel=0
					velrhop[p1].w = rhopfinal;
					sigma[p1] = sigmafinal;//mdbr
				}
				if (tslip == SLIP_NoSlip) {//-No-Slip
					const float3 v = motionvel[p1];
					velrhop[p1] = make_float4(v.x + v.x - velrhopfinal.x, v.y + v.y - velrhopfinal.y, v.z + v.z - velrhopfinal.z, rhopfinal);
					sigma[p1] = sigmafinal;//mdbr
				}
				if (tslip == SLIP_FreeSlip) {//-No-Penetration and free slip    SHABA
					float3 FSVelFinal; // final free slip boundary velocity
					const float3 v = motionvel[p1];
					float motion = sqrt(v.x*v.x + v.y*v.y + v.z*v.z); // to check if boundary moving
					float norm = sqrt(bnormalp1.x*bnormalp1.x + bnormalp1.y*bnormalp1.y + bnormalp1.z*bnormalp1.z);
					float3 normal; // creating a normailsed boundary normal
					normal.x = fabs(bnormalp1.x) / norm; normal.y = fabs(bnormalp1.y) / norm; normal.z = fabs(bnormalp1.z) / norm;

					// finding the velocity componants normal and tangential to boundary 
					float3 normvel = make_float3(velrhopfinal.x*normal.x, velrhopfinal.y*normal.y, velrhopfinal.z*normal.z); // velocity in direction of normal pointin ginto fluid)
					float3 tangvel = make_float3(velrhopfinal.x - normvel.x, velrhopfinal.y - normvel.y, velrhopfinal.z - normvel.z); // velocity tangential to normal

					if (motion > 0) { // if moving boundary
						float3 normmot = make_float3(v.x*normal.x, v.y*normal.y, v.z*normal.z); // boundary motion in direction normal to boundary 
						FSVelFinal = make_float3(normmot.x + normmot.x - normvel.x, normmot.y + normmot.y - normvel.y, normmot.z + normmot.z - normvel.z);
						// only velocity in normal direction for no-penetration
						// fluid sees zero velocity in the tangetial direction
					}
					else {
						FSVelFinal = make_float3(tangvel.x - normvel.x, tangvel.y - normvel.y, tangvel.z - normvel.z);
						// tangential velocity equal to fluid velocity for free slip
						// normal velocity reversed for no-penetration
					}

					// Save the velocity and density
					velrhop[p1] = make_float4(FSVelFinal.x, FSVelFinal.y, FSVelFinal.z, rhopfinal);
					sigma[p1] = sigmafinal;//mdbr
				}
			}
		}
	}
}
//------------------------------------------------------------------------------
/// Perform interaction between ghost node of selected bondary and domain particle for water phase.
//------------------------------------------------------------------------------
template<TpKernel tker, bool sim2d, TpSlipMode tslip> __global__ void KerInteractionMdbcCorrectionNNFluid_Dbl
(unsigned n, unsigned nbound, float determlimit, float mdbcthreshold
	, int scelldiv, int4 nc, int3 cellzero, const int2 *beginendcellfluid
	, const double2 *posxy, const double *posz, const typecode *code, const unsigned *idp
	, const float3 *boundnormal, const float3 *motionvel, float4 *velrhop)
{
	const unsigned p1 = blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
	if (p1<n) {
		const float3 bnormalp1 = boundnormal[p1];
		if (bnormalp1.x != 0 || bnormalp1.y != 0 || bnormalp1.z != 0) {
			float rhopfinal = FLT_MAX;
			float3 velrhopfinal = make_float3(0, 0, 0);
			float sumwab = 0;

			//-Calculates ghost node position.
			double3 gposp1 = make_double3(posxy[p1].x + bnormalp1.x, posxy[p1].y + bnormalp1.y, posz[p1] + bnormalp1.z);
			gposp1 = (CTE.periactive != 0 ? cusph::KerUpdatePeriodicPos(gposp1) : gposp1); //-Corrected interface Position.
																						   //-Initializes variables for calculation.
			float rhopp1 = 0;
			float3 gradrhopp1 = make_float3(0, 0, 0);
			float3 velp1 = make_float3(0, 0, 0);                              // -Only for velocity
			tmatrix3d a_corr2; if (sim2d) cumath::Tmatrix3dReset(a_corr2); //-Only for 2D.
			tmatrix4d a_corr3; if (!sim2d)cumath::Tmatrix4dReset(a_corr3); //-Only for 3D.

																		   //-Obtains neighborhood search limits.
			int ini1, fin1, ini2, fin2, ini3, fin3;
			cunsearch::InitCte(gposp1.x, gposp1.y, gposp1.z, scelldiv, nc, cellzero, ini1, fin1, ini2, fin2, ini3, fin3);

			//-Boundary-Fluid interaction.
			for (int c3 = ini3; c3<fin3; c3 += nc.w)for (int c2 = ini2; c2<fin2; c2 += nc.x) {
				unsigned pini, pfin = 0;  cunsearch::ParticleRange(c2, c3, ini1, fin1, beginendcellfluid, pini, pfin);
				if (pfin)for (unsigned p2 = pini; p2<pfin; p2++) {
					const double2 p2xy = posxy[p2];
					const float drx = float(gposp1.x - p2xy.x);
					const float dry = float(gposp1.y - p2xy.y);
					const float drz = float(gposp1.z - posz[p2]);
					const float rr2 = drx*drx + dry*dry + drz*drz;
					const typecode pp2 = CODE_GetTypeValue(code[p2]); //<vs_non-Newtonian>
					if (rr2 <= CTE.kernelsize2 && rr2 >= ALMOSTZERO && CODE_IsFluid(code[p2])&&pp2==0) {//-Only with domain particles in water phase (including inout).
						//-Computes kernel.
						float fac;
						const float wab = cufsph::GetKernel_WabFac<tker>(rr2, fac);
						const float frx = fac*drx, fry = fac*dry, frz = fac*drz; //-Gradients.

						//===== Get mass and volume of particle p2 =====
						const float4 velrhopp2 = velrhop[p2];
						float massp2 = PHASEARRAY[pp2].mass; //-Contiene masa de particula segun sea bound o fluid.

						//float massp2=CTE.massf;
						const float volp2 = massp2 / velrhopp2.w;

						//===== Density and its gradient =====
						rhopp1 += massp2*wab;
						gradrhopp1.x += massp2*frx;
						gradrhopp1.y += massp2*fry;
						gradrhopp1.z += massp2*frz;

						//===== Kernel values multiplied by volume =====
						const float vwab = wab*volp2;
						sumwab += vwab;
						const float vfrx = frx*volp2;
						const float vfry = fry*volp2;
						const float vfrz = frz*volp2;

						//===== Velocity =====
						if (tslip != SLIP_Vel0) {
							velp1.x += vwab*velrhopp2.x;
							velp1.y += vwab*velrhopp2.y;
							velp1.z += vwab*velrhopp2.z;
						}

						//===== Matrix A for correction =====
						if (sim2d) {
							a_corr2.a11 += vwab;  a_corr2.a12 += drx*vwab;  a_corr2.a13 += drz*vwab;
							a_corr2.a21 += vfrx;  a_corr2.a22 += drx*vfrx;  a_corr2.a23 += drz*vfrx;
							a_corr2.a31 += vfrz;  a_corr2.a32 += drx*vfrz;  a_corr2.a33 += drz*vfrz;
						}
						else {
							a_corr3.a11 += vwab;  a_corr3.a12 += drx*vwab;  a_corr3.a13 += dry*vwab;  a_corr3.a14 += drz*vwab;
							a_corr3.a21 += vfrx;  a_corr3.a22 += drx*vfrx;  a_corr3.a23 += dry*vfrx;  a_corr3.a24 += drz*vfrx;
							a_corr3.a31 += vfry;  a_corr3.a32 += drx*vfry;  a_corr3.a33 += dry*vfry;  a_corr3.a34 += drz*vfry;
							a_corr3.a41 += vfrz;  a_corr3.a42 += drx*vfrz;  a_corr3.a43 += dry*vfrz;  a_corr3.a44 += drz*vfrz;
						}
					}
				}
			}

			//-Store the results.
			//--------------------
			if (sumwab >= mdbcthreshold) {
				const float3 dpos = make_float3(-bnormalp1.x, -bnormalp1.y, -bnormalp1.z); //-Boundary particle position - ghost node position.
				if (sim2d) {
					const double determ = cumath::Determinant3x3(a_corr2);
					if (fabs(determ) >= determlimit) {//-Use 1e-3f (first_order) or 1e+3f (zeroth_order).
						const tmatrix3d invacorr2 = cumath::InverseMatrix3x3(a_corr2, determ);
						//-GHOST NODE DENSITY IS MIRRORED BACK TO THE BOUNDARY PARTICLES.
						const float rhoghost = float(invacorr2.a11*rhopp1 + invacorr2.a12*gradrhopp1.x + invacorr2.a13*gradrhopp1.z);
						const float grx = -float(invacorr2.a21*rhopp1 + invacorr2.a22*gradrhopp1.x + invacorr2.a23*gradrhopp1.z);
						const float grz = -float(invacorr2.a31*rhopp1 + invacorr2.a32*gradrhopp1.x + invacorr2.a33*gradrhopp1.z);
						rhopfinal = (rhoghost + grx*dpos.x + grz*dpos.z);
					}
					else if (a_corr2.a11>0) {//-Determinant is small but a11 is nonzero, 0th order ANGELO.
						rhopfinal = float(rhopp1 / a_corr2.a11);
					}
					//-Ghost node velocity (0th order).
					if (tslip != SLIP_Vel0) {
						velrhopfinal.x = float(velp1.x / a_corr2.a11);
						velrhopfinal.z = float(velp1.z / a_corr2.a11);
						velrhopfinal.y = 0;
					}
				}
				else {
					const double determ = cumath::Determinant4x4(a_corr3);
					if (fabs(determ) >= determlimit) {
						const tmatrix4d invacorr3 = cumath::InverseMatrix4x4(a_corr3, determ);
						//-GHOST NODE DENSITY IS MIRRORED BACK TO THE BOUNDARY PARTICLES.
						const float rhoghost = float(invacorr3.a11*rhopp1 + invacorr3.a12*gradrhopp1.x + invacorr3.a13*gradrhopp1.y + invacorr3.a14*gradrhopp1.z);
						const float grx = -float(invacorr3.a21*rhopp1 + invacorr3.a22*gradrhopp1.x + invacorr3.a23*gradrhopp1.y + invacorr3.a24*gradrhopp1.z);
						const float gry = -float(invacorr3.a31*rhopp1 + invacorr3.a32*gradrhopp1.x + invacorr3.a33*gradrhopp1.y + invacorr3.a34*gradrhopp1.z);
						const float grz = -float(invacorr3.a41*rhopp1 + invacorr3.a42*gradrhopp1.x + invacorr3.a43*gradrhopp1.y + invacorr3.a44*gradrhopp1.z);
						rhopfinal = (rhoghost + grx*dpos.x + gry*dpos.y + grz*dpos.z);
					}
					else if (a_corr3.a11>0) {//-Determinant is small but a11 is nonzero, 0th order ANGELO.
						rhopfinal = float(rhopp1 / a_corr3.a11);
					}
					//-Ghost node velocity (0th order).
					if (tslip != SLIP_Vel0) {
						velrhopfinal.x = float(velp1.x / a_corr3.a11);
						velrhopfinal.y = float(velp1.y / a_corr3.a11);
						velrhopfinal.z = float(velp1.z / a_corr3.a11);
					}
				}
				//-Store the results.
				rhopfinal = (rhopfinal != FLT_MAX ? rhopfinal : CTE.rhopzero);
				if (tslip == SLIP_Vel0) {//-DBC vel=0
					velrhop[p1].w = rhopfinal;
				}
				if (tslip == SLIP_NoSlip) {//-No-Slip
					const float3 v = motionvel[p1];
					velrhop[p1] = make_float4(v.x + v.x - velrhopfinal.x, v.y + v.y - velrhopfinal.y, v.z + v.z - velrhopfinal.z, rhopfinal);
				}
				if (tslip == SLIP_FreeSlip) {//-No-Penetration and free slip    SHABA
					float3 FSVelFinal; // final free slip boundary velocity
					const float3 v = motionvel[p1];
					float motion = sqrt(v.x*v.x + v.y*v.y + v.z*v.z); // to check if boundary moving
					float norm = sqrt(bnormalp1.x*bnormalp1.x + bnormalp1.y*bnormalp1.y + bnormalp1.z*bnormalp1.z);
					float3 normal; // creating a normailsed boundary normal
					normal.x = fabs(bnormalp1.x) / norm; normal.y = fabs(bnormalp1.y) / norm; normal.z = fabs(bnormalp1.z) / norm;

					// finding the velocity componants normal and tangential to boundary 
					float3 normvel = make_float3(velrhopfinal.x*normal.x, velrhopfinal.y*normal.y, velrhopfinal.z*normal.z); // velocity in direction of normal pointin ginto fluid)
					float3 tangvel = make_float3(velrhopfinal.x - normvel.x, velrhopfinal.y - normvel.y, velrhopfinal.z - normvel.z); // velocity tangential to normal

					if (motion > 0) { // if moving boundary
						float3 normmot = make_float3(v.x*normal.x, v.y*normal.y, v.z*normal.z); // boundary motion in direction normal to boundary 
						FSVelFinal = make_float3(normmot.x + normmot.x - normvel.x, normmot.y + normmot.y - normvel.y, normmot.z + normmot.z - normvel.z);
						// only velocity in normal direction for no-penetration
						// fluid sees zero velocity in the tangetial direction
					}
					else {
						FSVelFinal = make_float3(tangvel.x - normvel.x, tangvel.y - normvel.y, tangvel.z - normvel.z);
						// tangential velocity equal to fluid velocity for free slip
						// normal velocity reversed for no-penetration
					}

					// Save the velocity and density
					velrhop[p1] = make_float4(FSVelFinal.x, FSVelFinal.y, FSVelFinal.z, rhopfinal);
				}
			}
		}
	}
}

//------------------------------------------------------------------------------
/// Perform interaction between ghost node of selected bondary and domain particle for soil phase.
//------------------------------------------------------------------------------
template<TpKernel tker, bool sim2d, TpSlipMode tslip> __global__ void KerInteractionMdbcCorrectionNNGranular_Dbl
(unsigned n, unsigned nbound, float determlimit, float mdbcthreshold
	, int scelldiv, int4 nc, int3 cellzero, const int2 *beginendcellfluid
	, const double2 *posxy, const double *posz, const typecode *code, const unsigned *idp
	, const float3 *boundnormal, const float3 *motionvel, float4 *velrhop,tsymatrix3f *sigma)
{
	const unsigned p1 = blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
	if (p1<n) {
		const float3 bnormalp1 = boundnormal[p1];
		if (bnormalp1.x != 0 || bnormalp1.y != 0 || bnormalp1.z != 0) {
			float rhopfinal = FLT_MAX;
			float3 velrhopfinal = make_float3(0, 0, 0);
			tsymatrix3f sigmafinal = { 0,0,0,0,0,0 };//mdbr
			float sumwab = 0;

			//-Calculates ghost node position.
			double3 gposp1 = make_double3(posxy[p1].x + bnormalp1.x, posxy[p1].y + bnormalp1.y, posz[p1] + bnormalp1.z);
			gposp1 = (CTE.periactive != 0 ? cusph::KerUpdatePeriodicPos(gposp1) : gposp1); //-Corrected interface Position.
																						   //-Initializes variables for calculation.
			float rhopp1 = 0;
			float3 gradrhopp1 = make_float3(0, 0, 0);
			//===== mdbr
			//-Solid stress
			tsymatrix3f sigmap1 = { 0,0,0,0,0,0 };//-First-order Stress value
			float3 gradsigmaxxp1 = make_float3(0, 0, 0);//-Stress gradient
			float3 gradsigmayyp1 = make_float3(0, 0, 0);
			float3 gradsigmazzp1 = make_float3(0, 0, 0);
			float3 gradsigmaxyp1 = make_float3(0, 0, 0);
			float3 gradsigmayzp1 = make_float3(0, 0, 0);
			float3 gradsigmaxzp1 = make_float3(0, 0, 0);
			//=====
			float3 velp1 = make_float3(0, 0, 0);                              // -Only for velocity
			tmatrix3d a_corr2; if (sim2d) cumath::Tmatrix3dReset(a_corr2); //-Only for 2D.
			tmatrix4d a_corr3; if (!sim2d)cumath::Tmatrix4dReset(a_corr3); //-Only for 3D.

																		   //-Obtains neighborhood search limits.
			int ini1, fin1, ini2, fin2, ini3, fin3;
			cunsearch::InitCte(gposp1.x, gposp1.y, gposp1.z, scelldiv, nc, cellzero, ini1, fin1, ini2, fin2, ini3, fin3);

			//-Boundary-Fluid interaction.
			for (int c3 = ini3; c3<fin3; c3 += nc.w)for (int c2 = ini2; c2<fin2; c2 += nc.x) {
				unsigned pini, pfin = 0;  cunsearch::ParticleRange(c2, c3, ini1, fin1, beginendcellfluid, pini, pfin);
				if (pfin)for (unsigned p2 = pini; p2<pfin; p2++) {
					const double2 p2xy = posxy[p2];
					const float drx = float(gposp1.x - p2xy.x);
					const float dry = float(gposp1.y - p2xy.y);
					const float drz = float(gposp1.z - posz[p2]);
					const float rr2 = drx*drx + dry*dry + drz*drz;
					const typecode pp2 = CODE_GetTypeValue(code[p2]); //<vs_non-Newtonian>
					if (rr2 <= CTE.kernelsize2 && rr2 >= ALMOSTZERO && CODE_IsFluid(code[p2]) && pp2 == 1) {//-Only with domain particles in soil phase (including inout).
						//-Computes kernel.
						float fac;
						const float wab = cufsph::GetKernel_WabFac<tker>(rr2, fac);
						const float frx = fac*drx, fry = fac*dry, frz = fac*drz; //-Gradients.

						//===== Get mass and volume of particle p2 =====
						const float4 velrhopp2 = velrhop[p2];
						float massp2 = PHASEARRAY[pp2].mass; //-Contiene masa de particula segun sea bound o fluid.
						const tsymatrix3f sigmap2 = sigma[p2];//mdbr
						//float massp2=CTE.massf;
						const float volp2 = massp2 / velrhopp2.w;

						//===== Density and its gradient =====
						rhopp1 += massp2*wab;
						gradrhopp1.x += massp2*frx;
						gradrhopp1.y += massp2*fry;
						gradrhopp1.z += massp2*frz;

						//===== Kernel values multiplied by volume =====
						const float vwab = wab*volp2;
						sumwab += vwab;
						const float vfrx = frx*volp2;
						const float vfry = fry*volp2;
						const float vfrz = frz*volp2;
						//===== mdbr
						//===== Stress value =====
						sigmap1.xx += vwab*sigmap2.xx;
						sigmap1.yy += vwab*sigmap2.yy;
						sigmap1.zz += vwab*sigmap2.zz;
						sigmap1.xy += vwab*sigmap2.xy;
						sigmap1.yz += vwab*sigmap2.yz;
						sigmap1.xz += vwab*sigmap2.xz;

						//===== Stress gradient =====
						//===== xx
						gradsigmaxxp1.x += vfrx*sigmap2.xx;
						gradsigmaxxp1.y += vfry*sigmap2.xx;
						gradsigmaxxp1.z += vfrz*sigmap2.xx;
						//===== yy
						gradsigmayyp1.x += vfrx*sigmap2.yy;
						gradsigmayyp1.y += vfry*sigmap2.yy;
						gradsigmayyp1.z += vfrz*sigmap2.yy;
						//===== zz
						gradsigmazzp1.x += vfrx*sigmap2.zz;
						gradsigmazzp1.y += vfry*sigmap2.zz;
						gradsigmazzp1.z += vfrz*sigmap2.zz;
						//===== xy
						gradsigmaxyp1.x += vfrx*sigmap2.xy;
						gradsigmaxyp1.y += vfry*sigmap2.xy;
						gradsigmaxyp1.z += vfrz*sigmap2.xy;
						//===== yz
						gradsigmayzp1.x += vfrx*sigmap2.yz;
						gradsigmayzp1.y += vfry*sigmap2.yz;
						gradsigmayzp1.z += vfrz*sigmap2.yz;
						//===== xz
						gradsigmaxzp1.x += vfrx*sigmap2.xz;
						gradsigmaxzp1.y += vfry*sigmap2.xz;
						gradsigmaxzp1.z += vfrz*sigmap2.xz;
						//===== End
						//===== Velocity =====
						if (tslip != SLIP_Vel0) {
							velp1.x += vwab*velrhopp2.x;
							velp1.y += vwab*velrhopp2.y;
							velp1.z += vwab*velrhopp2.z;
						}

						//===== Matrix A for correction =====
						if (sim2d) {
							a_corr2.a11 += vwab;  a_corr2.a12 += drx*vwab;  a_corr2.a13 += drz*vwab;
							a_corr2.a21 += vfrx;  a_corr2.a22 += drx*vfrx;  a_corr2.a23 += drz*vfrx;
							a_corr2.a31 += vfrz;  a_corr2.a32 += drx*vfrz;  a_corr2.a33 += drz*vfrz;
						}
						else {
							a_corr3.a11 += vwab;  a_corr3.a12 += drx*vwab;  a_corr3.a13 += dry*vwab;  a_corr3.a14 += drz*vwab;
							a_corr3.a21 += vfrx;  a_corr3.a22 += drx*vfrx;  a_corr3.a23 += dry*vfrx;  a_corr3.a24 += drz*vfrx;
							a_corr3.a31 += vfry;  a_corr3.a32 += drx*vfry;  a_corr3.a33 += dry*vfry;  a_corr3.a34 += drz*vfry;
							a_corr3.a41 += vfrz;  a_corr3.a42 += drx*vfrz;  a_corr3.a43 += dry*vfrz;  a_corr3.a44 += drz*vfrz;
						}
					}
				}
			}

			//-Store the results.
			//--------------------
			if (sumwab >= mdbcthreshold) {
				const float3 dpos = make_float3(-bnormalp1.x, -bnormalp1.y, -bnormalp1.z); //-Boundary particle position - ghost node position.
				if (sim2d) {
					const double determ = cumath::Determinant3x3(a_corr2);
					if (fabs(determ) >= determlimit) {//-Use 1e-3f (first_order) or 1e+3f (zeroth_order).
						const tmatrix3d invacorr2 = cumath::InverseMatrix3x3(a_corr2, determ);
						//-GHOST NODE DENSITY IS MIRRORED BACK TO THE BOUNDARY PARTICLES.
						const float rhoghost = float(invacorr2.a11*rhopp1 + invacorr2.a12*gradrhopp1.x + invacorr2.a13*gradrhopp1.z);
						const float grx = -float(invacorr2.a21*rhopp1 + invacorr2.a22*gradrhopp1.x + invacorr2.a23*gradrhopp1.z);
						const float grz = -float(invacorr2.a31*rhopp1 + invacorr2.a32*gradrhopp1.x + invacorr2.a33*gradrhopp1.z);
						rhopfinal = (rhoghost + grx*dpos.x + grz*dpos.z);
						//-Ghost stress ==== mdbr
						//-xx
						const float sigmaxxg = float(invacorr2.a11*sigmap1.xx + invacorr2.a12*gradsigmaxxp1.x + invacorr2.a13*gradsigmaxxp1.z);
						const float sixxgrx = -float(invacorr2.a21*sigmap1.xx + invacorr2.a22*gradsigmaxxp1.x + invacorr2.a23*gradsigmaxxp1.z);
						const float sixxgrz = -float(invacorr2.a31*sigmap1.xx + invacorr2.a32*gradsigmaxxp1.x + invacorr2.a33*gradsigmaxxp1.z);
						//-zz
						const float sigmazzg = float(invacorr2.a11*sigmap1.zz + invacorr2.a12*gradsigmazzp1.x + invacorr2.a13*gradsigmazzp1.z);
						const float sizzgrx = -float(invacorr2.a21*sigmap1.zz + invacorr2.a22*gradsigmazzp1.x + invacorr2.a23*gradsigmazzp1.z);
						const float sizzgrz = -float(invacorr2.a31*sigmap1.zz + invacorr2.a32*gradsigmazzp1.x + invacorr2.a33*gradsigmazzp1.z);
						//-xz
						const float sigmaxzg = float(invacorr2.a11*sigmap1.xz + invacorr2.a12*gradsigmaxzp1.x + invacorr2.a13*gradsigmaxzp1.z);
						const float sixzgrx = -float(invacorr2.a21*sigmap1.xz + invacorr2.a22*gradsigmaxzp1.x + invacorr2.a23*gradsigmaxzp1.z);
						const float sixzgrz = -float(invacorr2.a31*sigmap1.xz + invacorr2.a32*gradsigmaxzp1.x + invacorr2.a33*gradsigmaxzp1.z);
						//-Final stress
						sigmafinal.xx = sigmaxxg + sixxgrx*dpos.x + sixxgrz*dpos.z;
						sigmafinal.zz = sigmazzg + sizzgrx*dpos.x + sizzgrz*dpos.z;
						sigmafinal.xz = sigmaxzg + sixzgrx*dpos.x + sixzgrz*dpos.z;
						//=====
					}
					else if (a_corr2.a11>0) {//-Determinant is small but a11 is nonzero, 0th order ANGELO.
						rhopfinal = float(rhopp1 / a_corr2.a11);
						//====mdbr
						sigmafinal.xx = float(sigmap1.xx / a_corr2.a11);
						sigmafinal.zz = float(sigmap1.zz / a_corr2.a11);
						sigmafinal.xz = float(sigmap1.xz / a_corr2.a11);
					}
					//-Ghost node velocity (0th order).
					if (tslip != SLIP_Vel0) {
						velrhopfinal.x = float(velp1.x / a_corr2.a11);
						velrhopfinal.z = float(velp1.z / a_corr2.a11);
						velrhopfinal.y = 0;
					}
				}
				else {
					const double determ = cumath::Determinant4x4(a_corr3);
					if (fabs(determ) >= determlimit) {
						const tmatrix4d invacorr3 = cumath::InverseMatrix4x4(a_corr3, determ);
						//-GHOST NODE DENSITY IS MIRRORED BACK TO THE BOUNDARY PARTICLES.
						const float rhoghost = float(invacorr3.a11*rhopp1 + invacorr3.a12*gradrhopp1.x + invacorr3.a13*gradrhopp1.y + invacorr3.a14*gradrhopp1.z);
						const float grx = -float(invacorr3.a21*rhopp1 + invacorr3.a22*gradrhopp1.x + invacorr3.a23*gradrhopp1.y + invacorr3.a24*gradrhopp1.z);
						const float gry = -float(invacorr3.a31*rhopp1 + invacorr3.a32*gradrhopp1.x + invacorr3.a33*gradrhopp1.y + invacorr3.a34*gradrhopp1.z);
						const float grz = -float(invacorr3.a41*rhopp1 + invacorr3.a42*gradrhopp1.x + invacorr3.a43*gradrhopp1.y + invacorr3.a44*gradrhopp1.z);
						rhopfinal = (rhoghost + grx*dpos.x + gry*dpos.y + grz*dpos.z);
						//-Ghost stress ==== mdbr
						//-xx
						const float sigmaxxg = float(invacorr3.a11*sigmap1.xx + invacorr3.a12*gradsigmaxxp1.x + invacorr3.a13*gradsigmaxxp1.y + invacorr3.a14*gradsigmaxxp1.z);
						const float sixxgrx = -float(invacorr3.a21*sigmap1.xx + invacorr3.a22*gradsigmaxxp1.x + invacorr3.a23*gradsigmaxxp1.y + invacorr3.a24*gradsigmaxxp1.z);
						const float sixxgry = -float(invacorr3.a31*sigmap1.xx + invacorr3.a32*gradsigmaxxp1.x + invacorr3.a33*gradsigmaxxp1.y + invacorr3.a34*gradsigmaxxp1.z);
						const float sixxgrz = -float(invacorr3.a41*sigmap1.xx + invacorr3.a42*gradsigmaxxp1.x + invacorr3.a43*gradsigmaxxp1.y + invacorr3.a44*gradsigmaxxp1.z);
						//-yy
						const float sigmayyg = float(invacorr3.a11*sigmap1.yy + invacorr3.a12*gradsigmayyp1.x + invacorr3.a13*gradsigmayyp1.y + invacorr3.a14*gradsigmayyp1.z);
						const float siyygrx = -float(invacorr3.a21*sigmap1.yy + invacorr3.a22*gradsigmayyp1.x + invacorr3.a23*gradsigmayyp1.y + invacorr3.a24*gradsigmayyp1.z);
						const float siyygry = -float(invacorr3.a31*sigmap1.yy + invacorr3.a32*gradsigmayyp1.x + invacorr3.a33*gradsigmayyp1.y + invacorr3.a34*gradsigmayyp1.z);
						const float siyygrz = -float(invacorr3.a41*sigmap1.yy + invacorr3.a42*gradsigmayyp1.x + invacorr3.a43*gradsigmayyp1.y + invacorr3.a44*gradsigmayyp1.z);
						//-zz
						const float sigmazzg = float(invacorr3.a11*sigmap1.zz + invacorr3.a12*gradsigmazzp1.x + invacorr3.a13*gradsigmazzp1.y + invacorr3.a14*gradsigmazzp1.z);
						const float sizzgrx = -float(invacorr3.a21*sigmap1.zz + invacorr3.a22*gradsigmazzp1.x + invacorr3.a23*gradsigmazzp1.y + invacorr3.a24*gradsigmazzp1.z);
						const float sizzgry = -float(invacorr3.a31*sigmap1.zz + invacorr3.a32*gradsigmazzp1.x + invacorr3.a33*gradsigmazzp1.y + invacorr3.a34*gradsigmazzp1.z);
						const float sizzgrz = -float(invacorr3.a41*sigmap1.zz + invacorr3.a42*gradsigmazzp1.x + invacorr3.a43*gradsigmazzp1.y + invacorr3.a44*gradsigmazzp1.z);
						//-xy
						const float sigmaxyg = float(invacorr3.a11*sigmap1.xy + invacorr3.a12*gradsigmaxyp1.x + invacorr3.a13*gradsigmaxyp1.y + invacorr3.a14*gradsigmaxyp1.z);
						const float sixygrx = -float(invacorr3.a21*sigmap1.xy + invacorr3.a22*gradsigmaxyp1.x + invacorr3.a23*gradsigmaxyp1.y + invacorr3.a24*gradsigmaxyp1.z);
						const float sixygry = -float(invacorr3.a31*sigmap1.xy + invacorr3.a32*gradsigmaxyp1.x + invacorr3.a33*gradsigmaxyp1.y + invacorr3.a34*gradsigmaxyp1.z);
						const float sixygrz = -float(invacorr3.a41*sigmap1.xy + invacorr3.a42*gradsigmaxyp1.x + invacorr3.a43*gradsigmaxyp1.y + invacorr3.a44*gradsigmaxyp1.z);
						//-yz
						const float sigmayzg = float(invacorr3.a11*sigmap1.yz + invacorr3.a12*gradsigmayzp1.x + invacorr3.a13*gradsigmayzp1.y + invacorr3.a14*gradsigmayzp1.z);
						const float siyzgrx = -float(invacorr3.a21*sigmap1.yz + invacorr3.a22*gradsigmayzp1.x + invacorr3.a23*gradsigmayzp1.y + invacorr3.a24*gradsigmayzp1.z);
						const float siyzgry = -float(invacorr3.a31*sigmap1.yz + invacorr3.a32*gradsigmayzp1.x + invacorr3.a33*gradsigmayzp1.y + invacorr3.a34*gradsigmayzp1.z);
						const float siyzgrz = -float(invacorr3.a41*sigmap1.yz + invacorr3.a42*gradsigmayzp1.x + invacorr3.a43*gradsigmayzp1.y + invacorr3.a44*gradsigmayzp1.z);
						//-xz
						const float sigmaxzg = float(invacorr3.a11*sigmap1.xz + invacorr3.a12*gradsigmaxzp1.x + invacorr3.a13*gradsigmaxzp1.y + invacorr3.a14*gradsigmaxzp1.z);
						const float sixzgrx = -float(invacorr3.a21*sigmap1.xz + invacorr3.a22*gradsigmaxzp1.x + invacorr3.a23*gradsigmaxzp1.y + invacorr3.a24*gradsigmaxzp1.z);
						const float sixzgry = -float(invacorr3.a31*sigmap1.xz + invacorr3.a32*gradsigmaxzp1.x + invacorr3.a33*gradsigmaxzp1.y + invacorr3.a34*gradsigmaxzp1.z);
						const float sixzgrz = -float(invacorr3.a41*sigmap1.xz + invacorr3.a42*gradsigmaxzp1.x + invacorr3.a43*gradsigmaxzp1.y + invacorr3.a44*gradsigmaxzp1.z);
						//-Final stress
						sigmafinal.xx = sigmaxxg + sixxgrx*dpos.x + sixxgry*dpos.y + sixxgrz*dpos.z;
						sigmafinal.yy = sigmayyg + siyygrx*dpos.x + siyygry*dpos.y + siyygrz*dpos.z;
						sigmafinal.zz = sigmazzg + sizzgrx*dpos.x + sizzgry*dpos.y + sizzgrz*dpos.z;
						sigmafinal.xy = sigmaxyg + sixygrx*dpos.x + sixygry*dpos.y + sixygrz*dpos.z;
						sigmafinal.yz = sigmayzg + siyzgrx*dpos.x + siyzgry*dpos.y + siyzgrz*dpos.z;
						sigmafinal.xz = sigmaxzg + sixzgrx*dpos.x + sixzgry*dpos.y + sixzgrz*dpos.z;
					}
					else if (a_corr3.a11>0) {//-Determinant is small but a11 is nonzero, 0th order ANGELO.
						rhopfinal = float(rhopp1 / a_corr3.a11);
						//==== mdbr
						sigmafinal.xx = float(sigmap1.xx / a_corr3.a11);
						sigmafinal.yy = float(sigmap1.yy / a_corr3.a11);
						sigmafinal.zz = float(sigmap1.zz / a_corr3.a11);
						sigmafinal.xy = float(sigmap1.xy / a_corr3.a11);
						sigmafinal.yz = float(sigmap1.yz / a_corr3.a11);
						sigmafinal.xz = float(sigmap1.xz / a_corr3.a11);
					}
					//-Ghost node velocity (0th order).
					if (tslip != SLIP_Vel0) {
						velrhopfinal.x = float(velp1.x / a_corr3.a11);
						velrhopfinal.y = float(velp1.y / a_corr3.a11);
						velrhopfinal.z = float(velp1.z / a_corr3.a11);
					}
				}
				//-Store the results.
				rhopfinal = (rhopfinal != FLT_MAX ? rhopfinal : CTE.rhopzero);
				if (tslip == SLIP_Vel0) {//-DBC vel=0
					velrhop[p1].w = rhopfinal;
					sigma[p1] = sigmafinal;//mdbr
				}
				if (tslip == SLIP_NoSlip) {//-No-Slip
					const float3 v = motionvel[p1];
					velrhop[p1] = make_float4(v.x + v.x - velrhopfinal.x, v.y + v.y - velrhopfinal.y, v.z + v.z - velrhopfinal.z, rhopfinal);
					sigma[p1] = sigmafinal;//mdbr
				}
				if (tslip == SLIP_FreeSlip) {//-No-Penetration and free slip    SHABA
					float3 FSVelFinal; // final free slip boundary velocity
					const float3 v = motionvel[p1];
					float motion = sqrt(v.x*v.x + v.y*v.y + v.z*v.z); // to check if boundary moving
					float norm = sqrt(bnormalp1.x*bnormalp1.x + bnormalp1.y*bnormalp1.y + bnormalp1.z*bnormalp1.z);
					float3 normal; // creating a normailsed boundary normal
					normal.x = fabs(bnormalp1.x) / norm; normal.y = fabs(bnormalp1.y) / norm; normal.z = fabs(bnormalp1.z) / norm;

					// finding the velocity componants normal and tangential to boundary 
					float3 normvel = make_float3(velrhopfinal.x*normal.x, velrhopfinal.y*normal.y, velrhopfinal.z*normal.z); // velocity in direction of normal pointin ginto fluid)
					float3 tangvel = make_float3(velrhopfinal.x - normvel.x, velrhopfinal.y - normvel.y, velrhopfinal.z - normvel.z); // velocity tangential to normal

					if (motion > 0) { // if moving boundary
						float3 normmot = make_float3(v.x*normal.x, v.y*normal.y, v.z*normal.z); // boundary motion in direction normal to boundary 
						FSVelFinal = make_float3(normmot.x + normmot.x - normvel.x, normmot.y + normmot.y - normvel.y, normmot.z + normmot.z - normvel.z);
						// only velocity in normal direction for no-penetration
						// fluid sees zero velocity in the tangetial direction
					}
					else {
						FSVelFinal = make_float3(tangvel.x - normvel.x, tangvel.y - normvel.y, tangvel.z - normvel.z);
						// tangential velocity equal to fluid velocity for free slip
						// normal velocity reversed for no-penetration
					}

					// Save the velocity and density
					velrhop[p1] = make_float4(FSVelFinal.x, FSVelFinal.y, FSVelFinal.z, rhopfinal);
					sigma[p1] = sigmafinal;//mdbr
				}
			}
		}
	}
}
//==============================================================================
/// Calculates extrapolated data on boundary particles from fluid domain for mDBC.
/// Calcula datos extrapolados en el contorno para mDBC.
//==============================================================================
template<TpKernel tker, bool sim2d, TpSlipMode tslip> void Interaction_MdbcCorrectionNNT2(
	bool fastsingle, unsigned n, unsigned nbound, float mdbcthreshold, const StDivDataGpu &dvd
	, const tdouble3 &mapposmin, const double2 *posxy, const double *posz, const float4 *poscell
	, const typecode *code, const unsigned *idp, const float3 *boundnormal, const float3 *motionvel
	, float4 *velrhop, tsymatrix3f *sigma)
{
	const int2* beginendcellfluid = dvd.beginendcell + dvd.cellfluid;
	const float determlimit = 1e-3f;
	//-Interaction GhostBoundaryNodes-Fluid.
	if (n) {
		const unsigned bsbound = 128;
		dim3 sgridb = cusph::GetSimpleGridSize(n, bsbound);
		if (fastsingle) {//-mDBC-Fast_v2
			KerInteractionMdbcCorrectionNNFluid_Fast <tker, sim2d, tslip> << <sgridb, bsbound >> > (n, nbound
				, determlimit, mdbcthreshold, Double3(mapposmin), dvd.poscellsize, poscell
				, dvd.scelldiv, dvd.nc, dvd.cellzero, beginendcellfluid
				, posxy, posz, code, idp, boundnormal, motionvel, velrhop);
			KerInteractionMdbcCorrectionNNGranular_Fast <tker, sim2d, tslip> << <sgridb, bsbound >> > (n, nbound
				, determlimit, mdbcthreshold, Double3(mapposmin), dvd.poscellsize, poscell
				, dvd.scelldiv, dvd.nc, dvd.cellzero, beginendcellfluid
				, posxy, posz, code, idp, boundnormal, motionvel, velrhop,sigma);
		}
		else {//-mDBC_v0
			KerInteractionMdbcCorrectionNNFluid_Dbl <tker, sim2d, tslip> << <sgridb, bsbound >> > (n, nbound
				, determlimit, mdbcthreshold, dvd.scelldiv, dvd.nc, dvd.cellzero, beginendcellfluid
				, posxy, posz, code, idp, boundnormal, motionvel, velrhop);
			KerInteractionMdbcCorrectionNNGranular_Dbl <tker, sim2d, tslip> << <sgridb, bsbound >> > (n, nbound
				, determlimit, mdbcthreshold, dvd.scelldiv, dvd.nc, dvd.cellzero, beginendcellfluid
				, posxy, posz, code, idp, boundnormal, motionvel, velrhop,sigma);
		}
	}
}
//==============================================================================
template<TpKernel tker> void Interaction_MdbcCorrectionNNT(bool simulate2d
	, TpSlipMode slipmode, bool fastsingle, unsigned n, unsigned nbound
	, float mdbcthreshold, const StDivDataGpu &dvd, const tdouble3 &mapposmin
	, const double2 *posxy, const double *posz, const float4 *poscell, const typecode *code
	, const unsigned *idp, const float3 *boundnormal, const float3 *motionvel, float4 *velrhop, tsymatrix3f *sigma)
{
	switch (slipmode) {
	case SLIP_Vel0: { const TpSlipMode tslip = SLIP_Vel0;
		if (simulate2d)Interaction_MdbcCorrectionNNT2 <tker, true, tslip>(fastsingle, n, nbound, mdbcthreshold, dvd, mapposmin, posxy, posz, poscell, code, idp, boundnormal, motionvel, velrhop, sigma);
		else          Interaction_MdbcCorrectionNNT2 <tker, false, tslip>(fastsingle, n, nbound, mdbcthreshold, dvd, mapposmin, posxy, posz, poscell, code, idp, boundnormal, motionvel, velrhop, sigma);
	}break;
#ifndef DISABLE_MDBC_EXTRAMODES
	case SLIP_NoSlip: { const TpSlipMode tslip = SLIP_NoSlip;
		if (simulate2d)Interaction_MdbcCorrectionNNT2 <tker, true, tslip>(fastsingle, n, nbound, mdbcthreshold, dvd, mapposmin, posxy, posz, poscell, code, idp, boundnormal, motionvel, velrhop, sigma);
		else          Interaction_MdbcCorrectionNNT2 <tker, false, tslip>(fastsingle, n, nbound, mdbcthreshold, dvd, mapposmin, posxy, posz, poscell, code, idp, boundnormal, motionvel, velrhop, sigma);
	}break;
	case SLIP_FreeSlip: { const TpSlipMode tslip = SLIP_FreeSlip;
		if (simulate2d)Interaction_MdbcCorrectionNNT2 <tker, true, tslip>(fastsingle, n, nbound, mdbcthreshold, dvd, mapposmin, posxy, posz, poscell, code, idp, boundnormal, motionvel, velrhop, sigma);
		else          Interaction_MdbcCorrectionNNT2 <tker, false, tslip>(fastsingle, n, nbound, mdbcthreshold, dvd, mapposmin, posxy, posz, poscell, code, idp, boundnormal, motionvel, velrhop, sigma);
	}break;
#endif
	default: throw "SlipMode unknown at Interaction_MdbcCorrectionT().";
	}
}
//==============================================================================
/// Calculates extrapolated data on boundary particles from fluid domain for mDBC.
/// Calcula datos extrapolados en el contorno para mDBC.
//==============================================================================
void Interaction_MdbcCorrectionNN(TpKernel tkernel, bool simulate2d, TpSlipMode slipmode
	, bool fastsingle, unsigned n, unsigned nbound, float mdbcthreshold
	, const StDivDataGpu &dvd, const tdouble3 &mapposmin
	, const double2 *posxy, const double *posz, const float4 *poscell, const typecode *code
	, const unsigned *idp, const float3 *boundnormal, const float3 *motionvel, float4 *velrhop,tsymatrix3f *sigma)
{
	switch (tkernel) {
	case KERNEL_Wendland: { const TpKernel tker = KERNEL_Wendland;
		Interaction_MdbcCorrectionNNT <tker>(simulate2d, slipmode, fastsingle, n, nbound, mdbcthreshold
			, dvd, mapposmin, posxy, posz, poscell, code, idp, boundnormal, motionvel, velrhop, sigma);
	}break;
#ifndef DISABLE_KERNELS_EXTRA
	case KERNEL_Cubic: { const TpKernel tker = KERNEL_Cubic;
		Interaction_MdbcCorrectionNNT <tker>(simulate2d, slipmode, fastsingle, n, nbound, mdbcthreshold
			, dvd, mapposmin, posxy, posz, poscell, code, idp, boundnormal, motionvel, velrhop, sigma);
	}break;
#endif
	default: throw "Kernel unknown at Interaction_MdbcCorrectionNN().";
	}
}

//======================End of Multi-layer MDBC=======================================


} //end of file