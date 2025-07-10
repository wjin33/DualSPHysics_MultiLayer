//HEAD_DSPH
/*
 <DUALSPHYSICS>  Copyright (c) 2020 by Dr Jose M. Dominguez et al. (see http://dual.sphysics.org/index.php/developers/). 

 EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
 School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

 This file is part of DualSPHysics. 

 DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or (at your option) any later version. 

 DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. 

 You should have received a copy of the GNU General Public License, along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>. 
*/

/// \file JSphGpuSimple_ker.cu \brief Implements functions and CUDA kernels for the Particle Interaction and System Update.

#include "JSphGpuSimple_ker.h"
//#include "Functions.h"
//#include "FunctionsCuda.h"
//#include <math_constants.h>
//#include "JDgKerPrint.h"
//#include "JDgKerPrint_ker.h"
#include <cfloat>

#define MAXNUMBERPHASE 10

__constant__ StPhaseDruckerPrager PHASEDRUCKERPRAGER[MAXNUMBERPHASE];

namespace cusphs{
#include "FunctionsBasic_iker.h"


//##############################################################################
//# Kernels to prepare data before Interaction_Forces().
//##############################################################################
//------------------------------------------------------------------------------
/// Update PosCellg[] according to current position of particles.
/// Actualiza PosCellg[] segun la posicion de las particulas.
//------------------------------------------------------------------------------
__global__ void KerUpdatePosCell(unsigned np,double3 posmin,float poscellsize
  ,const double2 *posxy,const double *posz,float4 *poscell)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<np){
    const double2 rxy=posxy[p];
    const double dx=rxy.x-posmin.x;
    const double dy=rxy.y-posmin.y;
    const double dz=posz[p]-posmin.z;
    const unsigned cx=unsigned(dx/poscellsize);
    const unsigned cy=unsigned(dy/poscellsize);
    const unsigned cz=unsigned(dz/poscellsize);
    const float px=float(dx-(double(poscellsize)*cx));
    const float py=float(dy-(double(poscellsize)*cy));
    const float pz=float(dz-(double(poscellsize)*cz));
    const float pw=__uint_as_float(CEL_Code(cx,cy,cz));
    poscell[p]=make_float4(px,py,pz,pw);
  }
}
//==============================================================================
/// Update PosCellg[] according to current position of particles.
/// Actualiza PosCellg[] segun la posicion de las particulas.
//==============================================================================
void UpdatePosCell(unsigned np,tdouble3 posmin,float poscellsize
  ,const double2 *posxy,const double *posz,float4 *poscell,cudaStream_t stm)
{
  const dim3 sgrid=GetSimpleGridSize(np,SPHBSIZE);
  if(np)KerUpdatePosCell <<<sgrid,SPHBSIZE,0,stm>>> (np,Double3(posmin),poscellsize,posxy,posz,poscell);
}

//------------------------------------------------------------------------------
/// Initialises ace array with 0 for bound and gravity for fluid.
/// Inicializa el array ace con 0 para contorno y gravity para fluido.
//------------------------------------------------------------------------------
__global__ void KerInitAceGravity(unsigned np,unsigned npb,float3 gravity,float3 *ace)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<np){
    ace[p]=(p<npb? make_float3(0,0,0): gravity);
  }
}
//==============================================================================
/// Initialises ace array with 0 for bound and gravity for fluid.
/// Inicializa el array ace con 0 para contorno y gravity para fluido.
//==============================================================================
void InitAceGravity(unsigned np,unsigned npb,tfloat3 gravity,float3 *ace,cudaStream_t stm){
  if(np){
    dim3 sgrid=GetSimpleGridSize(np,SPHBSIZE);
    KerInitAceGravity <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,Float3(gravity),ace);
  }
}


//##############################################################################
//# Kernels to run after Interaction_Forces().
//##############################################################################
//------------------------------------------------------------------------------
/// Sets v[].y to zero.
/// Pone v[].y a cero.
//------------------------------------------------------------------------------
__global__ void KerResety(unsigned n,unsigned ini,float3 *v)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n)v[p+ini].y=0;
}
//==============================================================================
/// Sets v[].y to zero.
/// Pone v[].y a cero.
//==============================================================================
void Resety(unsigned n,unsigned ini,float3 *v,cudaStream_t stm){
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerResety <<<sgrid,SPHBSIZE,0,stm>>> (n,ini,v);
  }
}

//========Implement Soil Constitutive Model=========


//==========Include DP without softening for testing===
//==============================================================================
__device__ void GetStressInvariant(float sigmaxx, float sigmayy, float sigmazz, float sigmaxy, float sigmayz, float sigmaxz,float &I1,float &J2)
{
  I1= sigmaxx + sigmayy + sigmazz;
  J2=((sigmaxx-sigmazz)*(sigmaxx-sigmazz)+(sigmayy-sigmazz)*(sigmayy-sigmazz)+(sigmayy-sigmaxx)*(sigmayy-sigmaxx))/6.+sigmaxy*sigmaxy+sigmayz*sigmayz+sigmaxz*sigmaxz;
}

__device__ void GetDPYieldFunction(float &f, float I1, float J2, const float DP_phi, const float DP_kc)
{
   f=sqrt(J2)+DP_phi*I1-DP_kc;
}
__device__ void ConsRelationEP(tsymatrix3f sigma, tsymatrix3f& nsigma
		, const float E_ModulusK, const float E_ModulusG, const float MC_phi, const float MC_c, const float MC_psi)
{//elastic predictor-plastic corrector
    double tsigma_xx, tsigma_yy, tsigma_zz, tsigma_xy, tsigma_yz, tsigma_xz, tk;// trial value
	unsigned iter;//number of iterations

	float ModulusK = E_ModulusK;
	float ModulusG = E_ModulusG;
	const float phi = MC_phi;
	const float coh = MC_c;
	const float psi = MC_psi;
	//Build Elastic stiffness matrix
	float K4G3 = float(ModulusK + 4.*ModulusG/3.);
	float K2G3 = float(ModulusK- 2.*ModulusG/3.);
	//Build Inversed elastic stiffness matrix
	float invK4G3 = (K2G3 + K4G3) / (-2 * K2G3*K2G3 + K2G3*K4G3 + K4G3*K4G3);
	float invK2G3 = -K2G3 / (-2 * K2G3*K2G3 + K2G3*K4G3 + K4G3*K4G3);
	float invm_a11 = invK4G3; float invm_a12 = invK2G3; float invm_a13 = invK2G3;
	float invm_a21 = invK2G3; float invm_a22 = invK4G3; float invm_a23 = invK2G3;
	float invm_a31 = invK2G3; float invm_a32 = invK2G3; float invm_a33 = invK4G3;
	//trail stress from elastic update
    tsigma_xx = sigma.xx;
    tsigma_yy = sigma.yy;
    tsigma_zz = sigma.zz;
    tsigma_xy = sigma.xy;
    tsigma_yz = sigma.yz;
    tsigma_xz = sigma.xz;
	iter = 0;
	//evaluate yeild condition
	///Plain strain
	//float DP_phi = tan(phi)/sqrt(9.+12.*tan(phi)*tan(phi)); // Make it as a function
    //float DP_kc = 3.*coh/sqrt(9.+12.*tan(phi)*tan(phi)); 
    //float DP_psi = tan(psi)/sqrt(9.+12.*tan(psi)*tan(psi)); //
	///Highest set
	float DP_phi = 2.f*sin(phi)/((3.f-sin(phi))* 1.732f);
	float DP_kc = 6.*coh*cos(phi)/((3.f-sin(phi))* 1.732f);
	float DP_psi = 2.f*sin(psi)/((3.f-sin(psi))* 1.732f);
	///Medium set
	//float DP_phi = 2.f*sin(phi)/((3.f+sin(phi))* 1.732f);
	//float DP_kc = 6.*coh*cos(phi)/((3.f+sin(phi))* 1.732f);
	//float DP_psi = 2.f*sin(psi)/((3.f+sin(psi))* 1.732f);
	///Lowest set
	//float DP_phi = sin(phi)/sqrt(9.f+3.f*sin(phi)*sin(phi));
	//float DP_kc = 3.*coh*cos(phi)/sqrt(9.f+3.f*sin(phi)*sin(phi));
	//float DP_psi = sin(psi)/sqrt(9.f+3.f*sin(psi)*sin(psi));
	float f=0, I1=0, J2=0;
	GetStressInvariant(tsigma_xx,tsigma_yy,tsigma_zz,tsigma_xy,tsigma_yz,tsigma_xz,I1,J2);
	GetDPYieldFunction(f,I1,J2,DP_phi,DP_kc);
	if (f<0) {//elastic 
		//Update plastic strain internal variable & stress
		nsigma.xx = tsigma_xx;
		nsigma.yy = tsigma_yy;
		nsigma.zz = tsigma_zz;
		nsigma.xy = tsigma_xy;
		nsigma.yz = tsigma_yz;
		nsigma.xz = tsigma_xz;
	}	
	else {//plastic corrector
		float err = 1e-5;
		double dsigmap_xx = 0, dsigmap_yy = 0, dsigmap_zz = 0, dsigmap_xy = 0, dsigmap_yz = 0, dsigmap_xz = 0
			 , depsp_xx = 0, depsp_yy = 0, depsp_zz = 0, depsp_xy = 0, depsp_yz = 0, depsp_xz = 0, dk = 0;
		while (f > err) {
			//MODIFY PLASTIC MULTIPLFY
			double dlambda = f /(9.*ModulusK*DP_phi*DP_psi+ModulusG);
			float GJ2=ModulusG/sqrt(J2);
			//evaluate De : plastic potential
      double Deppxx = 3.f*ModulusK*DP_psi+GJ2*(tsigma_xx-I1/3.f); 
			double Deppyy = 3.f*ModulusK*DP_psi+GJ2*(tsigma_yy-I1/3.f);
			double Deppzz = 3.f*ModulusK*DP_psi+GJ2*(tsigma_zz-I1/3.f);
			double Deppxy = GJ2*tsigma_xy;
			double Deppyz = GJ2*tsigma_yz;
			double Deppxz = GJ2*tsigma_xz;
			//Compute plastic stress increment
			dsigmap_xx = dlambda*Deppxx;
			dsigmap_yy = dlambda*Deppyy;
			dsigmap_zz = dlambda*Deppzz;
			dsigmap_xy = dlambda*Deppxy;
			dsigmap_yz = dlambda*Deppyz;
			dsigmap_xz = dlambda*Deppxz;
			//Update new stress matrix
			tsigma_xx = tsigma_xx - dsigmap_xx;
			tsigma_yy = tsigma_yy - dsigmap_yy;
			tsigma_zz = tsigma_zz - dsigmap_zz;
			tsigma_xy = tsigma_xy - dsigmap_xy;
			tsigma_yz = tsigma_yz - dsigmap_yz;
			tsigma_xz = tsigma_xz - dsigmap_xz;
			//Compute plastic strain increment
			 depsp_xx = invm_a11*dsigmap_xx + invm_a12*dsigmap_yy + invm_a13*dsigmap_zz;
			 depsp_yy = invm_a21*dsigmap_xx + invm_a22*dsigmap_yy + invm_a23*dsigmap_zz;
			 depsp_zz = invm_a31*dsigmap_xx + invm_a32*dsigmap_yy + invm_a33*dsigmap_zz;
			 depsp_xy = 0.5f/ModulusG*dsigmap_xy;
			 depsp_yz = 0.5f/ModulusG*dsigmap_yz;
			 depsp_xz = 0.5f/ModulusG*dsigmap_xz; 
			//bulk plastic strain increment
			float depsp_b = (depsp_xx + depsp_yy + depsp_zz) / 3.f;
			//deviatoric trail plastic strain
			float dtepsp_xx = depsp_xx - depsp_b;
			float dtepsp_yy = depsp_yy - depsp_b;
			float dtepsp_zz = depsp_zz - depsp_b;
			float dtepsp_xy = depsp_xy;
			float dtepsp_yz = depsp_yz;
			float dtepsp_xz = depsp_xz;
			//Update internal variable
			dk = sqrt((2. / 3.)*(dtepsp_xx*dtepsp_xx + dtepsp_yy*dtepsp_yy + dtepsp_zz*dtepsp_zz + 2.*dtepsp_xy*dtepsp_xy+ 2.*dtepsp_yz*dtepsp_yz +2.*dtepsp_xz*dtepsp_xz));
			tk = tk + dk;
			//Update stress invariant
			GetStressInvariant(tsigma_xx,tsigma_yy,tsigma_zz,tsigma_xy,tsigma_yz,tsigma_xz,I1,J2);
			//Update yeild function
			GetDPYieldFunction(f,I1,J2,DP_phi,DP_kc);
			iter = iter + 1;
			//double kc = 0, alphap = 0;
			//DPVariable(tk, alphap, kc);
			float kalpha = DP_kc / DP_phi;
			
			if (I1 > kalpha)// I1 > kalpha check tensile cracking & perform stress scaling
			{
				tsigma_xx = tsigma_xx - (I1 - kalpha) / 3.;
				tsigma_yy = tsigma_yy - (I1 - kalpha) / 3.;
				tsigma_zz = tsigma_zz - (I1 - kalpha) / 3.;
				GetStressInvariant(tsigma_xx,tsigma_yy,tsigma_zz,tsigma_xy,tsigma_yz,tsigma_xz,I1,J2);

				float fscale = (-DP_phi * I1 + DP_kc) / sqrt(J2);
				if (fscale < 1.f)// stress scaling
				{
					tsigma_xx = fscale * (tsigma_xx-I1/3.f) + I1 / 3.;
					tsigma_yy = fscale * (tsigma_yy-I1/3.f) + I1 / 3.;
					tsigma_zz = fscale * (tsigma_zz-I1/3.f) + I1 / 3.;
					tsigma_xy = fscale * tsigma_xy;
					tsigma_yz = fscale * tsigma_yz;
					tsigma_xz = fscale * tsigma_xz;
				GetStressInvariant(tsigma_xx,tsigma_yy,tsigma_zz,tsigma_xy,tsigma_yz,tsigma_xz,I1,J2);
				}
				GetDPYieldFunction(f,I1,J2,DP_phi,DP_kc);
			}
			
		}
		//Update plastic strain internal variable & stress
		nsigma.xx = float(tsigma_xx);
		nsigma.yy = float(tsigma_yy);
		nsigma.zz = float(tsigma_zz);
		nsigma.xy = float(tsigma_xy);
		nsigma.yz = float(tsigma_yz);
		nsigma.xz = float(tsigma_xz);
	}
}

//##############################################################################
//# Kernels for ComputeStep (vel & rhop).
//# Kernels para ComputeStep (vel & rhop).
//##############################################################################
//------------------------------------------------------------------------------
/// Computes new values for Pos, Check, Vel and Ros (using Verlet).
/// The value of Vel always set to be reset.
///
/// Calcula nuevos valores de  Pos, Check, Vel y Rhop (usando Verlet).
/// El valor de Vel para bound siempre se pone a cero.
//------------------------------------------------------------------------------
template<bool floating,bool shift,bool inout> __global__ void KerComputeStepVerlet
  (unsigned n,unsigned npb,float rhopzero,float rhopoutmin,float rhopoutmax
  ,const float4 *velrhop1,const float4 *velrhop2
  ,const float *ar,const float3 *ace,const float4 *shiftposfs,const float3 *indirvel
  ,double dt,double dt205,double dt2,float3 gravity
  ,double2 *movxy,double *movz,typecode *code,float4 *velrhopnew)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    if(p<npb){ //-Particles: Fixed & Moving.
      float rrhop=float(double(velrhop2[p].w)+dt2*ar[p]);
      rrhop=(rrhop<rhopzero? rhopzero: rrhop); //-To prevent absorption of fluid particles by boundaries. | Evita q las boundary absorvan a las fluidas.
      velrhopnew[p]=make_float4(0,0,0,rrhop);
    }
    else{ //-Particles: Floating & Fluid.
      const typecode rcode=code[p];
      //-Updates density.
      const float4 rvelrhop2=velrhop2[p];
      const float rhopnew=float(double(velrhop2[p].w)+dt2*ar[p]);
      float4 rvel1=velrhop1[p];
      if(!floating || CODE_IsFluid(rcode)){ //-Particles: Fluid.
        //-Calculate displacement. | Calcula desplazamiento.
        const float3 race=ace[p];
        const double acegrx=double(race.x)+gravity.x;
        const double acegry=double(race.y)+gravity.y;
        const double acegrz=double(race.z)+gravity.z;
        double dx=double(rvel1.x)*dt + acegrx*dt205;
        double dy=double(rvel1.y)*dt + acegry*dt205;
        double dz=double(rvel1.z)*dt + acegrz*dt205;
        if(shift){
          const float4 rshiftpos=shiftposfs[p];
          dx+=double(rshiftpos.x);
          dy+=double(rshiftpos.y);
          dz+=double(rshiftpos.z);
        }
        bool outrhop=(rhopnew<rhopoutmin || rhopnew>rhopoutmax);
        //-Calculate velocity & density. | Calcula velocidad y densidad.
        float4 rvelrhopnew=make_float4(
          float(double(rvelrhop2.x) + acegrx*dt2),
          float(double(rvelrhop2.y) + acegry*dt2),
          float(double(rvelrhop2.z) + acegrz*dt2),
          rhopnew);
        //-Restore data of inout particles.
        if(inout && CODE_IsFluidInout(rcode)){
          outrhop=false;
          rvelrhopnew=rvelrhop2;
          const float3 vd=indirvel[CODE_GetIzoneFluidInout(rcode)];
          if(vd.x!=FLT_MAX){
            const float v=rvel1.x*vd.x + rvel1.y*vd.y + rvel1.z*vd.z;
            dx=double(v*vd.x) * dt;
            dy=double(v*vd.y) * dt;
            dz=double(v*vd.z) * dt;
          }
          else{
            dx=double(rvel1.x) * dt;
            dy=double(rvel1.y) * dt;
            dz=double(rvel1.z) * dt;
          }
        }
        //-Update particle data.
        movxy[p]=make_double2(dx,dy);
        movz[p]=dz;
        if(outrhop){ //-Only brands as excluded normal particles (not periodic). | Solo marca como excluidas las normales (no periodicas).
          if(CODE_IsNormal(rcode))code[p]=CODE_SetOutRhop(rcode);
        }
        velrhopnew[p]=rvelrhopnew;
      }
      else{ //-Particles: Floating.
        rvel1.w=(rhopnew<rhopzero? rhopzero: rhopnew); //-To prevent absorption of fluid particles by boundaries. | Evita q las floating absorvan a las fluidas.
        velrhopnew[p]=rvel1;
      }
    }
  }
}
//==============================================================================
/// Updates particles according to forces and dt using Verlet. 
/// Actualizacion de particulas segun fuerzas y dt usando Verlet.
//==============================================================================
void ComputeStepVerlet(bool floating,bool shift,bool inout,unsigned np,unsigned npb
  ,const float4 *velrhop1,const float4 *velrhop2
  ,const float *ar,const float3 *ace,const float4 *shiftposfs,const float3 *indirvel
  ,double dt,double dt2,float rhopzero,float rhopoutmin,float rhopoutmax,tfloat3 gravity
  ,typecode *code,double2 *movxy,double *movz,float4 *velrhopnew,cudaStream_t stm)
{
  double dt205=(0.5*dt*dt);
  if(np){
    dim3 sgrid=GetSimpleGridSize(np,SPHBSIZE);
    if(inout){      const bool tinout=true;
      if(shift){    const bool shift=true;
        if(floating)KerComputeStepVerlet<true ,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,rhopzero,rhopoutmin,rhopoutmax,velrhop1,velrhop2,ar,ace,shiftposfs,indirvel,dt,dt205,dt2,Float3(gravity),movxy,movz,code,velrhopnew);
        else        KerComputeStepVerlet<false,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,rhopzero,rhopoutmin,rhopoutmax,velrhop1,velrhop2,ar,ace,shiftposfs,indirvel,dt,dt205,dt2,Float3(gravity),movxy,movz,code,velrhopnew);
      }else{        const bool shift=false;
        if(floating)KerComputeStepVerlet<true ,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,rhopzero,rhopoutmin,rhopoutmax,velrhop1,velrhop2,ar,ace,shiftposfs,indirvel,dt,dt205,dt2,Float3(gravity),movxy,movz,code,velrhopnew);
        else        KerComputeStepVerlet<false,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,rhopzero,rhopoutmin,rhopoutmax,velrhop1,velrhop2,ar,ace,shiftposfs,indirvel,dt,dt205,dt2,Float3(gravity),movxy,movz,code,velrhopnew);
      }
    }
    else{           const bool tinout=false;
      if(shift){    const bool shift=true;
        if(floating)KerComputeStepVerlet<true ,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,rhopzero,rhopoutmin,rhopoutmax,velrhop1,velrhop2,ar,ace,shiftposfs,indirvel,dt,dt205,dt2,Float3(gravity),movxy,movz,code,velrhopnew);
        else        KerComputeStepVerlet<false,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,rhopzero,rhopoutmin,rhopoutmax,velrhop1,velrhop2,ar,ace,shiftposfs,indirvel,dt,dt205,dt2,Float3(gravity),movxy,movz,code,velrhopnew);
      }else{        const bool shift=false;
        if(floating)KerComputeStepVerlet<true ,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,rhopzero,rhopoutmin,rhopoutmax,velrhop1,velrhop2,ar,ace,shiftposfs,indirvel,dt,dt205,dt2,Float3(gravity),movxy,movz,code,velrhopnew);
        else        KerComputeStepVerlet<false,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,rhopzero,rhopoutmin,rhopoutmax,velrhop1,velrhop2,ar,ace,shiftposfs,indirvel,dt,dt205,dt2,Float3(gravity),movxy,movz,code,velrhopnew);
      }
    }
  }
}

//------------------------------------------------------------------------------
/// Computes new values for Pos, Check, Vel and Ros (used with Symplectic-Predictor).
/// Calcula los nuevos valores de Pos, Vel y Rhop (usando para Symplectic-Predictor).
//------------------------------------------------------------------------------
template<bool floating,bool shift,bool inout> __global__ void KerComputeStepSymplecticPre
  (unsigned n,unsigned npb
  ,const float4 *velrhoppre,const float *ar,const float3 *ace,const float4 *shiftposfs
  ,const tsymatrix3f *sigmapre,const tsymatrix3f *rsigma
  ,const float3 *indirvel,double dtm,float rhopzero,float rhopoutmin,float rhopoutmax,float3 gravity
  ,typecode *code,double2 *movxy,double *movz,float4 *velrhop,tsymatrix3f *sigma)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    if(p<npb){ //-Particles: Fixed & Moving.
      float4 rvelrhop=velrhoppre[p];
      rvelrhop.w=float(double(rvelrhop.w)+dtm*ar[p]);
      rvelrhop.w=(rvelrhop.w<rhopzero? rhopzero: rvelrhop.w); //-To prevent absorption of fluid particles by boundaries. | Evita que las boundary absorvan a las fluidas.
      velrhop[p]=rvelrhop;
      //==== mdbr ==boundary keeps the same
	    sigma[p] = sigmapre[p];
    }
    else{ //-Particles: Floating & Fluid.
      const typecode rcode=code[p];
      const typecode pp1=CODE_GetTypeValue(rcode);
      //-Updates density.
      const float4 rvelrhoppre=velrhoppre[p];
      float4 rvelrhopnew=rvelrhoppre;
      rvelrhopnew.w=float(double(rvelrhoppre.w)+dtm*ar[p]);
      if(!floating || CODE_IsFluid(rcode)){ //-Particles: Fluid.
        //-Calculate displacement. | Calcula desplazamiento.
        double dx=double(rvelrhoppre.x)*dtm;
        double dy=double(rvelrhoppre.y)*dtm;
        double dz=double(rvelrhoppre.z)*dtm;
        if(shift){
          const float4 rshiftpos=shiftposfs[p];
          dx+=double(rshiftpos.x);
          dy+=double(rshiftpos.y);
          dz+=double(rshiftpos.z);
        }
        bool outrhop=(rvelrhopnew.w<rhopoutmin || rvelrhopnew.w>rhopoutmax);
        //-Calculate velocity & density. | Calcula velocidad y densidad.
        const float3 race=ace[p];
        rvelrhopnew.x=float(double(rvelrhoppre.x) + (double(race.x)+gravity.x) * dtm);
        rvelrhopnew.y=float(double(rvelrhoppre.y) + (double(race.y)+gravity.y) * dtm);
        rvelrhopnew.z=float(double(rvelrhoppre.z) + (double(race.z)+gravity.z) * dtm);
        //-Calculate elastic stress
        tsymatrix3f sigma_e = { 0,0,0,0,0,0 }; tsymatrix3f sigmanew = { 0,0,0,0,0,0 };
        //float kplasticold = kplasticpre[p];
        sigma_e.xx = float(double(sigmapre[p].xx) + rsigma[p].xx * dtm);
        sigma_e.yy = float(double(sigmapre[p].yy) + rsigma[p].yy * dtm);
        sigma_e.zz = float(double(sigmapre[p].zz) + rsigma[p].zz * dtm);
        sigma_e.xy = float(double(sigmapre[p].xy) + rsigma[p].xy * dtm);
        sigma_e.yz = float(double(sigmapre[p].yz) + rsigma[p].yz * dtm);
        sigma_e.xz = float(double(sigmapre[p].xz) + rsigma[p].xz * dtm);
        //-Plastic corrector
        const float DP_K=PHASEDRUCKERPRAGER[pp1].DP_K; ///<  Elastic bulk modulus
        const float DP_G=PHASEDRUCKERPRAGER[pp1].DP_G;    ///< Elastic shear modulus
        const float MC_phi=PHASEDRUCKERPRAGER[pp1].MC_phi;    ///< Friction angle in MC model, to be converted to DP yield surface parameters DP_AlphaPhi and DP_kc
        const float MC_c= PHASEDRUCKERPRAGER[pp1].MC_c;    ///< Cohesion in MC model, to be converted to DP yield surface parameters DP_AlphaPhi and DP_kc
        const float MC_psi = PHASEDRUCKERPRAGER[pp1].MC_psi;    ///< Dilatancy angle in MC model, to be converted to DP non-associate flow rule parameter DP_psi
        //sigmanew=sigma_e;
        ConsRelationEP(sigma_e,sigmanew,DP_K,DP_G,MC_phi,MC_c,MC_psi);
        //-Update stress/equivelant plastic strain
        sigma[p]=sigmanew;
        //-Restore data of inout particles.
        if(inout && CODE_IsFluidInout(rcode)){
          outrhop=false;
          rvelrhopnew=rvelrhoppre;
          const float3 vd=indirvel[CODE_GetIzoneFluidInout(rcode)];
          if(vd.x!=FLT_MAX){
            const float v=rvelrhopnew.x*vd.x + rvelrhopnew.y*vd.y + rvelrhopnew.z*vd.z;
            dx=double(v*vd.x) * dtm;
            dy=double(v*vd.y) * dtm;
            dz=double(v*vd.z) * dtm;
          }
        }
        //-Update particle data.
        movxy[p]=make_double2(dx,dy);
        movz[p]=dz;
        if(outrhop){ //-Only brands as excluded normal particles (not periodic). | Solo marca como excluidas las normales (no periodicas).
          if(CODE_IsNormal(rcode))code[p]=CODE_SetOutRhop(rcode);
        }
      }
      else{ //-Particles: Floating.
        rvelrhopnew.w=(rvelrhopnew.w<rhopzero? rhopzero: rvelrhopnew.w); //-To prevent absorption of fluid particles by boundaries. | Evita q las floating absorvan a las fluidas.
      }
      //-Stores new velocity and density.
      velrhop[p]=rvelrhopnew;
    }
  }
}
//==============================================================================
/// Updates particles using Symplectic-Predictor.
/// Actualizacion de particulas usando Symplectic-Predictor.
//==============================================================================   
void ComputeStepSymplecticPre(bool floating,bool shift,bool inout,unsigned np,unsigned npb
  ,const float4 *velrhoppre,const float *ar,const float3 *ace,const float4 *shiftposfs
  ,const tsymatrix3f *sigmapre,const tsymatrix3f *rsigma
  ,const float3 *indirvel,double dtm,float rhopzero,float rhopoutmin,float rhopoutmax,tfloat3 gravity
  ,typecode *code,double2 *movxy,double *movz,float4 *velrhop,tsymatrix3f *sigma,cudaStream_t stm)
{
  if(np){
    dim3 sgrid=GetSimpleGridSize(np,SPHBSIZE);
    if(inout){      const bool tinout=true;
      if(shift){    const bool shift=true;
        if(floating)KerComputeStepSymplecticPre<true ,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,velrhoppre,ar,ace,shiftposfs,sigmapre,rsigma,indirvel,dtm,rhopzero,rhopoutmin,rhopoutmax,Float3(gravity),code,movxy,movz,velrhop,sigma);
        else        KerComputeStepSymplecticPre<false,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,velrhoppre,ar,ace,shiftposfs,sigmapre,rsigma,indirvel,dtm,rhopzero,rhopoutmin,rhopoutmax,Float3(gravity),code,movxy,movz,velrhop,sigma);
      }else{        const bool shift=false;
        if(floating)KerComputeStepSymplecticPre<true ,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,velrhoppre,ar,ace,shiftposfs,sigmapre,rsigma,indirvel,dtm,rhopzero,rhopoutmin,rhopoutmax,Float3(gravity),code,movxy,movz,velrhop,sigma);
        else        KerComputeStepSymplecticPre<false,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,velrhoppre,ar,ace,shiftposfs,sigmapre,rsigma,indirvel,dtm,rhopzero,rhopoutmin,rhopoutmax,Float3(gravity),code,movxy,movz,velrhop,sigma);
      }
    }
    else{           const bool tinout=false;
      if(shift){    const bool shift=true;
        if(floating)KerComputeStepSymplecticPre<true ,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,velrhoppre,ar,ace,shiftposfs,sigmapre,rsigma,indirvel,dtm,rhopzero,rhopoutmin,rhopoutmax,Float3(gravity),code,movxy,movz,velrhop,sigma);
        else        KerComputeStepSymplecticPre<false,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,velrhoppre,ar,ace,shiftposfs,sigmapre,rsigma,indirvel,dtm,rhopzero,rhopoutmin,rhopoutmax,Float3(gravity),code,movxy,movz,velrhop,sigma);
      }else{        const bool shift=false;
        if(floating)KerComputeStepSymplecticPre<true ,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,velrhoppre,ar,ace,shiftposfs,sigmapre,rsigma,indirvel,dtm,rhopzero,rhopoutmin,rhopoutmax,Float3(gravity),code,movxy,movz,velrhop,sigma);
        else        KerComputeStepSymplecticPre<false,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,velrhoppre,ar,ace,shiftposfs,sigmapre,rsigma,indirvel,dtm,rhopzero,rhopoutmin,rhopoutmax,Float3(gravity),code,movxy,movz,velrhop,sigma);
      }
    }
  }
}

//------------------------------------------------------------------------------
/// Computes new values for Pos, Check, Vel and Ros (using Verlet).
/// The value of Vel always set to be reset.
///
/// Calcula los nuevos valores de Pos, Vel y Rhop (usandopara Symplectic-Corrector).
/// Pone vel de contorno a cero.
//------------------------------------------------------------------------------
template<bool floating,bool shift,bool inout> __global__ void KerComputeStepSymplecticCor
  (unsigned n,unsigned npb
  ,const float4 *velrhoppre,const float *ar,const float3 *ace,const float4 *shiftposfs
  ,const tsymatrix3f *sigmapre,const tsymatrix3f *rsigma
  ,const float3 *indirvel,double dtm,double dt,float rhopzero,float rhopoutmin,float rhopoutmax,float3 gravity
  ,typecode *code,double2 *movxy,double *movz,float4 *velrhop,tsymatrix3f *sigma)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    if(p<npb){ //-Particles: Fixed & Moving.
      double epsilon_rdot=(-double(ar[p])/double(velrhop[p].w))*dt;
      float rrhop=float(double(velrhoppre[p].w) * (2.-epsilon_rdot)/(2.+epsilon_rdot));
      rrhop=(rrhop<rhopzero? rhopzero: rrhop); //-To prevent absorption of fluid particles by boundaries. | Evita q las boundary absorvan a las fluidas.
      velrhop[p]=make_float4(0,0,0,rrhop);
       //==== mdbr ==boundary keeps the same
      sigma[p] = sigmapre[p];
    }
    else{ //-Particles: Floating & Fluid.
      const typecode rcode=code[p];
      const typecode pp1=CODE_GetTypeValue(rcode);
      //-Updates density.
      const double epsilon_rdot=(-double(ar[p])/double(velrhop[p].w))*dt;
      const float4 rvelrhoppre=velrhoppre[p];
      float4 rvelrhopnew=rvelrhoppre;
      rvelrhopnew.w=float(double(rvelrhoppre.w) * (2.-epsilon_rdot)/(2.+epsilon_rdot));
      if(!floating || CODE_IsFluid(rcode)){//-Particles: Fluid.
        //-Calculate velocity. | Calcula velocidad.
        const float3 race=ace[p];
        rvelrhopnew.x=float(double(rvelrhoppre.x) + (double(race.x)+gravity.x) * dt);
        rvelrhopnew.y=float(double(rvelrhoppre.y) + (double(race.y)+gravity.y) * dt);
        rvelrhopnew.z=float(double(rvelrhoppre.z) + (double(race.z)+gravity.z) * dt);
        //-Calculate displacement. | Calcula desplazamiento.
        double dx=(double(rvelrhoppre.x)+double(rvelrhopnew.x)) * dtm;
        double dy=(double(rvelrhoppre.y)+double(rvelrhopnew.y)) * dtm;
        double dz=(double(rvelrhoppre.z)+double(rvelrhopnew.z)) * dtm;
        if(shift){
          const float4 rshiftpos=shiftposfs[p];
          dx+=double(rshiftpos.x);
          dy+=double(rshiftpos.y);
          dz+=double(rshiftpos.z);
        }
        bool outrhop=(rvelrhopnew.w<rhopoutmin || rvelrhopnew.w>rhopoutmax);
                //-Calculate elastic stress
        tsymatrix3f sigma_e = { 0,0,0,0,0,0 }; tsymatrix3f sigmanew = { 0,0,0,0,0,0 };
        //float kplasticold = kplasticpre[p]; float kplasticnew = 0;
        sigma_e.xx = float(double(sigmapre[p].xx) + rsigma[p].xx*dt);
        sigma_e.yy = float(double(sigmapre[p].yy) + rsigma[p].yy*dt);
        sigma_e.zz = float(double(sigmapre[p].zz) + rsigma[p].zz*dt);
        sigma_e.xy = float(double(sigmapre[p].xy) + rsigma[p].xy*dt);
        sigma_e.yz = float(double(sigmapre[p].yz) + rsigma[p].yz*dt);
        sigma_e.xz = float(double(sigmapre[p].xz) + rsigma[p].xz*dt);
        //-Plastic corrector
        const float DP_K=PHASEDRUCKERPRAGER[pp1].DP_K; ///<  Elastic bulk modulus
        const float DP_G=PHASEDRUCKERPRAGER[pp1].DP_G;    ///< Elastic shear modulus
        const float MC_phi=PHASEDRUCKERPRAGER[pp1].MC_phi;    ///< Friction angle in MC model, to be converted to DP yield surface parameters DP_AlphaPhi and DP_kc
        const float MC_c= PHASEDRUCKERPRAGER[pp1].MC_c;    ///< Cohesion in MC model, to be converted to DP yield surface parameters DP_AlphaPhi and DP_kc
        const float MC_psi = PHASEDRUCKERPRAGER[pp1].MC_psi;    ///< Dilatancy angle in MC model, to be converted to DP non-associate flow rule parameter DP_psi
        //sigmanew=sigma_e;
        ConsRelationEP(sigma_e,sigmanew,DP_K,DP_G,MC_phi,MC_c,MC_psi);
        //-Update stress/equivelant plastic strain
        sigma[p] = sigmanew;
        //-Restore data of inout particles.
        if(inout && CODE_IsFluidInout(rcode)){
          outrhop=false;
          rvelrhopnew=rvelrhoppre;
          const float3 vd=indirvel[CODE_GetIzoneFluidInout(rcode)];
          if(vd.x!=FLT_MAX){
            const float v=rvelrhopnew.x*vd.x + rvelrhopnew.y*vd.y + rvelrhopnew.z*vd.z;
            dx=double(v*vd.x) * dt;
            dy=double(v*vd.y) * dt;
            dz=double(v*vd.z) * dt;
          }
          else{
            dx=double(rvelrhopnew.x) * dt; 
            dy=double(rvelrhopnew.y) * dt; 
            dz=double(rvelrhopnew.z) * dt;
          }
        }
        //-Update particle data.
        movxy[p]=make_double2(dx,dy);
        movz[p]=dz;
        if(outrhop){ //-Only brands as excluded normal particles (not periodic). | Solo marca como excluidas las normales (no periodicas).
          const typecode rcode=code[p];
          if(CODE_IsNormal(rcode))code[p]=CODE_SetOutRhop(rcode);
        }
      }
      else{ //-Particles: Floating.
        rvelrhopnew.w=(rvelrhopnew.w<rhopzero? rhopzero: rvelrhopnew.w); //-To prevent absorption of fluid particles by boundaries. | Evita q las floating absorvan a las fluidas.
      }
      //-Stores new velocity and density.
      velrhop[p]=rvelrhopnew;
    }
  }
}
//==============================================================================
/// Updates particles using Symplectic-Corrector.
/// Actualizacion de particulas usando Symplectic-Corrector.
//==============================================================================   
void ComputeStepSymplecticCor(bool floating,bool shift,bool inout,unsigned np,unsigned npb
  ,const float4 *velrhoppre,const float *ar,const float3 *ace,const float4 *shiftposfs
  ,const tsymatrix3f* sigmapre,const tsymatrix3f* rsigma
  ,const float3 *indirvel,double dtm,double dt,float rhopzero,float rhopoutmin,float rhopoutmax,tfloat3 gravity
  ,typecode *code,double2 *movxy,double *movz,float4 *velrhop,tsymatrix3f* sigma,cudaStream_t stm)
{
  if(np){
    dim3 sgrid=GetSimpleGridSize(np,SPHBSIZE);
    if(inout){      const bool tinout=true;
      if(shift){    const bool shift=true;
        if(floating)KerComputeStepSymplecticCor<true ,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,velrhoppre,ar,ace,shiftposfs,sigmapre,rsigma,indirvel,dtm,dt,rhopzero,rhopoutmin,rhopoutmax,Float3(gravity),code,movxy,movz,velrhop,sigma);
        else        KerComputeStepSymplecticCor<false,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,velrhoppre,ar,ace,shiftposfs,sigmapre,rsigma,indirvel,dtm,dt,rhopzero,rhopoutmin,rhopoutmax,Float3(gravity),code,movxy,movz,velrhop,sigma);
      }else{        const bool shift=false;
        if(floating)KerComputeStepSymplecticCor<true ,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,velrhoppre,ar,ace,shiftposfs,sigmapre,rsigma,indirvel,dtm,dt,rhopzero,rhopoutmin,rhopoutmax,Float3(gravity),code,movxy,movz,velrhop,sigma);
        else        KerComputeStepSymplecticCor<false,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,velrhoppre,ar,ace,shiftposfs,sigmapre,rsigma,indirvel,dtm,dt,rhopzero,rhopoutmin,rhopoutmax,Float3(gravity),code,movxy,movz,velrhop,sigma);
      }
    }
    else{           const bool tinout=false;
      if(shift){    const bool shift=true;
        if(floating)KerComputeStepSymplecticCor<true ,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,velrhoppre,ar,ace,shiftposfs,sigmapre,rsigma,indirvel,dtm,dt,rhopzero,rhopoutmin,rhopoutmax,Float3(gravity),code,movxy,movz,velrhop,sigma);
        else        KerComputeStepSymplecticCor<false,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,velrhoppre,ar,ace,shiftposfs,sigmapre,rsigma,indirvel,dtm,dt,rhopzero,rhopoutmin,rhopoutmax,Float3(gravity),code,movxy,movz,velrhop,sigma);
      }else{        const bool shift=false;
        if(floating)KerComputeStepSymplecticCor<true ,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,velrhoppre,ar,ace,shiftposfs,sigmapre,rsigma,indirvel,dtm,dt,rhopzero,rhopoutmin,rhopoutmax,Float3(gravity),code,movxy,movz,velrhop,sigma);
        else        KerComputeStepSymplecticCor<false,shift,tinout> <<<sgrid,SPHBSIZE,0,stm>>> (np,npb,velrhoppre,ar,ace,shiftposfs,sigmapre,rsigma,indirvel,dtm,dt,rhopzero,rhopoutmin,rhopoutmax,Float3(gravity),code,movxy,movz,velrhop,sigma);
      }
    }
  }
}


}


