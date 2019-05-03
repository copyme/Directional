// This file is part of Directional, a library for directional field processing.
// Copyright (C) 2018 Amir Vaxman <avaxman@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#ifndef DIRECTIONAL_PARAMETERIZE_H
#define DIRECTIONAL_PARAMETERIZE_H

#include <iostream>
#include <fstream>
#include <queue>
#include <vector>
#include <cmath>
#include <Eigen/Core>
#include <igl/igl_inline.h>
#include <igl/gaussian_curvature.h>
#include <igl/local_basis.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/edge_topology.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/matlab_format.h>
#include <igl/bounding_box_diagonal.h>
#include <directional/tree.h>
#include <directional/representative_to_raw.h>
#include <directional/principal_matching.h>
#include <directional/setup_parameterization.h>


namespace directional
{
  // we do not really need this kind of rounding but well, added
  void cubeRound(const Eigen::VectorXd & fullx, Eigen::VectorXd & roundedX)
  {
    roundedX.setZero();
    for(unsigned int i = 0; i + 1 < fullx.rows(); i += 2)
    {
      Eigen::Vector3d cube(fullx(i), -fullx(i+1), -fullx(i) + fullx(i + 1));
      Eigen::Vector3d cubeR(std::round( 0.5 * fullx(i)), std::round(0.5 * fullx(i + 1)), std::round(0.5 * (-fullx(i) + fullx(i + 1))));
      Eigen::Vector3d diff(std::abs(cubeR(0) - 0.5 * cube(0)), std::abs(cubeR(1) - 0.5 * cube(1)), std::abs(cubeR(2) - 0.5 * cube(2)));

      if(diff(0) > diff(1) && diff(0) > diff(2))
      {
        roundedX(i) = (cubeR(1) - cubeR(2));
        roundedX(i + 1) = cubeR(1);
      }
      else if (diff(1) > diff(2))
      {
        roundedX(i) = cubeR(0);
        roundedX(i + 1) = (cubeR(0) + cubeR(2));
      }
      else
      {
        roundedX(i) = cubeR(0);
        roundedX(i + 1) = cubeR(1);
      }
    }
  }

  // Creates a parameterization of (currently supported) (u,v, -u,-v) functions from a directional field by solving the Poisson equation, with custom edge weights
  // Input:
  //  wholeV:       #V x 3 vertex coordinates of the original mesh.
  //  wholeF:       #F x 3 face vertex indices of the original mesh.
  //  FE:           #F x 3 faces to edges indices.
  //  rawField:     #F by 3*N  The directional field, assumed to be ordered CCW, and in xyzxyz raw format. The degree is inferred by the size. (currently only supporting sign-symmetric 4-fields).
  //  lengthRatio   #edgeLength/bounding_box_diagonal of quad mesh (scaling the gradient).
  //  pd:           Parameterization data obtained from directional::setup_parameterization.
  //  cutV:         #cV x 3 vertices of the cut mesh.
  //  cutF:         #F x 3 faces of the cut mesh.
  //  roundIntegers;   which variables (from #V+#T) are rounded iteratively to double integers. for each "x" entry that means that the [4*x,4*x+4] entries of vt will be double integer.
  // Output:
  //  cutUV:        #cV x 2 (u,v) coordinates per cut vertex.
  IGL_INLINE void parameterize(const Eigen::MatrixXd& wholeV,
                               const Eigen::MatrixXi& wholeF,
                               const Eigen::MatrixXi& FE,
                               const Eigen::MatrixXd rawField,
                               const double lengthRatio,
                               const ParameterizationData& pd,
                               const Eigen::MatrixXd& cutV,
                               const Eigen::MatrixXi& cutF,
                               const bool roundIntegers,
                               Eigen::MatrixXd& cutUV)
  
  
  {
    using namespace Eigen;
    using namespace std;
    
    //assert (rawField.cols()==12 && "Currently only supporting N=4 parameterization");
    
    VectorXd edgeWeights = VectorXd::Constant(FE.maxCoeff() + 1, 1.0);
    double length = igl::bounding_box_diagonal(wholeV) * lengthRatio;
    
    //TODO: in vertex space, not corner...
    int N = rawField.cols() / 3;
    int numVars = pd.symmMat.cols();
    //constructing face differentials
    vector<Triplet<double> >  d0Triplets;
    vector<Triplet<double> > M1Triplets;
    VectorXd gamma(3 * N * wholeF.rows());
    for(int i = 0; i < cutF.rows(); i++)
    {
      for(int j = 0; j < 3; j++)
      {
        for(int k = 0; k < N; k++)
        {
          d0Triplets.emplace_back(3 * N * i + N * j + k, N * cutF(i, j) + k, -1.0);
          d0Triplets.emplace_back(3 * N * i + N * j + k, N * cutF(i, (j + 1) % 3) + k, 1.0);
          Vector3d edgeVector = (cutV.row(cutF(i, (j + 1) % 3)) - cutV.row(cutF(i, j))).transpose();
          gamma(3 * N * i + N * j + k) = (rawField.block(i, 3 * k, 1, 3) * edgeVector)(0, 0) / length;
          M1Triplets.emplace_back(3 * N * i + N * j + k, 3 * N * i + N * j + k, edgeWeights(FE(i, j)));
        }
      }
    }
    SparseMatrix<double> d0(3 * N * wholeF.rows(), N * cutV.rows());
    d0.setFromTriplets(d0Triplets.begin(), d0Triplets.end());
    SparseMatrix<double> M1(3 * N * wholeF.rows(), 3 * N * wholeF.rows());
    M1.setFromTriplets(M1Triplets.begin(), M1Triplets.end());
    SparseMatrix<double> d0T = d0.transpose();


    //the variables that should be fixed in the end
    VectorXi fixedMask(numVars);
    fixedMask.setZero();
    if(N == 6)
    {
      for(int i = 0; i < (int)std::round(N / 3); i++)
      fixedMask(i) = 1;  //first vertex is always (0,0)
    }
    else
    {
      for (int i = 0; i < (int)std::round(N / 2); i++)
        fixedMask(i) = 1;  //first vertex is always (0,0)
    }
    
    if(roundIntegers)
    {
      for(int i = 0; i < pd.integerVars.size(); i++)
        fixedMask(pd.integerVars(i)) = 1;
    }
    
    //the variables that were already fixed in the previous iteration
    VectorXi alreadyFixed(numVars);
    alreadyFixed.setZero();
    if(N == 6)
    {
      for(int i = 0; i < (int)std::round(N / 3); i++)
        alreadyFixed(i) = 1;  //first vertex is always (0,0)
    }
    else
    {
      for(int i = 0; i < (int)std::round(N / 2); i++)
        alreadyFixed(i) = 1;  //first vertex is always (0,0)
    }

    //the values for the fixed variables (size is as all variables)
    VectorXd fixedValues(numVars);
    fixedValues.setZero();
    
    SparseMatrix<double> Efull = d0 * pd.vertexTrans2CutMat * pd.symmMat;
    VectorXd x, xprev;

    // until then all the N depedencies should be resolved?

    //reducing constraintMat
    SparseQR<SparseMatrix<double>, COLAMDOrdering<int> > qrsolver;
    SparseMatrix<double> Cfull = pd.constraintMat * pd.symmMat;
    qrsolver.compute(Cfull.transpose());
    int CRank = qrsolver.rank();

    //creating sliced permutation matrix
    VectorXi PIndices = qrsolver.colsPermutation().indices();

    vector<Triplet<double> > CTriplets;
    for(int k = 0; k < Cfull.outerSize(); ++k)
    {
      for(SparseMatrix<double>::InnerIterator it(Cfull, k); it; ++it)
      {
        for(int j = 0; j < CRank; j++)
          if(it.row() == PIndices(j))
            CTriplets.emplace_back(j, it.col(), it.value());
      }
    }

    Cfull.resize(CRank, Cfull.cols());
    Cfull.setFromTriplets(CTriplets.begin(), CTriplets.end());
    SparseMatrix<double> var2AllMat;
    VectorXd fullx(numVars); fullx.setZero();
    VectorXd roundedX(numVars); fullx.setZero();
    for(int intIter = 0; intIter < fixedMask.sum(); intIter++)
    {
      //the non-fixed variables to all variables
      var2AllMat.resize(numVars, numVars - alreadyFixed.sum());
      int varCounter = 0;
      vector<Triplet<double> > var2AllTriplets;
      for(int i = 0; i < numVars; i++)
      {
        if (!alreadyFixed(i))
          var2AllTriplets.emplace_back(i, varCounter++, 1.0);
      }
      var2AllMat.setFromTriplets(var2AllTriplets.begin(), var2AllTriplets.end());
      
      SparseMatrix<double> Epart = Efull * var2AllMat;
      VectorXd torhs = -Efull * fixedValues;
      SparseMatrix<double> EtE = Epart.transpose() * M1 * Epart;
      SparseMatrix<double> Cpart = Cfull * var2AllMat;
      
      //reducing rank on Cpart
      qrsolver.compute(Cpart.transpose());
      int CpartRank = qrsolver.rank();
  
      //creating sliced permutation matrix
      VectorXi PIndices = qrsolver.colsPermutation().indices();

      vector<Triplet<double> > CPartTriplets;
      VectorXd bpart(CpartRank);
      for(int k = 0; k < Cpart.outerSize(); ++k)
      {
        for (SparseMatrix<double>::InnerIterator it(Cpart, k); it; ++it)
        {
          for (int j = 0; j < CpartRank; j++)
            if (it.row() == PIndices(j))
              CPartTriplets.emplace_back(j, it.col(), it.value());
        }
      }

      Cpart.resize(CpartRank, Cpart.cols());
      Cpart.setFromTriplets(CPartTriplets.begin(), CPartTriplets.end());
      SparseMatrix<double> A(EtE.rows()+ Cpart.rows(), EtE.rows() + Cpart.rows());
      
      vector<Triplet<double>> ATriplets;
      for(int k = 0; k < EtE.outerSize(); ++k)
      {
        for (SparseMatrix<double>::InnerIterator it(EtE, k); it; ++it)
          ATriplets.push_back(Triplet<double>(it.row(), it.col(), it.value()));
      }

      for(int k = 0; k < Cpart.outerSize(); ++k)
      {
        for(SparseMatrix<double>::InnerIterator it(Cpart, k); it; ++it)
        {
          ATriplets.emplace_back(it.row() + EtE.rows(), it.col(), it.value());
          ATriplets.emplace_back(it.col(), it.row() + EtE.rows(), it.value());
        }
      }
      
      A.setFromTriplets(ATriplets.begin(), ATriplets.end());
      
      VectorXd b = VectorXd::Zero(EtE.rows() + Cpart.rows());
      b.segment(0, EtE.rows())= Epart.transpose() * M1 * (gamma + torhs);
      VectorXd bfull = -Cfull * fixedValues;
      for(int k = 0; k < CpartRank; k++)
        bpart(k)=bfull(PIndices(k));
      b.segment(EtE.rows(), Cpart.rows()) = bpart;

      SparseLU<SparseMatrix<double> > lusolver;
      lusolver.compute(A);
      if(lusolver.info() != Success)
        throw std::runtime_error("LU decomposition failed!");
      x = lusolver.solve(b);

      fullx = var2AllMat * x.head(numVars - alreadyFixed.sum()) + fixedValues;

      cout << "(Cfull * fullx).lpNorm<Infinity>(): "<< (Cfull * fullx).lpNorm<Infinity>() << endl;
      cout << "Poisson error: " << (Efull * fullx - gamma).lpNorm<Infinity>() << endl;
      
      if((alreadyFixed - fixedMask).sum() == 0)
        break;
      
      double minIntDiff = 5000.0;
      int minIntDiffIndex = -1;

      cubeRound(fullx, roundedX);

      for(int i = 0; i < numVars; i++)
      {
        if((fixedMask(i)) && (!alreadyFixed(i)))
        {
          double currIntDiff = std::abs(0.5 * fullx(i) - roundedX(i));
          if(currIntDiff < minIntDiff)
          {
            minIntDiff = currIntDiff;
            minIntDiffIndex = i;
          }
        }
      }
      
      cout << "Integer variable: " << minIntDiffIndex << endl;
      cout << "Integer error: " << minIntDiff << endl;
      
      if(minIntDiffIndex != -1)
      {
        alreadyFixed(minIntDiffIndex) = 1;
        fixedValues(minIntDiffIndex) = roundedX(minIntDiffIndex) * 2.;
      }
      
      xprev.resize(x.rows() - 1);
      varCounter = 0;
      for(int i = 0; i < numVars; i++)
      {
        if (!alreadyFixed(i))
          xprev(varCounter++) = fullx(i);
      }
      xprev.tail(Cpart.rows()) = x.tail(Cpart.rows());
    }

    //the results are packets of N functions for each vertex, and need to be allocated for corners
    //cout<<"fullx: "<<fullx<<endl;
    VectorXd cutUVVec = pd.vertexTrans2CutMat * pd.symmMat * fullx;
    //cutUVW.conservativeResize(cutV.rows(),N/2);
    cutUV.conservativeResize(cutV.rows(), 2);

    Matrix2d c;
    c << sqrt(3)  / sqrt(3), -sqrt(3) / 2./ sqrt(3), 0 , -3 / 2.;
    std::cout << c << std::endl;

    for(int i = 0; i < cutV.rows(); i++)
      cutUV.row(i) << (c * cutUVVec.segment(N * i, N / 3)).transpose();

    //cout<<"symmMat*fullx: "<<symmMat*fullx<<endl;
    //cout<<"cutUVWVec: "<<cutUVWVec<<endl;
    //cout<<"cutUVW: "<<cutUVW<<endl;
    //if (N==4){
      //cutUV = cutUVW.block(0,0,cutUVW.rows(),2);
    /*}else {
      double yratio = 2.0/sqrt(3.0);
      RowVector3d UAxis; UAxis<<1, 1,0; UAxis.normalize();
      RowVector3d VAxis; VAxis<<-1, 1, 2; VAxis.normalize();
      VectorXd a = cutUVW.col(0)+cutUVW.col(1);
      VectorXd b = -cutUVW.col(0)+cutUVW.col(2);
      cutUV.col(0) = a-b/2.0;
      cutUV.col(1) = b*sqrt(3.0)/2.0*yratio;
      //cutUV = cutUVW.block(0,0,cutUVW.rows(),2);
    }*/
  }
}
  
#endif

  
