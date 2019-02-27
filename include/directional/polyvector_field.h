// This file is part of Directional, a library for directional field processing.
// Copyright (C) 2017 Daniele Panozzo <daniele.panozzo@gmail.com>, Amir Vaxman <avaxman@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#ifndef DIRECTIONAL_POLYVECTOR_FIELD_H
#define DIRECTIONAL_POLYVECTOR_FIELD_H

#include <iterator>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/Polynomials>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/local_basis.h>
#include <igl/edge_topology.h>
#include <igl/speye.h>
#include <igl/eigs.h>

namespace directional
{
    class PolyVectorComputer
    {
    // Type definitions
    public:
      typedef Eigen::SparseMatrix<std::complex<double> > SparseMatrix;
      typedef Eigen::Triplet<std::complex<double> > Triplet;
      typedef std::vector<Triplet> TContainer;
      typedef TContainer::const_iterator TCConstIterator;
      typedef std::back_insert_iterator<TContainer> OutputIterator;

    private:
       Eigen::VectorXi constIndices;
       Eigen::VectorXcd constValues;

       Eigen::VectorXi softIndices;
       Eigen::VectorXcd softValues;
       Eigen::VectorXcd softWeights;

       Eigen::VectorXi full2var;

       unsigned int N;
       double mAlpha;

       const Eigen::MatrixXd & mV;
       const Eigen::MatrixXd & mB1;
       const Eigen::MatrixXd & mB2;
       const Eigen::VectorXi & mBc;
       const Eigen::MatrixXd & mB;

       const Eigen::VectorXi & mBcSoft;
       const Eigen::MatrixXd & mWSoft;
       const Eigen::MatrixXd & mBSoft;

       SparseMatrix & mAfull;
       SparseMatrix & mAVar;
       SparseMatrix mASoft;
       Eigen::SimplicialLDLT<SparseMatrix> & mSolver;

       void treatHardConstraints()
       {
         Eigen::MatrixXcd constValuesMat(mB.rows(), N);
         assert((mB.cols() == 3 * N) || (mB.cols() == 3));
         if(mB.cols() == 3)  //N-RoSy constraint
         {
           constValuesMat.setZero();
           for(unsigned int i = 0; i < mB.rows(); i++)
           {
             std::complex<double> bComplex(mB.row(i).dot(mB1.row(mBc(i))), mB.row(i).dot(mB2.row(mBc(i))));
             constValuesMat(i, 0) = std::pow(bComplex, N);
           }
         }
         else
         {
           for(unsigned int i = 0; i < mB.rows(); i++)
           {
             Eigen::RowVectorXcd poly, roots(N);
             for(unsigned int n = 0; n < N; n++)
             {
               Eigen::RowVector3d vec = mB.block(i, 3 * n, 1, 3);
               roots(n) = std::complex<double>(vec.dot(mB1.row(mBc(i))), vec.dot(mB2.row(mBc(i))));
             }
             roots_to_monicPolynomial(roots, poly);
             constValuesMat.row(i) << poly.head(N);
           }
         }
         constValues.resize(N * mB.size());
         constValues.setZero();
         constIndices.resize(N * mBc.size());
         constIndices.setZero();
         for(unsigned int n = 0; n < N; n++)
         {
           constIndices.segment(mBc.rows() * n, mBc.rows()) = mBc.array() + n * mB1.rows();
           constValues.segment(mB.rows() * n, mB.rows()) = constValuesMat.col(n);
         }
       }

      void treatSoftConstraints()
      {
        Eigen::MatrixXcd softValuesMat(mBSoft.rows(), N);
        softValuesMat.setZero();
        if(!((mBSoft.cols() == 3 * N) || (mBSoft.cols() == 3)) || !((mWSoft.cols() == N) || (mWSoft.cols() == 1)))
          throw std::runtime_error("directional::PolyVectorComputer:::treatSoftConstraints: Missing information!");

        if(mBSoft.cols() == 3)  //N-RoSy constraint
        {
          for(unsigned int i = 0; i < mBSoft.rows(); i++)
          {
            std::complex<double> bComplex(mBSoft.row(i).dot(mB1.row(mBcSoft(i))), mBSoft.row(i).dot(mB2.row(mBcSoft(i))));
            softValuesMat(i, 0) = std::pow(mWSoft(i, 0) * bComplex, N);
          }
        }
        else
        {
          for(unsigned int i = 0; i < mBSoft.rows(); i++)
          {
            Eigen::RowVectorXcd poly, roots(N);
            for(unsigned int n = 0; n < N; n++)
            {
              Eigen::RowVector3d vec = mBSoft.block(i, 3 * n, 1, 3);
              roots(n) = mWSoft(i, n) * std::complex<double>(vec.dot(mB1.row(mBcSoft(i))), vec.dot(mB2.row(mBcSoft(i))));
            }
            roots_to_monicPolynomial(roots, poly);
            softValuesMat.row(i) << poly.head(N);
          }
        }

        Eigen::MatrixXcd softWeightsMat(mWSoft.rows(), N);
        softWeightsMat.setZero();
        if(mWSoft.cols() == 1)  //N-RoSy constraint
        {
          for(unsigned int i = 0; i < mWSoft.rows(); i++)
            softWeightsMat(i, 0) = std::complex<double>(mWSoft(i, 0), 0);
        }
        else
        {
          for(unsigned int i = 0; i < mWSoft.rows(); i++)
            for(unsigned int n = 0; n < N; n++)
              softWeightsMat(i, n) = std::complex<double>(mWSoft(i, n), 0);
        }

        softValues.resize(N * mBSoft.size());
        softValues.setZero();
        softWeights.resize(N * mWSoft.rows());
        softWeights.setZero();
        softIndices.resize(N * mBcSoft.size());
        softIndices.setZero();
        for(unsigned int n = 0; n < N; n++)
        {
          softIndices.segment(mBcSoft.rows() * n, mBcSoft.rows()) = mBcSoft.array() + n * mB1.rows();
          softWeights.segment(mWSoft.rows() * n, mWSoft.rows()) = softWeightsMat.col(n);
          softValues.segment(mBSoft.rows() * n, mBSoft.rows()) = softValuesMat.col(n);
        }
       }

      void tagVariables()
      {
        //removing columns pertaining to constant indices
        Eigen::VectorXi varMask = Eigen::VectorXi::Constant(N * mB1.rows(), 1);
        for (unsigned int i = 0; i < constIndices.size(); i++)
          varMask(constIndices(i)) = 0;

        full2var = Eigen::VectorXi::Constant(N * mB1.rows(), -1);
        unsigned int varCounter = 0;
        for (unsigned int i = 0; i < N * mB1.rows(); i++)
          if (varMask(i))
            full2var(i) = varCounter++;
        assert(varCounter == N * (mB1.rows() - mBc.size()));
      }

      unsigned int buildFullEnergyMatrix(const Eigen::MatrixXi & EV, const Eigen::MatrixXi & EF, OutputIterator result)
      {
         unsigned int rowCounter = 0;
         // Build the sparse matrix, with an energy term for each edge and degree
         for(unsigned int n = 0; n < N; n++)
         {
           for(unsigned int i = 0; i < EF.rows(); i++)
           {
             if ((EF(i, 0) == -1) || (EF(i, 1) == -1))
               continue;  //boundary edge

             // Compute the complex representation of the common edge
             Eigen::RowVector3d e = mV.row(EV(i,1)) - mV.row(EV(i,0));
             Eigen::RowVector2d vef = Eigen::Vector2d(e.dot(mB1.row(EF(i,0))), e.dot(mB2.row(EF(i,0)))).normalized();
             std::complex<double> ef(vef(0), vef(1));
             Eigen::Vector2d veg = Eigen::Vector2d(e.dot(mB1.row(EF(i,1))), e.dot(mB2.row(EF(i,1)))).normalized();
             std::complex<double> eg(veg(0), veg(1));

             // Add the term conj(f)^n*ui - conj(g)^n*uj to the energy matrix
             // ensure the sign consistance over the soft constraints
             *result++ = Triplet(rowCounter, n * mB1.rows() + EF(i, 0),  std::pow(conj(ef), N - n));
             if((softIndices.array() == EF(i, 0)).any() || (softIndices.array() == EF(i, 1)).any())
                 *result++ = Triplet(rowCounter++, n * mB1.rows() + EF(i, 1), std::pow(conj(eg), N - n));
             else
                 *result++ = Triplet(rowCounter++, n * mB1.rows() + EF(i, 1), -1. * std::pow(conj(eg), N - n));
           }
         }
         return rowCounter;
      }

      void extractVarPartOfEnergy(TCConstIterator begin, TCConstIterator end, OutputIterator result)
      {
        for(auto it = begin; it != end; ++it)
          if(full2var(it->col()) != -1)
            *result++ = Triplet(it->row(), full2var(it->col()), it->value());
      }

      void buildSoftConstraintsEnergyMatrix(OutputIterator result)
      {
        Eigen::VectorXcd soft(N * mB1.rows(), 1);
        soft.setZero();
        for(size_t i = 0; i < softIndices.size(); i++)
          soft(softIndices(i)) = softWeights(i);

        for(unsigned int i = 0; i < full2var.rows(); i++)
          if(full2var(i) != -1 && soft(i) != 0.)
            *result++ = Triplet(full2var(i), full2var(i), soft(i));
      }

      void buildEnergyMatrics(const Eigen::MatrixXi & EV, const Eigen::MatrixXi & EF)
      {
        TContainer AfullTriplets;
        unsigned int rowCounter = buildFullEnergyMatrix(EV, EF, OutputIterator(AfullTriplets));
        mAfull.conservativeResize(rowCounter, N * mB1.rows());
        mAfull.setFromTriplets(AfullTriplets.cbegin(), AfullTriplets.cend());

        TContainer AVarTriplets;
        extractVarPartOfEnergy(AfullTriplets.cbegin(), AfullTriplets.cend(), OutputIterator(AVarTriplets));
        mAVar.conservativeResize(rowCounter, N * (mB1.rows() - mBc.size()));
        mAVar.setFromTriplets(AVarTriplets.cbegin(), AVarTriplets.cend());

        TContainer TSoft;
        buildSoftConstraintsEnergyMatrix(OutputIterator(TSoft));
        mASoft.resize(N * (mB1.rows() - mBc.size()), N * (mB1.rows() - mBc.size()));
        mASoft.setFromTriplets(TSoft.cbegin(), TSoft.cend());
      }

      void evalNoConstraints(Eigen::MatrixXcd &polyVectorField)
      {
        //extracting first eigenvector into the field
        //Have to use reals because libigl does not currently support complex eigs.
        Eigen::SparseMatrix<double> M;
        igl::speye(2 * mB1.rows(), 2 * mB1.rows(), M);
        //creating a matrix of only the N-rosy interpolation
        SparseMatrix AfullNRosy(int((double)mAfull.rows() / (double)N), int((double)mAfull.cols() / (double)N));

        TContainer AfullNRosyTriplets;
        for(int k = 0; k < mAfull.outerSize(); ++k)
          for(SparseMatrix::InnerIterator it(mAfull, k); it; ++it)
            if((it.row() < (double)mAfull.rows() / (double)N) && (it.col() < (double)mAfull.cols() / (double)N))
              AfullNRosyTriplets.emplace_back(it.row(), it.col(), it.value());

        AfullNRosy.setFromTriplets(AfullNRosyTriplets.begin(), AfullNRosyTriplets.end());

        SparseMatrix LComplex = AfullNRosy.adjoint() * AfullNRosy;
        Eigen::SparseMatrix<double> L(2 * mB1.rows(), 2 * mB1.rows());
        std::vector<Eigen::Triplet<double> > LTriplets;
        for (unsigned int k = 0; k < LComplex.outerSize(); ++k)
          for(SparseMatrix::InnerIterator it(LComplex, k); it; ++it)
          {
            LTriplets.emplace_back(it.row(), it.col(), it.value().real());
            LTriplets.emplace_back(it.row(), LComplex.cols()+it.col(), -it.value().imag());
            LTriplets.emplace_back(LComplex.rows()+it.row(), it.col(), it.value().imag());
            LTriplets.emplace_back(LComplex.rows()+it.row(), LComplex.cols()+it.col(), it.value().real());
          }
        L.setFromTriplets(LTriplets.begin(), LTriplets.end());
        Eigen::MatrixXd U;
        Eigen::VectorXd S;
        igl::eigs(L, M, 5, igl::EIGS_TYPE_SM, U, S);

        polyVectorField = Eigen::MatrixXcd::Constant(mB1.rows(), N, std::complex<double>(0., 0.));
        polyVectorField.col(0) = U.block(0, 0, (long int)((double)U.rows() / 2.), 1).cast<std::complex<double> >().array() * std::complex<double>(1., 0.) +
                                 U.block(int((double)U.rows() / 2.), 0, int((double)U.rows() / 2.), 1).cast<std::complex<double> >().array() * std::complex<double>(0., 1.);
       }

    public:
        // Inputs:
        //  V:      #V by 3 vertex coordinates.
        //  B1, B2:       #F by 3 matrices representing the local base of each face.
        //  bc:           The faces on which the polyvector is prescribed.
        //  b:            The directionals on the faces indicated by bc. Should be given in either #bc by N raw format X1,Y1,Z1,X2,Y2,Z2,Xn,Yn,Zn, or representative #bc by 3 format (single xyz), implying N-RoSy
        //  solver:       With prefactorized left-hand side
        //  Afull, AVar:  Left-hand side matrices (with and without constraints) of the system
        //  N:            The degree of the field.
       PolyVectorComputer(const Eigen::MatrixXd & V,
                          const Eigen::MatrixXd & B1,
                          const Eigen::MatrixXd & B2,
                          const Eigen::VectorXi & bc,
                          const Eigen::MatrixXd & b,
                          const Eigen::VectorXi & bcSoft,
                          const Eigen::MatrixXd & wSoft,
                          const Eigen::MatrixXd & bSoft,
                          Eigen::SimplicialLDLT<Eigen::SparseMatrix<std::complex<double> > > & solver,
                          Eigen::SparseMatrix<std::complex<double> > & Afull,
                          Eigen::SparseMatrix<std::complex<double> > & AVar,
                          unsigned int N
                          ) : mV(V), mB1(B1), mB2(B2), mBc(bc), mB(b), mBSoft(bSoft), mWSoft(wSoft), mBcSoft(bcSoft),
                              mAfull(Afull), mAVar(AVar), mSolver(solver)
       {
         if(mBc.size() != mB.rows())
           throw std::runtime_error("directional::PolyVectorComputer: The hard constraines data are inconsistant!");

         if((mBSoft.rows() + mBcSoft.rows() + mWSoft.rows()) / 3. != mBSoft.rows())
           throw std::runtime_error("directional::PolyVectorComputer: The soft constraines data are inconsistant!");

         this->N = N;
         if (mBcSoft.size() == 0)
           mAlpha = 0.;
         else
           mAlpha = .5;
       }

        // Precalculate the polyvector LDLt solvers. Must be recalculated whenever
        // bc changes or the mesh changes.
        // Inputs:
        //  EV:     #E by 2 matrix of edges (vertex indices)
        //  EF:     #E by 2 matrix of oriented adjacent faces
        IGL_INLINE void precompute(const Eigen::MatrixXi & EV,
                                   const Eigen::MatrixXi & EF)
        {
          if(mB.rows() > 0)
            treatHardConstraints();
          if(mBSoft.rows() > 0)
            treatSoftConstraints();
          tagVariables();
          buildEnergyMatrics(EV, EF);

          mSolver.compute(mAVar.adjoint() * mAVar + mASoft);
        }

        // Computes a polyvector on the entire mesh from given values at the prescribed indices.
        // polyvector_precompute must be called in advance, and "b" must be on the given "bc"
        // If no constraints are given the Fielder eigenvector field will be returned.
        // Outputs:
        //  polyVectorField: #F by N The output interpolated field, in polyvector (complex polynomial) format.
       void eval(Eigen::MatrixXcd & polyVectorField)
       {
         assert(mSolver.rows() != 0);
         if (mBc.size() == 0 && mWSoft.size() == 0)
         {
           evalNoConstraints(polyVectorField);
           return;
         }

         Eigen::VectorXcd torhs(N * mB1.rows(), 1);
         torhs.setZero();
         for(size_t i = 0; i < constIndices.size(); i++)
           torhs(constIndices(i)) = constValues(i);

         Eigen::VectorXcd torhsSoft(N * mB1.rows(), 1);
         torhsSoft.setZero();
         for(size_t i = 0; i < softIndices.size(); i++)
           torhsSoft(softIndices(i)) = softValues(i);

         Eigen::VectorXcd rhs = -mAVar.adjoint() * mAfull * (torhs + torhsSoft);
         Eigen::VectorXcd varFieldVector = mSolver.solve(rhs);
         if (mSolver.info() != Eigen::Success)
           throw std::runtime_error("directional::PolyVectorComputer::eval: Solving the system finished with a failure!");

         // plug the hard constraints to the result
         Eigen::VectorXcd polyVectorFieldVector(N * mB1.rows());
         for (size_t i = 0; i < constIndices.size(); i++)
           polyVectorFieldVector(constIndices(i)) = constValues(i);

         // extract non-hard constrained results
         for(unsigned int i = 0; i < N * mB1.rows(); i++)
           if(full2var(i) != -1)
             polyVectorFieldVector(i) = varFieldVector(full2var(i));

         //converting to matrix form
         polyVectorField.conservativeResize(mB1.rows(), N);
         for(unsigned int n = 0; n < N; n++)
           polyVectorField.col(n) = polyVectorFieldVector.segment(n * mB1.rows(), mB1.rows());
       }
    };


  // Computes a polyvector on the entire mesh from given values at the prescribed indices.
  // polyvector_precompute must be called in advance, and "b" must be on the given "bc"
  // If no constraints are given the Fielder eigenvector field will be returned.
  // Inputs:
  //  B1, B2:       #F by 3 matrices representing the local base of each face.
  //  bc:           The faces on which the polyvector is prescribed.
  //  b:            The directionals on the faces indicated by bc. Should be given in either
  //  #bc by N raw format X1,Y1,Z1,X2,Y2,Z2,Xn,Yn,Zn, or representative #bc by 3 format (single xyz), implying N-RoSy
  //  solver:       With prefactorized left-hand side
  //  Afull, AVar:  Left-hand side matrices (with and without constraints) of the system
  //  N:            The degree of the field.
  // Outputs:
  //  polyVectorField: #F by N The output interpolated field, in polyvector (complex polynomial) format.
  IGL_INLINE void polyvector_field(const Eigen::MatrixXd & V,
                                   const Eigen::MatrixXi & F,
                                   const Eigen::VectorXi & bc,
                                   const Eigen::MatrixXd & b,
                                   const Eigen::VectorXi & bcSoft,
                                   const Eigen::MatrixXd & wSoft,
                                   const Eigen::MatrixXd & bSoft,
                                   const unsigned int N,
                                   Eigen::MatrixXcd& polyVectorField)
  {
    Eigen::MatrixXi EV, xi, EF;
    igl::edge_topology(V, F, EV, xi, EF);
    Eigen::MatrixXd B1, B2, xd;
    igl::local_basis(V, F, B1, B2, xd);
    Eigen::SparseMatrix<std::complex<double> > Afull, AVar;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<std::complex<double> > > solver;
    PolyVectorComputer pvComputer(V, B1, B2, bc, b, bcSoft, wSoft, bSoft, solver, Afull, AVar, N);
    pvComputer.precompute(EV, EF);
    pvComputer.eval(polyVectorField);
  }


  IGL_INLINE void polyvector_field(const Eigen::MatrixXd & V,
                                   const Eigen::MatrixXi & F,
                                   const Eigen::VectorXi & bc,
                                   const Eigen::MatrixXd & b,
                                   const unsigned int N,
                                   Eigen::MatrixXcd& polyVectorField)
  {
    Eigen::MatrixXi EV, xi, EF;
    igl::edge_topology(V, F, EV, xi, EF);
    Eigen::MatrixXd B1, B2, xd;
    igl::local_basis(V, F, B1, B2, xd);
    Eigen::SparseMatrix<std::complex<double> > Afull, AVar;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<std::complex<double> > > solver;

    Eigen::VectorXi bcSoft;
    Eigen::MatrixXd wSoft;
    Eigen::MatrixXd bSoft;

    PolyVectorComputer pvComputer(V, B1, B2, bc, b, bcSoft, wSoft, bSoft, solver, Afull, AVar, N);
    pvComputer.precompute(EV, EF);
    pvComputer.eval(polyVectorField);
  }
}
#endif
