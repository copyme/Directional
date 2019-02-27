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
                                   Eigen::MatrixXcd& polyVectorField);


  IGL_INLINE void polyvector_field(const Eigen::MatrixXd & V,
                                   const Eigen::MatrixXi & F,
                                   const Eigen::VectorXi & bc,
                                   const Eigen::MatrixXd & b,
                                   const unsigned int N,
                                   Eigen::MatrixXcd& polyVectorField);
}

#include "polyvector_field.cpp"

#endif
