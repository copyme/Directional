// This file is part of Directional, a library for directional field processing.
// Copyright (C) 2017 Daniele Panozzo <daniele.panozzo@gmail.com>, Amir Vaxman <avaxman@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.


#ifndef DIRECTIONAL_POLYVECTOR_FIELD_H
#define DIRECTIONAL_POLYVECTOR_FIELD_H

#include <Eigen/Core>

namespace directional
{
  /**
   * Computes a polyvector on the entire mesh from given values at the prescribed indices.
   * If no constraints (hard/soft) are given the Fielder eigenvector field will be returned.
   *
   * @param[in]  vertices #V by 3 list of the vertex positions
   * @param[in]  faces #F by 3 list of the faces
   * @param[in]  hardConstrIDs list of constrained faces' IDs. The number of rows must be equal
   *             to the number of rows of hardConstrDir.
   * @param[in]  hardConstrDir a list of constrained directions at the faces given in hardConstrIndices.
   *             Should be given in either as a matrix of size: no. of constrained faces by N, i.e.,
   *             X1, Y1, Z1, X2, Y2, Z2, Xn, Yn, Zn, or as a matrix of size: number of constrained face
   *             by 3, i.e., a single direction per row, implying N-RoSy. The number of rows must be equal
   *             to the number of rows in hardConstrIndices.
   * @param[in]  softConstrIDs list of constrained faces' IDs. The number of rows must be equal
   *             to the number of rows in softConstrWeights and softConstrDir.
   * @param[in]  softConstrWeights weights for the soft constraints. It can be given either as a matrix of size: no. of
   *             constrained face times N, or as a matrix of size number of constrained faces times 1. The number of
   *             rows must be equal to the number of rows in softConstrID and softConstrDir.
   * @param[in]  softConstrDir a list of constrained directions at the faces given in hardConstrIndices.
   *             Should be given in either as a matrix of size: no. of constrained faces by N, i.e.,
   *             X1, Y1, Z1, X2, Y2, Z2, Xn, Yn, Zn, or as a matrix of size: number of constrained face
   *             by 3, i.e., a single direction per row, implying N-RoSy. The number of
   *             rows must be equal to the number of rows in softConstrID and softConstrWeights.
   * @param[in]  N the degree of the field
   * @param[out] polyVectorField the output interpolated field, in polyvector (complex polynomial) format.
   *             The size of the output is the number of faces times N.
   */
  IGL_INLINE void polyvector_field(const Eigen::MatrixXd & vertices,
                                   const Eigen::MatrixXi & faces,
                                   const Eigen::VectorXi & hardConstrIDs,
                                   const Eigen::MatrixXd & hardConstrDir,
                                   const Eigen::VectorXi & softConstrIDs,
                                   const Eigen::MatrixXd & softConstrWeights,
                                   const Eigen::MatrixXd & softConstrDir,
                                   unsigned int N,
                                   Eigen::MatrixXcd & polyVectorField);



  /**
   * Computes a polyvector on the entire mesh from given values at the prescribed indices.
   * If no constraints (hard/soft) are given the Fielder eigenvector field will be returned.
   *
   * @param[in]  vertices #V by 3 list of the vertex positions
   * @param[in]  B1 each row represent a vector of the local coordinate basis at the respective face identified
   *             by the row number. Each such vector is orthogonal to a respective vector in B2.
   * @param[in]  B2 each row represent a vector of the local coordinate basis at the respective face identified
   *             by the row number. Each such vector is orthogonal to a respective vector in B1.
   * @param[in]  EV a matrix of edges (vertex indices), its size has to be no. of edges times 2.
   *             Each row containts IDs of vertices which constitute the edge identified by the row.
   * @param[in]  EF a matrix of faces adjecent to edges, it size has to be no. of edges times 2.
   *             Each row contains IDs of faces adjacent to a given edge identified with a row.
   * @param[in]  hardConstrIDs list of constrained faces' IDs. The number of rows must be equal
   *             to the number of rows of hardConstrDir.
   * @param[in]  hardConstrDir a list of constrained directions at the faces given in hardConstrIndices.
   *             Should be given in either as a matrix of size: no. of constrained faces by N, i.e.,
   *             X1, Y1, Z1, X2, Y2, Z2, Xn, Yn, Zn, or as a matrix of size: number of constrained face
   *             by 3, i.e., a single direction per row, implying N-RoSy. The number of rows must be equal
   *             to the number of rows in hardConstrIndices.
   * @param[in]  softConstrIDs list of constrained faces' IDs. The number of rows must be equal
   *             to the number of rows in softConstrWeights and softConstrDir.
   * @param[in]  softConstrWeights weights for the soft constraints. It can be given either as a matrix of size: no. of
   *             constrained face times N, or as a matrix of size number of constrained faces times 1. The number of
   *             rows must be equal to the number of rows in softConstrID and softConstrDir.
   * @param[in]  softConstrDir a list of constrained directions at the faces given in hardConstrIndices.
   *             Should be given in either as a matrix of size: no. of constrained faces by N, i.e.,
   *             X1, Y1, Z1, X2, Y2, Z2, Xn, Yn, Zn, or as a matrix of size: number of constrained face
   *             by 3, i.e., a single direction per row, implying N-RoSy. The number of
   *             rows must be equal to the number of rows in softConstrID and softConstrWeights.
   * @param[in]  N the degree of the field
   * @param[out] polyVectorField the output interpolated field, in polyvector (complex polynomial) format.
   *             The size of the output is the number of faces times N.
   */
  IGL_INLINE void polyvector_field(const Eigen::MatrixXd & vertices,
                                   const Eigen::MatrixXd & B1,
                                   const Eigen::MatrixXd & B2,
                                   const Eigen::MatrixXi & EV,
                                   const Eigen::MatrixXi & EF,
                                   const Eigen::VectorXi & hardConstrIDs,
                                   const Eigen::MatrixXd & hardConstrDir,
                                   const Eigen::VectorXi & softConstrIDs,
                                   const Eigen::MatrixXd & softConstrWeights,
                                   const Eigen::MatrixXd & softConstrDir,
                                   unsigned int N,
                                   Eigen::MatrixXcd & polyVectorField);
}

#include "polyvector_field.cpp"

#endif
