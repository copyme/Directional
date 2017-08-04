// Copyright (C) 2017 Amir Vaxman <avaxman@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#ifndef REPRESENTATIVE_TO_RAW_H
#define REPRESENTATIVE_TO_RAW_H
#include <igl/per_face_normals.h>
#include <igl/igl_inline.h>
#include <igl/PI.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>


namespace directional
{
    
    // Computes the raw vector field given a set of representative vectors.
    // This version accepts the face-based normals as input, instead of (V,F).
    // Input:
    //  normals:        #F by 3 coordinates of the normals of each face.
    //  representative: #F by 3 coordinates of representative vectors per face.
    //  N:              the degree of the field.
    // Output:
    //  raw:            #F by 3*N matrix with all N explicit vectors of each directional. Each row is arranged xyzxyzxyz of vectors in counterclockwise order.
    IGL_INLINE void representative_to_raw(const Eigen::MatrixXd& normals,
                                          const Eigen::MatrixXd& representative,
                                          const int N,
                                          Eigen::MatrixXd& raw)
    {
        raw.conservativeResize(representative.rows(), 3 * N);
        
        for (int i = 0; i < representative.rows(); i++)
        {
            raw.block<1, 3>(i, 0) << representative.row(i);
            
            Eigen::MatrixXd rot = Eigen::AngleAxisd((2.0*igl::PI)/(double)N, normals.row(i)).toRotationMatrix().transpose();
            for (int j = 1; j < N; j++)
                raw.block<1, 3>(i, j * 3) << raw.block<1, 3>(i, (j - 1) * 3)*rot;
            
        }
        
    }
    
    // Computes the raw vector field given a set of representative vectors.
    // Input:
    //  V:                  #V X 3 vertex coordinates.
    //  F:                  #F by 3 face vertex indices.
    //  representative:     #F by 3 coordinates of representative vectors.
    //  N:                  the degree of the field.
    // Output:
    //  raw:                #F by 3*N matrix with all N explicit normalized vectors of each directional. Each row is arranged xyzxyzxyz of vectors in counterclockwise order
    IGL_INLINE void representative_to_raw(const Eigen::MatrixXd& V,
                                          const Eigen::MatrixXi& F,
                                          const Eigen::MatrixXd& representative,
                                          const int N,
                                          Eigen::MatrixXd& raw)
    {
        Eigen::MatrixXd normals;
        igl::per_face_normals(V, F, normals);
        representative_to_raw(normals, representative, N, raw);
    }

}

#endif
