#include <Eigen/Dense>
#include <vector>
#include "robot.h"

using std::vector;
using namespace Eigen;

//vector<double>Lengths = {5.0, 5.0};
//vector<double>Base = {3.0, 3.0};


void Robot::set_values (vector<double> base,
                        vector<double> lengths,
                        vector<double> start,
                        vector<double> goal) {
    Base = base;
    Lengths = lengths;
    Start = start;
    Goal = goal;
}

vector<vector<double>> Robot::forward_kinematics(vector<double>qlist){

    int dim = qlist.size();

    Matrix3d rotmat[dim];
    Matrix3d transmat[dim];

    for (int i=0; i<dim; i++){
        //MatrixXd rotmat(3, 3);
        rotmat[i] << cos(qlist[i]), -sin(qlist[i]), 0.0,
                     sin(qlist[i]), cos(qlist[i]), 0.0,
                     0.0, 0.0, 1.0;
    }

    for (int i=0; i<dim; i++){
        // MatrixXd transmat(3, 3);
        transmat[i] << 1.0, 0.0, Lengths[i],
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0;
    }

    Matrix3d currentT;
    currentT << 1.0, 0.0, Base[0],
                0.0, 1.0, Base[1],
                0.0, 0.0, 1.0;

    vector<vector<double>> pts = {Base};
    MatrixXd mat(3, 1);
    mat << 0.0, 0.0, 1.0;
    for (int i=0; i<dim; i++){
        currentT = currentT * rotmat[i] * transmat[i];
        MatrixXd pt(3, 1); pt = currentT * mat;
        pts.insert(pts.end(), {pt(0, 0), pt(1, 0)});
    }

    return pts;

}
