#include <Eigen/Dense>
#include <vector>
#include "robot.h"

using std::vector;
using namespace Eigen;

vector<double> start1 = {-2.0, -0.6};
vector<double> goal1 = {3.5, 0.5};
vector<double> base1 = {5.0, 5.0};
vector<double> lengths1 = {5.0, 5.0};
vector<double> start2 = {1.2, -0.6};
vector<double> goal2 = {0.9, -0.8};
vector<double> base2 = {-5.0, -5.0};
vector<double> lengths2 = {5.0, 5.0};

// Set robot parameter.
Robot robot[2] =
{
    {base1, lengths1, start1, goal1},
    {base2, lengths2, start2, goal2}
};

//robot[0].set_values(base1, lengths1, start1, goal1);
//robot[1].set_values(base2, lengths2, start2, goal2);
/*
void Robot::set_values (vector<double> base,
                        vector<double> lengths,
                        vector<double> start,
                        vector<double> goal) {
    Base = base;
    Lengths = lengths;
    Start = start;
    Goal = goal;
}
*/

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
