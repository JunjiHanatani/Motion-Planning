#ifndef ROBOT_H
#define ROBOT_H
#include <vector>
using std::vector;

class Robot {

  public:
    vector<double>Start;
    vector<double>Goal;
    vector<double>Lengths;
    vector<double>Base;

    vector< vector<double> >forward_kinematics(vector<double>);
    void set_values(vector<double>, vector<double>, vector<double>, vector<double>);
};

#endif
