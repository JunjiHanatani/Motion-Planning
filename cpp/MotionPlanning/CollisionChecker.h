#ifndef COLLISIONCHECKER_H
#define COLLISIONCHECKER_H

#include <vector>
using std::vector;

bool isCollisionRoughCheck(vector<vector<double>>, vector<vector<double>>, double);
bool isCollisionLineLine(vector<vector<double>>, vector<vector<double>>);
bool isCollisionLineCircle(vector<vector<double>>, vector<double>, double);

#endif
