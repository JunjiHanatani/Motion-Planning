#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <stdlib.h>
#include "utility.h"
#include "robot.h"
#include "Vector3D.h"
#include "CollisionChecker.h"
#include "GeneticOperators.h"
#include "test.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;
using namespace Eigen;

void test(void){

vector<vector<double>>path1 =
{{-2.00000, 	-0.600000},
{-2.26915, 	-0.432140},
{-2.56195, 	-0.259422},
{-2.90279, 	-0.060084},
{-2.90052, 	-0.049509},
{-2.83699, 	0.284488 },
{-2.78319, 	0.500000 }};

vector<vector<double>>path2 =
{{	1.20000 ,	-0.600000},
{	1.69189 ,	-0.907173},
{	1.99875 ,	-1.124810},
{	2.20551 ,	-1.265780},
{	1.74359 ,	-1.146160},
{ 1.01995, 	-0.838040},
{ 0.90000, 	-0.800000}};


Individual ind1; Individual ind2;
ind1.path=path1; ind2.path=path2;
ind1.robotID=0; ind2.robotID=1;

double coll = calcCollision(ind1, ind2);
cout << coll << endl;
cout << ind1.collision_points[0] << ind1.collision_points[1] << endl;
exit(0);

}
