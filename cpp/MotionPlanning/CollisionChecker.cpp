#include "CollisionChecker.h"
#include "Vector3D.h"
#include <iostream>
#include <vector>
#include <math.h>
using std::cout;
using std::endl;
/*
Reference:
http://www5d.biglobe.ne.jp/~tomoya03/shtml/algorithm/Intersection.htm
http://mslabo.sakura.ne.jp/WordPress/make/processing
*/

bool isCollisionRoughCheck(vector<vector<double>>line1, vector<vector<double>>line2, double margin){

    // Rough check in x-axis
    if (line1[0][0] >= line1[1][0]){
        if ((line1[0][0] + margin < line2[0][0] - margin && line1[0][0] + margin < line2[1][0] - margin) ||
            (line1[1][0] - margin > line2[0][0] + margin && line1[1][0] - margin > line2[1][0] + margin )){
            return false;
        }
    }else{
        if ((line1[1][0] + margin < line2[0][0] - margin && line1[1][0] + margin < line2[1][0] - margin) ||
            (line1[0][0] - margin > line2[0][0] + margin && line1[0][0] - margin > line2[1][0] + margin)){
            return false;
        }
    }

    // Rough check in y-axis
    if (line1[0][1] >= line1[1][1]){
        if ((line1[0][1] + margin < line2[0][1] - margin && line1[0][1] + margin < line2[1][1] - margin) ||
            (line1[1][1] - margin > line2[0][1] + margin && line1[1][1] - margin > line2[1][1] + margin )){
            return false;
        }
    }else{
        if ((line1[1][1] + margin < line2[0][1] - margin && line1[1][1] + margin < line2[1][1] - margin) ||
            (line1[0][1] - margin > line2[0][1] + margin && line1[0][1] - margin > line2[1][1] + margin)){
            return false;
        }
    }

    return true;
}

bool isCollisionLineLine(vector<vector<double>>line1, vector<vector<double>>line2){

    // Check intersection
    if (((line1[0][0] - line1[1][0]) * (line2[0][1] - line1[0][1]) + (line1[0][1] - line1[1][1]) * (line1[0][0] - line2[0][0])) *
        ((line1[0][0] - line1[1][0]) * (line2[1][1] - line1[0][1]) + (line1[0][1] - line1[1][1]) * (line1[0][0] - line2[1][0])) > 0.0){
        return false;
    }

    if (((line2[0][0] - line2[1][0]) * (line1[0][1] - line2[0][1]) + (line2[0][1] - line2[1][1]) * (line2[0][0] - line1[0][0])) *
        ((line2[0][0] - line2[1][0]) * (line1[1][1] - line2[0][1]) + (line2[0][1] - line2[1][1]) * (line2[0][0] - line1[1][0])) > 0.0){
        return false;
    }

    return true;

}

bool isCollisionLineCircle(vector<vector<double>>line, vector<double>pt, double r){

    vector<double> AB = sub(line[1], line[0]);
    vector<double> AP = sub(pt, line[0]);

    double lenAB = calcDistance(line[1], line[0]);
    vector<double> normAB = calcUnitVector(AB);
    double lenAX = dotProduct(normAB, AP);

    if (lenAX < 0){
        return false;
    }else if(lenAB < lenAX){
        return false;
    }else{
        double dist = fabs(normAB[0] * AP[1] - normAB[1] * AP[0]);
        if (dist>r) return false;
    }

    return true;
}
