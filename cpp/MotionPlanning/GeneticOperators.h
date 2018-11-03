#ifndef GENETICOPERATORS_H
#define GENETICOPERATORS_H

#include <functional>
#include <vector>
#include "robot.h"

using std::vector;

struct Individual{
    vector<vector<double>> path;
    double fitness;
    double distance;
    int collision;
    double diversity;
    int age;
    int robotID;
};

extern int N_POP, N_PTS, N_JOINT, N_GEN, N_ELITES, N_TOURNAMENT, N_LAYERS;
extern Robot robot[2];
extern vector<Individual> pops1[5];
extern vector<Individual> pops2[5];
//extern vector<Individual> pop2;
extern int gen;

vector<Individual> createInitialPop(vector<double>, vector<double>, int);
vector<vector<double>> createPath(vector<double>, vector<double>);
double calcPathLength(vector<vector<double>>);
int calcCollision(Individual, Individual);
void evaluate(vector<Individual>&, const vector<Individual>);
vector<vector<double>> ferguson_spline(vector<vector<double>>, int);
void sort_pop(vector<Individual>&);
vector<Individual> tournamentSelection(vector<Individual> const &);
vector<Individual> rouletteSelection(vector<Individual> &);
vector<Individual> elitistSelection(vector<Individual> const &);
void mutation(std::function<Individual(Individual&)>, vector<Individual> &);
Individual mutNormal(Individual);
void crossover(std::function< vector<Individual>(Individual&, vector<Individual>&) >, vector<Individual> &);
vector<Individual> oneptcx(Individual &, vector<Individual> &);
void adjust_num_pts(vector<Individual> &);
void calcDiversity(vector<Individual>&);
vector<Individual> overageSelection(vector<Individual>&);

#endif

