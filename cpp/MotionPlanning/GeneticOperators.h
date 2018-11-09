#ifndef GENETICOPERATORS_H
#define GENETICOPERATORS_H

#include <functional>
#include <vector>
#include "robot.h"

using std::vector;

struct Individual{
    vector<vector<double>> path;
    double fitness;
    double subfitness;
    double distance;
    int collision;
    double diversity;
    int age;
    int robotID;
    vector<int> collision_points;

    Individual():
        path(),
        fitness(),
        subfitness(),
        distance(),
        collision(),
        diversity(),
        age(1),
        robotID(-1),
        collision_points({-1,-1})
        {}
};

extern int N_POP, N_PTS, N_JOINT, N_GEN, N_ELITES, N_TOURNAMENT, N_LAYERS;
extern vector<Individual> pops1[5];
extern vector<Individual> pops2[5];
extern int gen;

// Create population
vector<Individual> createInitialPop(int, int);
vector<vector<double>> createPath(vector<double>, vector<double>);

// Evaluation
double calcPathLength(vector<vector<double>> &);
double calcTravelLength(Individual&);
int calcCollision(Individual&, Individual&);
void calcDiversity(vector<Individual>&);
double calcSubFitness(Individual&, vector<Individual>&);
void evaluate(vector<Individual>&, vector<Individual>[]);
void sort_pop(vector<Individual>&);

// Selection
vector<Individual> tournamentSelection(vector<Individual> const &, int);
vector<Individual> rouletteSelection(vector<Individual> &);
vector<Individual> elitistSelection(vector<Individual> const &, int);
void agelayeredSelection(vector<Individual>[]);
vector<Individual> overageSelection(vector<Individual>&, int);

// Mutation
void mutation(vector<Individual> &);
void mutNormal(Individual&);
void mutHillclimb(Individual&);

// Crossover
void crossover(vector<Individual> &, const vector<Individual> &);
vector<Individual> oneptcx(Individual &, const vector<Individual> &);
vector<Individual> oneptcx2(Individual &, const vector<Individual> &);

// Post process
void adjust_num_pts(vector<Individual> &);
vector<vector<double>> ferguson_spline(vector<vector<double>>, int);

#endif

