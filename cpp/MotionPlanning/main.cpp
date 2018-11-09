#include <iostream>
#include <fstream>
#include <vector>
#include <sys/stat.h>
#include <cstdio>
#include "utility.h"
#include "robot.h"
#include "Vector3D.h"
#include "test.h"
#include "CollisionChecker.h"
#include "GeneticOperators.h"
#include "RecordLog.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;

double duration[5];
vector<Individual> totalPOP1, totalPOP2;
void alpsGA(vector<Individual>[] ,vector<Individual>[]);

int main()
{
    // Create a folder for output files.
    mkdir("log", 0775);

    //test();

    // Create initial populations.
    pops1[0] = createInitialPop(0, N_POP);
    pops2[0] = createInitialPop(1, N_POP);

    // Evaluation
    evaluate(pops1[0], pops2);
    evaluate(pops2[0], pops1);

    totalPOP1 = pops1[0];
    totalPOP2 = pops2[0];

    //sort_pop(pops1[0]);
    //sort_pop(pops2[0]);

    // Log
    RecordLog(true);

    // Start co-evolution.
    for (gen=1; gen<=N_GEN; gen++){
        ofs_log << " ----- GENERATION: " << gen << " ----- " <<endl;

        if ((int)(gen/1)%2==0){
            alpsGA(pops1, pops2);

            //totalPOP1.clear();
            //for (int id=0; id<N_LAYERS; id++) totalPOP1.insert(totalPOP1.end(), pops1[id].begin(), pops1[id].end());
            //sort_pop(totalPOP1);

        } else{
            alpsGA(pops2, pops1);

            //totalPOP2.clear();
            //for (int id=0; id<N_LAYERS; id++) totalPOP2.insert(totalPOP2.end(), pops2[id].begin(), pops2[id].end());
            //sort_pop(totalPOP2);
        }

        ofs_log << "END" << endl; ofs_log << endl;
    }

    cout << endl;
    cout << " --- Average elapsed time --- " << endl;
    cout << "Crossover : " << duration[0]/N_GEN << endl;
    cout << "Mutation  : " << duration[1]/N_GEN << endl;
    cout << "Evaluation: " << duration[2]/N_GEN << endl;
    cout << "Selection : " << duration[3]/N_GEN << endl;
    cout << "Output    : " << duration[4]/N_GEN << endl;

    return 0;
}


void alpsGA(vector<Individual>pops[] ,vector<Individual>competitors[]){

    Timer tm;
    for (int id=N_LAYERS-1; id>=0; id--){

        ofs_log << "  LAYER: " << id << endl;

        if(pops[id].size()==0){
            ofs_log << "    No Population: " << endl;
            continue;
        }

        vector<Individual> elites = elitistSelection(pops[id], N_ELITES);
        vector<Individual> offspring = pops[id];

        // --------------------------------------
        // Crossover
        // --------------------------------------
        tm.restart();
        ofs_log << "    CROSSOVER" << endl;

        vector<Individual> partners;
        for (int i=0; i<=id; i++){
            partners.insert(partners.begin(), pops[i].begin(), pops[i].end());
        }

        crossover(offspring, partners);
        duration[0] += tm.elapsed();

        // --------------------------------------
        // Mutation
        // --------------------------------------
        tm.restart();
        ofs_log << "    MUTATION" << endl;
        mutation(offspring);
        duration[1] += tm.elapsed();

        // --------------------------------------
        //  Evaluation
        // --------------------------------------
        tm.restart();
        ofs_log << "    EVALUATION" << endl;

        offspring.insert(offspring.end(), elites.begin(), elites.end());
        evaluate(offspring, competitors);
        sort_pop(offspring);
        for(Individual &ind:offspring) ind.age += 1;
        pops[id] = offspring;

        duration[2] += tm.elapsed();
    }


    // --------------------------------------
    // Selection
    // --------------------------------------
    tm.restart();
    ofs_log << "  SELECTION" << endl;

    // --- Age-layered Selection
    agelayeredSelection(pops);

    // --- Create new population
    if (pops[0].size()==0){
        int robotID=pops[1][0].robotID;
        pops[0] = createInitialPop(robotID, N_POP);
        evaluate(pops[0], competitors);
        ofs_log << "    New population created. " << endl;
    }

    // --- Sort.
    for (int i=0; i<N_LAYERS; i++) sort_pop(pops[i]);

    duration[3] += tm.elapsed();

    // ---------------------------------------
    // Output
    // ---------------------------------------
    tm.restart();
    ofs_log << "  OUTPUT" << endl;
    RecordLog(true);
    duration[4] += tm.elapsed(); tm.restart();

}
