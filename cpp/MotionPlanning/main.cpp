#include <iostream>
#include <fstream>
#include <vector>
#include <sys/stat.h>
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

vector<double> start1 = {-2.0, -0.6};
vector<double> goal1 = {3.5, 0.5};
vector<double> base1 = {3.0, 3.0};
vector<double> lengths1 = {5.0, 5.0};
vector<double> start2 = {1.2, -0.6};
vector<double> goal2 = {0.9, -0.8};
vector<double> base2 = {-3.0, -3.0};
vector<double> lengths2 = {5.0, 5.0};

int main()
{
    // Create a folder for output files.
    mkdir("log", 0775);

    vector<Individual> elites(N_ELITES);
    vector<Individual> offspring;
    double duration[5];

    // Set robot parameter.
    robot[0].set_values(base1, lengths1, start1, goal1);
    robot[1].set_values(base2, lengths2, start2, goal2);

    // Create initial populations.
    pops1[0] = createInitialPop(robot[0].Start, robot[0].Goal, 0);
    pops2[0] = createInitialPop(robot[1].Start, robot[1].Goal, 1);

    // Evaluation
    evaluate(pops1[0], pops2[0]);
    //evaluate(pop2, pop1);

    sort_pop(pops1[0]);
    //sort_pop(pop2);

    // Log
    RecordLog(true);

    Timer tm;
    for (gen=1; gen<=N_GEN; gen++){

        for (int id=N_LAYERS-1; id>=0; id--){

            if (id==0 && (gen % AGE_GAP==0)){
                pops1[id] = createInitialPop(robot[0].Start, robot[0].Goal, 0);
            }

            if(pops1[id].size()==0)continue;

            offspring = pops1[id];

            // --------------------------------------
            // Crossover
            // --------------------------------------
            crossover(oneptcx, offspring);
            duration[0] += tm.elapsed(); tm.restart();

            // --------------------------------------
            // Mutation
            // --------------------------------------
            mutation(mutNormal, offspring);
            duration[1] += tm.elapsed(); tm.restart();

            // --------------------------------------
            //  Evaluation
            // --------------------------------------
            evaluate(offspring, pops2[id]);
            sort_pop(offspring);
            duration[2] += tm.elapsed(); tm.restart();

            // --------------------------------------
            // Selection
            // --------------------------------------
            // For overages
            if (id != N_LAYERS-1){
                overages = overageSelection(offspring);     // Select and store overages. Also, remove overages from offspring.
                pops1[id+1].erase(pops1[id+1].end()-overages.size(), pops1[id+1].end());
                pops1[id+1].insert(pops1[id+1].end(), overages.begin(), overages.end());
            }
            // For current layer
            elites = elitistSelection(offspring);       // Elitist.
            offspring = tournamentSelection(offspring); // Selection. Or "rouletteSelection(pop1);" can be used instead.
            offspring.insert(offspring.end(), elites.begin(), elites.end());
            pops1[id] = offspring;

            duration[3] += tm.elapsed(); tm.restart();
        }

        // --- Output
        RecordLog(true);
        duration[4] += tm.elapsed(); tm.restart();
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
