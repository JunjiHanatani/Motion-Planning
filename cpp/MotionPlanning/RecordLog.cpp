#include <iostream>
#include <fstream>
#include "RecordLog.h"
#include "GeneticOperators.h"

int const FREQ_OUT = 10;
using std::cout;
using std::endl;
using std::to_string;

std::ofstream ofs_log("./log/log.csv");
std::ofstream ofs_best("./log/best_fitness.csv");

void RecordLog(vector<Individual>pops1[], vector<Individual>pops2[], bool verbose){

    vector<Individual> totalPOP1, totalPOP2;
    for (int i=0; i<N_LAYERS; i++){
        totalPOP1.insert(totalPOP1.end(), pops1[i].begin(), pops1[i].end());
        totalPOP2.insert(totalPOP2.end(), pops2[i].begin(), pops2[i].end());
    }
    sort_pop(totalPOP1);
    sort_pop(totalPOP2);

    // Adjust the size of the population.
    int num1 = totalPOP1.size();
    int num2 = totalPOP2.size();
    int num = num1;

    vector<vector<double>> empty_path(N_PTS, vector<double>(2));
    Individual ind_empty;
    ind_empty.path = empty_path;

    if (num1 > num2){
        for (int i=0; i<num1-num2; i++) totalPOP2.push_back(ind_empty);
        num = num1;
    }else if(num2 > num1){
        for (int i=0; i<num2-num1; i++) totalPOP1.push_back(ind_empty);
        num = num2;
    }

    // Output
    if (gen%FREQ_OUT==0){

        // Output File
        std::ofstream ofs_path("./log/path_" + to_string(gen) + ".csv");
        std::ofstream ofs_config("./log/config_" + to_string(gen) + ".csv");
        std::ofstream ofs_fitness("./log/fitness_" + to_string(gen) + ".csv");
        // Path in C-space
        ofs_path << "robo1_q0" << "," << "robo1_q1" << ","
                 << "robo2_q0" << "," << "robo2_q1" << endl;
        for (int i=0; i<num; i++){
            for (int j=0; j<N_PTS; j++){
                ofs_path << totalPOP1[i].path[j][0] << "," << totalPOP1[i].path[j][1] << ","
                         << totalPOP2[i].path[j][0] << "," << totalPOP2[i].path[j][1] << endl;
            }
        }

        ofs_log << "    Path Output done." << endl;
        /*
        // Configuration
        int pop_index = 0;
        int DIV = 10;
        vector<vector<double>> path1 = ferguson_spline(totalPOP1[pop_index].path, DIV);
        vector<vector<double>> path2 = ferguson_spline(totalPOP2[pop_index].path, DIV);

        ofs_config << "robo1_x" << "," <<  "robo1_y" << ","
                   << "robo2_x" << "," <<  "robo2_y" << endl;

        for (int i=0; i<DIV*(N_PTS-1)+1; i++){
            vector<vector<double>> config1 = robot[0].forward_kinematics(path1[i]);
            vector<vector<double>> config2 = robot[1].forward_kinematics(path2[i]);

            for (int j=0; j<N_JOINT+1; j++){
                ofs_config << config1[j][0] << "," << config1[j][1] << ","
                           << config2[j][0] << "," << config2[j][1] << endl;
            }
        }
        ofs_log << "    Configuration Output done." << endl;
        */
        // Fitness
        ofs_fitness << "fitness1" << "," << "subfitness1" << "," << "distance1" << ","
                    << "collision1" << "," << "diversity1" << "," << "age1" << ","
                    //<< "collision_points_begin1" << "," << "collision_points_end1" << ","
                    << "fitness2" << "," << "subfitness2" << "," << "distance2" << ","
                    << "collision2" << "," << "diversity2" << "," << "age2" << ","
                    //<< "collision_points_begin2" << "," << "collision_points_end2"
                    <<endl;

        for (int i=0; i<num; i++){
            ofs_fitness << totalPOP1[i].fitness << "," << totalPOP1[i].subfitness << "," << totalPOP1[i].distance << ","
                        << totalPOP1[i].collision << "," << totalPOP1[i].diversity << "," << totalPOP1[i].age << ","
                        //<< totalPOP1[i].collision_points[0] << "," << totalPOP1[i].collision_points[1] << ","
                        << totalPOP2[i].fitness << "," << totalPOP2[i].subfitness << "," << totalPOP2[i].distance << ","
                        << totalPOP2[i].collision << "," << totalPOP2[i].diversity << "," << totalPOP2[i].age << ","
                        //<< totalPOP2[i].collision_points[0] << "," << totalPOP2[i].collision_points[1]
                        << endl;
        }
        ofs_log << "    Fitness Output done." << endl;

    }

    // Header
    if (gen==0){
        ofs_best << "generation"<<",";
        for (int i=0;i<N_LAYERS;i++) ofs_best << "LAYER1-" + to_string(i) <<",";
        for (int i=0;i<N_LAYERS;i++) ofs_best << "LAYER2-" + to_string(i) <<",";
        ofs_best << endl;
    }
    ofs_best << gen << ",";
    for (int i=0; i<N_LAYERS; i++){
        if (pops1[i].size()!=0){
            ofs_best << pops1[i][0].fitness << ",";
        }else{
            ofs_best << ",";
        }
    }
    for (int i=0; i<N_LAYERS; i++){
        if (pops2[i].size()!=0){
            ofs_best << pops2[i][0].fitness << ",";
        }else{
            ofs_best << ",";
        }
    }
    ofs_best << endl;

    if (verbose) cout << "Generation: "  << gen
                      << "  Fitness1: " << totalPOP1[0].fitness
                      << "  Fitness2: " << totalPOP2[0].fitness
                      << endl;
}

void config_check(Individual ind, int n){
    // Configuration
    std::ofstream ofs_config_check("./log/config_check_" + to_string(n) + ".csv");
    vector<vector<double>> path = ind.path;
    int robotID = ind.robotID;

    ofs_config_check << "robo1_x" << "," <<  "robo1_y" << endl;
    int num = path.size();
    for (int i=0; i<num; i++){
        vector<vector<double>> config = robot[robotID].forward_kinematics(path[i]);
        for (int j=0; j<N_JOINT+1; j++){
            ofs_config_check << config[j][0] << "," << config[j][1] << endl;
        }
    }
}

void path_check(Individual ind, std::string name){
    // Configuration
    std::ofstream ofs_path_check("./log/path_check_" + name + ".csv");
    vector<vector<double>> path = ind.path;

    ofs_path_check << "robo1_q0" << "," <<  "robo1_q1" << endl;
    int num = path.size();
    for (int i=0; i<num; i++){
        ofs_path_check << path[i][0] << "," << path[i][1] << endl;
    }

}
