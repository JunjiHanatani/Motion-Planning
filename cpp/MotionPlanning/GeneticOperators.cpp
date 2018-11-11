#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include "utility.h"
#include "GeneticOperators.h"
#include "Vector3D.h"
#include "CollisionChecker.h"
#include "RecordLog.h"

using std::vector;
using std::cout;
using std::endl;;
using namespace Eigen;

int gen = 0;
int N_GEN = 3000;
int N_POP = 50;
int N_SPLINE = 3;
int N_FINE_SPLINE = 10;
int N_PTS = N_SPLINE * 2 + 1;
int N_JOINT = 2;
int N_ELITES = 1;
int N_TOURNAMENT = 3;
int N_LAYERS = 5;
int MAX_AGE[5] = {3, 6, 12, 24, 48};
double P_MUT = 0.7;
double RATE_OVERAGE = 0.2;
double COLLISION_WEIGHT = 10.0;
int COLLISION_CHECK_DIV = 10;
double COLLISION_MARGIN = 0.5;

// -----------------------------------------------
// CREATE INITIAL POPULATION
// -----------------------------------------------

vector<Individual> createInitialPop(int robotID, int num){

    vector<Individual> pop(num);

    //Create population
    for (int i=0; i<num; i++){
        Individual ind;
        ind.path = createPath(robot[robotID].Start, robot[robotID].Goal);
        ind.robotID = robotID;
        pop[i] = ind;
    }

    return pop;
}

vector<vector<double>> createPath(vector<double>start, vector<double>goal){
    vector<double> mid_pt = {get_rand_range_dbl(-2.0*PI, 2.0*PI),
                             get_rand_range_dbl(-2.0*PI, 2.0*PI)};

    vector<vector<double>>goals;
    for (int i=-1; i<=1; i++){
        for (int j=-1; j<=1; j++){
            double add_pt0 = goal[0] + 2.0*PI*i;
            double add_pt1 = goal[1] + 2.0*PI*j;

            if (-2.0*PI < add_pt0 && add_pt0< 2.0*PI &&
                -2.0*PI < add_pt1 && add_pt1< 2.0*PI){
                vector<vector<double>> add_goal = {{add_pt0, add_pt1}};
                goals.insert(goals.end(), add_goal.begin(), add_goal.end());
            }
        }
    }

    vector<double> nearest_goal;
    double min_dist = 100.0;
    for (vector<double>goal:goals){
        vector<double>vec = sub(mid_pt, goal);
        double dist = calcNorm(vec);
        if (dist < min_dist){
            min_dist = dist;
            nearest_goal = goal;
        }
    }

    vector<vector<double>>path = {start, mid_pt, nearest_goal};
    vector<vector<double>>spline_path = ferguson_spline(path, N_SPLINE);
    return spline_path;
}


// -----------------------------------------------
// EVALUATION
// -----------------------------------------------

void evaluate(vector<Individual>&pop, vector<Individual> competitors[]){
    int num = pop.size();
    vector<Individual> best_inds;

    for (int i=0; i<N_LAYERS; i++) {
        if (competitors[i].size()!=0) best_inds.push_back(competitors[i][0]);
    };

    // --- Calculate diversity.
    //calcDiversity(pop);

    for (int i=0; i<num; i++) {

        // --- Calculate traveling distance (Select c-space or Cartesian space)
        //pop1[i].distance = calcPathLength(pop[i].path);   // C-space
        pop[i].distance = calcTravelLength(pop[i]);         // Cartesian space

        pop[i].fitness = 1e6;
        for(Individual best_ind: best_inds){
            int best_collision = calcCollision(pop[i], best_ind);
            double best_distance = std::max(pop[i].distance, best_ind.distance);
            double best_fitness = best_distance + best_collision * COLLISION_WEIGHT;

            if (best_fitness < pop[i].fitness){
                pop[i].collision = best_collision;
                pop[i].fitness = best_fitness;
            }
        }

        // --- Calculate subjective fitness.
        //pop[i].subfitness = calcSubFitness(pop[i], competitors);
    }
}

double calcSubFitness(Individual &ind, vector<Individual>&competitors){
    double subFitness=0.0;
    int num=competitors.size();

    for (int i=0; i<num; i++){
        if (ind.fitness < competitors[i].fitness){
            subFitness += 1.0;
        }
    }
    subFitness = (double)subFitness / num;
    return subFitness;
}

double calcPathLength(vector<vector<double>> &path){
    double distance=0.0;
    int num=path.size();

    for (int i=0; i<num-1; i++){
         distance += calcDistance(path[i], path[i+1]);
    }

    return distance;
}

double calcTravelLength(Individual &ind){
    vector<vector<double>> travel;
    double distance=0.0;
    vector<vector<double>> path = ferguson_spline(ind.path, N_FINE_SPLINE);

    int num=path.size();
    for (int i=0; i<num; i++){
        vector<vector<double>> config = robot[ind.robotID].forward_kinematics(path[i]);
        travel.push_back(config.back());
        if (i>0){
            distance += calcDistance(travel[i-1], travel[i]);
        }
    }
    return distance;
}

int calcCollision(Individual &ind1, Individual &ind2){
    vector<vector<double>> spline1 = ferguson_spline(ind1.path, COLLISION_CHECK_DIV);
    vector<vector<double>> spline2 = ferguson_spline(ind2.path, COLLISION_CHECK_DIV);
    vector<double> cp;
    vector<vector<double>> config1, config2;
    vector<vector<double>> line1, line2;
    vector<double> pt1, pt2;
    int num = spline1.size();
    int collision_counter=0;

    for (int i=0; i<num; i++){
        config1 = robot[ind1.robotID].forward_kinematics(spline1[i]);
        config2 = robot[ind2.robotID].forward_kinematics(spline2[i]);

        // Line pair.
        for (int j=0; j<N_JOINT; j++){
            for (int k=0; k<N_JOINT; k++){

                line1={config1[j], config1[j+1]};
                line2={config2[k], config2[k+1]};

                // If rough check is true.
                if (isCollisionRoughCheck(line1, line2, COLLISION_MARGIN)){

                    // Line-Line check.
                    if (isCollisionLineLine(line1, line2)){
                        collision_counter += 1;
                        cp.push_back(i);
                        goto NEXT;
                    }
                    // Line-Circle check
                    if (isCollisionLineCircle(line1, line2[0], COLLISION_MARGIN)){
                        collision_counter += 1;
                        cp.push_back(i);
                        goto NEXT;
                    }
                    // Line-Circle check
                    if (isCollisionLineCircle(line1, line2[1], COLLISION_MARGIN)){
                        collision_counter += 1;
                        cp.push_back(i);
                        goto NEXT;
                    }
                    // Line-Circle check
                    if (isCollisionLineCircle(line2, line1[0], COLLISION_MARGIN)){
                        collision_counter += 1;
                        cp.push_back(i);
                        goto NEXT;
                    }
                    // Line-Circle check
                    if (isCollisionLineCircle(line2, line1[1], COLLISION_MARGIN)){
                        collision_counter += 1;
                        cp.push_back(i);
                        goto NEXT;
                    }

                }
            }
        }
        NEXT:;
    }


    if (cp.size() == 0){
        ind1.collision_points[0] = -1;
        ind1.collision_points[1] = -1;
    }else{
        ind1.collision_points[0] = (int)cp[0]/COLLISION_CHECK_DIV;
        ind1.collision_points[1] = (int)cp[cp.size()-1]/COLLISION_CHECK_DIV + 1;
    }

    return collision_counter;
}


// -----------------------------------------------
//  SORT
// -----------------------------------------------

bool operator<(const Individual& left, const Individual& right){
  return left.fitness< right.fitness ;
}

void sort_pop(vector<Individual> &pop){
    std::sort(pop.begin(), pop.end());
}


// -----------------------------------------------
// SELECTION
// -----------------------------------------------


vector<Individual> tournamentSelection(vector<Individual> const &pop, int n_offspring, int n_tournament){

    vector<Individual> offspring(n_offspring);
    int rand_index;
    int min_index;

    for(int i=0; i<n_offspring; i++){
        min_index = pop.size();
        for (int j=0; j<n_tournament; j++){
            rand_index = get_rand_range_int(0, pop.size()-1);
            if (rand_index < min_index) min_index = rand_index;
        }
        offspring[i] = pop[min_index];
    }

    return offspring;
}

vector<Individual> rouletteSelection(vector<Individual> &pop, int n_offspring){

    vector<Individual> offspring;
    vector<double> rand_list(n_offspring);

    // Calculate sum of the fitness over all population.
    double sum_fitness = std::accumulate(pop.begin(), pop.end(), 0.0,
                     //[](double sum, Individual& ind ){ return sum+1.0/ind.subfitness; } );
                     [](double sum, Individual& ind ){ return sum+ ind.subfitness; } );

    // Generate random list.
    for (int i=0; i<n_offspring; i++){
        rand_list[i] = get_rand_range_dbl(0.0, sum_fitness);
    }

    // Sort random_list.
    std::sort(rand_list.begin(), rand_list.end());
    std::reverse(rand_list.begin(), rand_list.end());

    double thresh = 0.0;
    for (Individual ind: pop){
        thresh += ind.subfitness;
        //thresh += 1.0/ind.fitness;
        while(rand_list.size()!=0){
            double rand = rand_list.back();
            if (rand<thresh){
                offspring.push_back(ind);
                rand_list.pop_back();
            }else{
                break;
            }
        }
    }
    return offspring;
}

vector<Individual> elitistSelection(vector<Individual> const &pop, int n_elites){
    vector<Individual> elites(n_elites);
    for (int i=0; i<N_ELITES; i++) elites[i] = pop[i];
    return elites;
}

vector<Individual> overageSelection(vector<Individual>&offspring, int id){
    vector<Individual> overages;
    int num=offspring.size();
    for (int i=num-1; i>=0; i--){
        if(offspring[i].age > MAX_AGE[id]){
            overages.insert(overages.begin(), offspring[i]);
            offspring.erase(offspring.begin()+i);
        }
    }
    return overages;
}

void agelayeredSelection(vector<Individual>pops[]){

    vector<Individual> overages[N_LAYERS];
    vector<Individual> elites(N_ELITES);

    for (int id=0; id<N_LAYERS; id++){

        if (id != N_LAYERS-1){
            overages[id+1] = overageSelection(pops[id], id);
        }

        int M = N_POP*(1.0 - RATE_OVERAGE);
        int N = N_POP*RATE_OVERAGE;
        int m = pops[id].size();
        int n = overages[id].size();

        if (m < M && n < N){
            pops[id].insert(pops[id].end(), overages[id].begin(), overages[id].end());

        }else if(m < M && n > N){
            N = N_POP - m;
            if (N < n){
                elites = elitistSelection(overages[id], N_ELITES);
                //overages[id] = rouletteSelection(overages[id], N - N_ELITES);
                overages[id] = tournamentSelection(overages[id], N - N_ELITES, N_TOURNAMENT);
                overages[id].insert(overages[id].end(), elites.begin(), elites.end());
            }
            pops[id].insert(pops[id].end(), overages[id].begin(), overages[id].end());

        }else if(m > M && n < N){
            M = N_POP - n;
            if (M < m){
                elites = elitistSelection(pops[id], N_ELITES);
                //pops[id] = rouletteSelection(pops[id], M - N_ELITES);
                pops[id] = tournamentSelection(pops[id], M - N_ELITES, N_TOURNAMENT);
                pops[id].insert(pops[id].end(), elites.begin(), elites.end());
            }
            pops[id].insert(pops[id].end(), overages[id].begin(), overages[id].end());

        }else if (m >= M && n >= N){
            elites = elitistSelection(pops[id], N_ELITES);
            //pops[id] = rouletteSelection(pops[id], M - N_ELITES);
            pops[id] = tournamentSelection(pops[id], M - N_ELITES, N_TOURNAMENT);
            pops[id].insert(pops[id].end(), elites.begin(), elites.end());

            elites = elitistSelection(overages[id], N_ELITES);
            //overages[id] = rouletteSelection(overages[id], N - N_ELITES);
            overages[id] = tournamentSelection(overages[id], N - N_ELITES, N_TOURNAMENT);
            overages[id].insert(overages[id].end(), elites.begin(), elites.end());

            pops[id].insert(pops[id].end(), overages[id].begin(), overages[id].end());
        }

        vector<double> age_list;
        int min_age; int max_age;
        if (id==0){min_age = 1;}else{min_age = MAX_AGE[id-1]+1;}
        max_age = MAX_AGE[id];
        for (Individual ind:pops[id]) age_list.push_back(ind.age);
        ofs_log << "    LAYER: " << id << endl;
        for (int age=min_age; age<=max_age; age++){
            size_t n_count = std::count(age_list.begin(), age_list.end(), age);
            ofs_log << "      AGE " << age << " : " << n_count <<endl;
        }
        ofs_log << "      TOTAL: " << pops[id].size() << endl;

    }
}

// -----------------------------------------------
// MUTATION
// -----------------------------------------------

void mutation(vector<Individual> &pop){
    //vector<Individual> add_pop;
    //int num = pop.size();
    //for (int i=0; i<num; i++){
        //if (get_rand_range_dbl(0.0, 1.0) < P_MUT){
            //Individual child = mut_operator(pop[i]);
            //add_pop.insert(add_pop.end(), child);
        //}

        //for(int j=0; j<5; j++){
        //    double p = pow(1.0 - P_MUT, j) * P_MUT;
        //    if (get_rand_range_dbl(0.0, 1.0) < p){
        //        mut_operator(pop[i]);
        //    }
        //}
    //}
    //pop.insert(pop.end(), add_pop.begin(), add_pop.end());

    int num = pop.size();
    for (int i=0; i<num; i++){
        if (get_rand_range_dbl(0.0, 1.0) < P_MUT){
            //mutHillclimb(pop[i]);
            mutNormal(pop[i]);
        }
    }
}

void mutHillclimb(Individual &ind){
    int index = get_rand_range_int(1, N_PTS-2);
    double theta = get_rand_range_dbl(-PI, PI);
    for (int i=0; i<30; i++){
        ind.path[index][0] += 0.1 * cos(theta);
        ind.path[index][1] += 0.1 * sin(theta);
        if (ind.distance < calcPathLength(ind.path)){
            break;
        }
    }
}

void mutNormal(Individual &ind){
    std::normal_distribution<> dist(0.0, 0.1);
    int rand_int = get_rand_range_int(1, N_PTS-2);
    //for (int i=0; i<N_PTS-1; i++){
        //if (get_rand_range_dbl(0.0, 1.0) < 0.3){
            //ind.path[i][0] += get_rand_range_dbl(-0.1, 0.1);
            //ind.path[i][1] += get_rand_range_dbl(-0.1, 0.1);
    ind.path[rand_int][0] += dist(mt_engine);
    ind.path[rand_int][1] += dist(mt_engine);
        //}
    //}

    //Individual child;
    //child.path = ind.path; child.age = ind.age; child.robotID = ind.robotID;
    //path_check(ind, "mut0");
    //path_check(child, "mut1");
    //return child;
}

// -----------------------------------------------
// CROSSOVER
// -----------------------------------------------

void crossover(vector<Individual> &pop1, const vector<Individual> &pop2){

    int num = pop1.size();
    vector<Individual> add_pop;
    //std::shuffle(pop.begin(), pop.end(), mt_engine);
    for(int i=0; i<num; i++){
        vector<Individual> children = oneptcx(pop1[i], pop2);
        if (children.size()!=0){
            add_pop.insert(add_pop.end(), children.begin(), children.end());
        }
    }
    pop1 = add_pop;
    //pop1.insert(pop1.end(), add_pop.begin(), add_pop.end());

    adjust_num_pts(pop1);

}

// --- Single-point crossover
vector<Individual> oneptcx(Individual &ind1, const vector<Individual> &pop){
    int cut_pt1 = get_rand_range_int(1, N_PTS-2);
    int cut_pt2;
    vector<double> pt1 = ind1.path[cut_pt1];
    vector<double> pt2;
    Individual ind2;
    vector<Individual> children;

    // Search the nearest point.
    double min_dist=PI/5;
    int num=pop.size();
    for (int i=0; i<num; i++){

        double sum = 0;
        for (int k=0; k<N_PTS; k++){
            vector<double> v1 = pop[i].path[k];
            vector<double> v2 = ind1.path[k];
            sum += calcDistance(v1, v2);
        }
        if (sum < 3.0) continue;

        for (int j=1; j<N_PTS-1; j++){
            pt2 = pop[i].path[j];
            double dist = calcDistance(pt1, pt2);
            if (dist < min_dist){
                min_dist = dist;
                ind2 = pop[i];
                cut_pt2 = j;
            }
        }
    }

    // Swap.
    if (ind2.path.size() != 0) {
        vector<vector<double>> path1 = ind1.path;
        vector<vector<double>> path2 = ind2.path;
        vector<vector<double>> child_path1;
        vector<vector<double>> child_path2;
        child_path1.insert(child_path1.end(), path1.begin(), path1.begin()+cut_pt1);
        child_path1.insert(child_path1.end(), path2.begin()+cut_pt2, path2.end());
        child_path2.insert(child_path2.end(), path2.begin(), path2.begin()+cut_pt2);
        child_path2.insert(child_path2.end(), path1.begin()+cut_pt1, path1.end());
        int age = std::max(ind1.age, ind2.age);
        // New individuals.
        Individual child1; child1.path=child_path1; child1.age=age; child1.robotID=ind1.robotID;
        Individual child2; child2.path=child_path2; child2.age=age; child2.robotID=ind2.robotID;
        children = {child1, child2};
        //path_check(ind1, "cx0");
        //path_check(ind2, "cx1");
        //path_check(child1, "cx2");
        //path_check(child2, "cx3");
    }else{
        children = {ind1};
        ofs_log << "    FAIL" << endl;
    }
    return children;
}

vector<Individual> oneptcx2(Individual &ind1, const vector<Individual> &pop){
    int cut_pt1, cut_pt2;
    vector<double> pt1, pt2;
    Individual ind2;
    vector<Individual> children;
    double min_dist=PI/10;
    int num=pop.size();

    // Cut point
    cut_pt1 = ind1.collision_points[0];
    if (cut_pt1 == -1 || cut_pt1 == 0) cut_pt1 = get_rand_range_int(1, N_PTS-2);
    pt1 = ind1.path[cut_pt1];

    // Search the nearest point.
    for (int i=0; i<num; i++){
        if (pop[i].distance == ind1.distance) continue;
        for (int j=0; j<N_PTS; j++){
            pt2 = pop[i].path[j];
            double dist = calcDistance(pt1, pt2);
            if (dist < min_dist){
                min_dist = dist;
                ind2 = pop[i];
                cut_pt2 = j;
            }
        }
    }

    // Swap.
    if (ind2.path.size() != 0) {
        vector<vector<double>> path1 = ind1.path;
        vector<vector<double>> path2 = ind2.path;
        vector<vector<double>> child_path1;
        child_path1.insert(child_path1.end(), path1.begin(), path1.begin()+cut_pt1);
        child_path1.insert(child_path1.end(), path2.begin()+cut_pt2, path2.end());
        int age = std::max(ind1.age, ind2.age);
        // New individuals.
        Individual child; child.path=child_path1; child.age=age; child.robotID=ind1.robotID;
        children.push_back(child);
    }

    // Cut point
    cut_pt1 = ind1.collision_points[1];
    if (cut_pt1 == -1 || cut_pt1 == 0) cut_pt1 = get_rand_range_int(1, N_PTS-2);
    pt1 = ind1.path[cut_pt1];

    // Search the nearest point.
    for (int i=0; i<num; i++){
        if (pop[i].distance == ind1.distance) continue;
        for (int j=0; j<N_PTS; j++){
            pt2 = pop[i].path[j];
            double dist = calcDistance(pt1, pt2);
            if (dist < min_dist){
                min_dist = dist;
                ind2 = pop[i];
                cut_pt2 = j;
            }
        }
    }

    // Swap.
    if (ind2.path.size() != 0) {
        vector<vector<double>> path1 = ind1.path;
        vector<vector<double>> path2 = ind2.path;
        vector<vector<double>> child_path1;
        child_path1.insert(child_path1.end(), path2.begin(), path2.begin()+cut_pt2);
        child_path1.insert(child_path1.end(), path1.begin()+cut_pt1, path1.end());
        int age = std::max(ind1.age, ind2.age);
        // New individuals.
        Individual child; child.path=child_path1; child.age=age; child.robotID=ind1.robotID;
        children.push_back(child);
    }

    return children;
}

// -----------------------------------------------

void calcDiversity(vector<Individual> &pop){
    //double diversity_alpha = 0.5;
    //double diversity_thresh = 1.0;

    //double sharing;
    int num = pop.size();

    // Initialize
    for (int i=0; i<num; i++) pop[i].diversity = 0.0;

    // Calculate distance between two paths.
    for (int i=0; i<num-1; i++){
        for (int j=i+1; j<num; j++){
            for (int k=0; k<N_PTS; k++){
                vector<double> pt1 = pop[i].path[k];
                vector<double> pt2 = pop[j].path[k];
                double dist = calcDistance(pt1, pt2);
                //if (dist < diversity_thresh){
                //    sharing = 1.0 - pow((dist/diversity_thresh), diversity_alpha);
                //}else{
                //    sharing = 0.0;
                //}
                //pop[i].diversity += sharing;
                //pop[j].diversity += sharing;
                pop[i].diversity += dist;
                pop[j].diversity += dist;
            }
        }
    }

    // Average distance.
    for (int i=0; i<num; i++) pop[i].diversity = pop[i].diversity/(num-1);

}

void adjust_num_pts(vector<Individual> &pop){

    int num = pop.size();
    for (int i=0; i<num; i++){
        int num = pop[i].path.size();
        if (num < N_PTS){
            int k = ((N_PTS - 1)/(num - 1)) + 1;
            pop[i].path = ferguson_spline(pop[i].path, k);
        }
        num = pop[i].path.size();
        while (num>N_PTS){
            int rand_index = get_rand_range_int(1, pop[i].path.size()-2);
            pop[i].path.erase(pop[i].path.begin() + rand_index);
            num = pop[i].path.size();
        }
    }
}

vector<vector<double>> ferguson_spline(vector<vector<double>>pts, int const num){

    const int n_pts = pts.size();

    // s matrix
    MatrixXd s_mat(num, 4);
    for (int i=0; i<num; i++){
        double s = (double)i/num;
        s_mat(i, 0) = s*s*s;
        s_mat(i, 1) = s*s;
        s_mat(i, 2) = s;
        s_mat(i, 3) = 1.0;
    }

    // Ferguson Spline Matrix
    Matrix<double, 4, 4> trans_matrix;
    trans_matrix << 2, 1, -2, 1,
                    -3, -2, 3, -1,
                    0, 1, 0, 0,
                    1, 0, 0, 0;

    // Velocity on each point
    MatrixXd velo(n_pts, 2);
    for (int i=0; i<n_pts; i++){
        for (int j=0; j<2; j++){
            if(i == n_pts-1){
                velo(i, j) = pts[i][j] - pts[i-1][j];
            }else if(i == 0){
                velo(i, j) = pts[i+1][j] - pts[i][j];
            }else{
                velo(i, j) = (pts[i+1][j] - pts[i-1][j])/2.0;
            }
        }
    }

    // Calc. spline
    vector<vector<double>> q;
    vector<vector<double>> q_add(num, vector<double>(2));
    Matrix<double, 4, 2> vec;
    MatrixXd q_add_mat(num, 2);

    for (int i=0; i<n_pts-1; i++){

        vec << pts[i][0], pts[i][1],
               velo(i,0), velo(i,1),
               pts[i+1][0], pts[i+1][1],
               velo(i+1,0), velo(i+1,1);

        q_add_mat = s_mat * trans_matrix * vec;

        // Matrix -> Vector conversion
        for (int i=0; i<num; i++){
            for (int j=0; j<2; j++){
                q_add[i][j] = q_add_mat(i, j);
            }
        }

        // Add vector.
        q.insert(q.end(), q_add.begin(), q_add.end());
    }
    q.insert(q.end(), pts.begin()+n_pts-1, pts.end());

    //std::ofstream ofs("./test.csv");
    //for (vector<double> pt:q){
    //    cout << pt[0] << " " << pt[1] << endl;
    //    ofs << pt[0] << "," << pt[1] << endl;
    //}

    return q;
}


