#ifndef RECORDLOG_H
#define RECORDLOG_H

#include <string>
#include "GeneticOperators.h"

void RecordLog(vector<Individual>[], vector<Individual>[], bool);
void config_check(Individual, int);
void path_check(Individual, std::string);
extern std::ofstream ofs_log;

#endif
