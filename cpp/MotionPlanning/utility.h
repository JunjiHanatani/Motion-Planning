#ifndef _UTILITY_H_
#define _UTILITY_H_
#include <random>
#include <string>
#include <chrono>

// Random Seed
extern std::mt19937 mt_engine;

int get_rand_range_int(int, int);
double get_rand_range_dbl(double, double);

extern double PI;

// Class for the measurement of elapsed time using timeGetTime()
class Timer
{
public:
    Timer() { restart(); }
public:
    void  restart(){
        start = std::chrono::system_clock::now();
    }

    double  elapsed(){
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        return elapsed_seconds.count() ;
    }
private:
    std::chrono::system_clock::time_point start;
};


#endif // _UTILITY_H_

