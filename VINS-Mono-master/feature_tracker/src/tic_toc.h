#pragma once

#include <ctime>
#include <cstdlib>
#include <chrono>

class TicToc
{
  public:
    TicToc()
    {
        tic();
    }

    void tic()
    {
        start = std::chrono::system_clock::now();  //获取系统时间戳
    }

    double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;   //时间间隔s
        return elapsed_seconds.count() * 1000;   //count()得到当前对象保存的_Period的个数,*1000表示ms
    }                                            //chrono::duration_cast<chrono::milliseconds>elapsed_seconds.count();//时间间隔ms

  private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};
