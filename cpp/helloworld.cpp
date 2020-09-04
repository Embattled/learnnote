#include <iostream>
#include "math.h"
#include "stdlib.h"
#include <vector>
#include <cstring>

int main(int argc, char const *argv[])
{
    int t;
    std::cin >> t;
    for (int i = 0; i < t; i++)
    {
        int N;
        std::cin >> N;
        std::vector<int> ch;
        for (int j = 0; j < N; j++)
        {
            std::cout << j << std::endl;
            // std::cin >> ch[j];
            int c;
            std::cin >> c;
            ch.push_back(c);
        }

        long long n = N * N;
        long long *mysum = new long long[n];
        std::memset(mysum, 0, n * sizeof(long long));

        for (int i = 0; i < N; i++)
        {
            mysum[i * N + i] = ch[i];
        }
        for (int span = 1; span < N; span++)
        {
            for (int start = 0; start < N; start++)
            {
                int end = start + span;
                if (end >= N)
                    continue;
                std::cout << start << "  " << end << std::endl;
                long long index = start * N + end;
                mysum[index] = mysum[index - 1] + ch[end];
            }
        }
        long long max = -1;
        for (long long i = 0; i < n; i++)
        {
            max = abs(mysum[i]) > max ? abs(mysum[i]) : max;
        }
        std::cout << max;
        delete[] mysum;
    }

    return 0;
}
