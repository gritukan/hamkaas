#include "common.h"

#include <iostream>
#include <vector>

std::vector<double> SiLUGpu(std::vector<double> data)
{
    // Your code here.
}

std::vector<double> SiLUCpu(std::vector<double> data)
{
    for (int i = 0; i < data.size(); i++) {
        double x = data[i];
        data[i] = x / (1 + exp(-x));
    }

    return data;
}

bool DoTest(std::vector<double> data)
{
    auto gpuResult = SiLUGpu(data);
    auto cpuResult = SiLUCpu(data);

    if (gpuResult == cpuResult) {
        std::cerr << "Test passed (n = " << data.size() << ")" << std::endl;
        return true;
    } else {
        std::cerr << "Test failed (n = " << data.size() << ")" << std::endl;
        return false;
    }
}

int main()
{
    std::vector<std::vector<double>> testValues;

    for (int size : {1, 2, 3, 100, 1000, 10000}) {
        std::vector<double> a(size);
        for (int i = 0; i < size; i++) {
            if (i % 2 == 0) {
                a[i] = i / 1000;
            } else {
                a[i] = -i / 1000;
            }
        }
        testValues.push_back(a);
    }

    int passedTests = 0;
    for (auto data : testValues) {
        if (DoTest(data)) {
            passedTests++;
        }
    }

    if (passedTests == testValues.size()) {
        std::cerr << "All tests passed" << std::endl;
        return 0;
    } else {
        std::cerr << "Some tests failed" << std::endl;
        return 1;
    }
}
