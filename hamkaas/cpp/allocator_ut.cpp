#include "allocator.h"

#include <iostream>
#include <utility>
#include <vector>

bool DoTest(int allocations, int minConcurrentAllocations, int maxConcurrentAllocations)
{
    std::cerr
        << "Running test with " << allocations << " allocations, " 
        << minConcurrentAllocations << " min concurrent allocations, and "
        << maxConcurrentAllocations << " max concurrent allocations." << std::endl;

    std::vector<std::pair<int64_t, int64_t>> ranges;
    int64_t totalSize = 0;

    NHamKaas::TAllocator allocator;
    for (int i = 0; i < allocations; i++) {
        bool allocate = false;
        if (ranges.size() < minConcurrentAllocations) {
            allocate = true;
        } else if (ranges.size() < maxConcurrentAllocations & rand() % 2 == 0) {
            allocate = true;
        }

        if (allocate) {
            int size = 0;
            if (rand() % 10 != 0) {
                size = rand() % (1000 * 1000);
            }

            auto begin = allocator.Allocate(size);
            if (begin % 256) {
                std::cerr << "Test failed: alignment mismatch!" << std::endl;
                return false;
            }

            auto end = begin + size;

            totalSize = std::max(totalSize, end);

            for (auto& range : ranges) {
                if (std::max(range.first, begin) < std::min(range.second, end)) {
                    std::cerr << "Test failed: overlap detected!" << std::endl;
                    return false;
                }
            }

            ranges.push_back({begin, end});
        } else {
            int index = rand() % ranges.size();
            std::swap(ranges[index], ranges.back());

            allocator.Free(ranges.back().first, ranges.back().second - ranges.back().first);
            ranges.pop_back();
        }
    }

    if (allocator.GetWorkingSetSize() < totalSize) {
        std::cerr << "Test failed: working set size mismatch!" << std::endl;
        return false;
    }

    std::cerr << "Test passed!" << std::endl;

    return true;
}

int main()
{
    bool ok = true;
    ok &= DoTest(1000, 1, 1);
    ok &= DoTest(1000, 1, 10);
    ok &= DoTest(1000, 10, 10);
    ok &= DoTest(1000, 10, 100);

    if (ok) {
        std::cerr << "All tests passed!" << std::endl;
        return 0;
    } else {
        std::cerr << "Some tests failed!" << std::endl;
        return 1;
    }
}