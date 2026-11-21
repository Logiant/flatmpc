TP := ../third_party/TinyMPC-0.0.8
CXX ?= g++
# -O3 only (no -march=native) so the numbers are not tuned beyond a generic release build.
CXXFLAGS := -O3 -DNDEBUG -std=c++17 -w -I$(TP)/include -I$(TP)/include/Eigen -I$(TP)/src
SRCS := tiny_bench.cpp $(TP)/src/tinympc/admm.cpp $(TP)/src/tinympc/tiny_api.cpp $(TP)/src/tinympc/rho_benchmark.cpp

build/tiny_bench: $(SRCS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRCS) -o $@

clean:
	rm -rf build
.PHONY: clean
