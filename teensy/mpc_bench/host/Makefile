# x86 build of src/main.cpp (both solvers). Same flags for ours and TinyMPC: -O3 -DNDEBUG, no -march=native.
TP ?= /home/intsyslab/OCP-MPC/compare/third_party/TinyMPC-0.0.8
D := ../../../develop/cpp
CXXFLAGS := -O3 -DNDEBUG -std=c++17 -w -DBENCH_OURS -DBENCH_TINY -I../src -I$(D) -I$(TP)/include/Eigen -I$(TP)/src/tinympc
SRCS := ../src/main.cpp $(D)/lqocp.cpp $(D)/lqocp_state.cpp $(TP)/src/tinympc/admm.cpp $(TP)/src/tinympc/tiny_api.cpp $(TP)/src/tinympc/rho_benchmark.cpp

build/mpc_bench: $(SRCS) ../src/mpc_data.h
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRCS) -o $@
