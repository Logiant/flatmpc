CXX ?= g++
CXXFLAGS ?= -O2 -std=c++14 -Wall -Wextra -fno-exceptions
BUILD := build

all: $(BUILD)/lqocp_cli

$(BUILD)/lqocp_cli: lqocp_cli.cpp lqocp.cpp lqocp_basis.cpp lqocp.hpp case_data.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ lqocp_cli.cpp lqocp.cpp lqocp_basis.cpp

$(BUILD):
	mkdir -p $(BUILD)

# Regenerate the case-data header from problem/quad_problem.py (committed copy is used by default).
case_data:
	python3 gen_case_data.py case_data.h

# Flash table of TINY's stable bases + the A/B/TINY problems for teensy/lqocp_bench (add ALL=all
# to tabulate every valid active set instead of those a host solve of TINY touches).
teensy_table: gen_basis_table.cpp lqocp.cpp lqocp_basis.cpp lqocp.hpp case_data.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $(BUILD)/gen_basis_table gen_basis_table.cpp lqocp.cpp lqocp_basis.cpp
	$(BUILD)/gen_basis_table ../../teensy/lqocp_bench/src/bench_data.h $(ALL)

clean:
	rm -rf $(BUILD)

.PHONY: all clean case_data teensy_table
