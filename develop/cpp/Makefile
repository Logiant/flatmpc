CXX ?= g++
CXXFLAGS ?= -O2 -std=c++14 -Wall -Wextra -fno-exceptions
BUILD := build

all: $(BUILD)/lqocp_cli

$(BUILD)/lqocp_cli: lqocp_cli.cpp lqocp.cpp lqocp.hpp case_data.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ lqocp_cli.cpp lqocp.cpp

$(BUILD):
	mkdir -p $(BUILD)

# Regenerate the case-data header from problem/quad_problem.py (committed copy is used by default).
case_data:
	python3 gen_case_data.py case_data.h

clean:
	rm -rf $(BUILD)

.PHONY: all clean case_data
