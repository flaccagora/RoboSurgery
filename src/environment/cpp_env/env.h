#ifndef ENV_H
#define ENV_H

#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <cmath>
#include <Eigen/Dense>
#include <algorithm>
#include <tuple>
#include "state.cpp"

class GridEnvDeform {
public:

    GridEnvDeform(const Eigen::MatrixXi& maze, int l0, int h0, int l1, int h1);

    std::tuple<std::tuple<int, int, int>, std::tuple<int, int>> reset();
    std::tuple<std::tuple<int, int, int>, int, bool, bool, std::string> step(int action);
    void render();
    bool is_done();
    std::vector<int> get_observation();
    std::tuple<std::tuple<int,int>> get_goal_pos();
    // Helper functions
    Eigen::MatrixXi stretch_maze(const std::tuple<int, int>& thetas);
    void set_deformed_maze(const std::tuple<int, int>& thetas);
    void set_state(const std::tuple<int, int, int>& state, const std::tuple<int, int>& thetas);
    std::tuple<int, int> agent_pos;
    int agent_orientation;
    int max_shape[2];
    Eigen::MatrixXi original_maze;
    Eigen::MatrixXi maze;
    std::vector<int> actions = {0, 1, 2, 3}; // up, right, down, left
    std::vector<int> orientations = {0, 1, 2, 3};
    std::vector<std::tuple<int, int>> deformations;
    Eigen::Vector2i goal_pos;
    int l0, h0, l1, h1;
    int timestep;

};

#endif  // ENV_H
