#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <torch/torch.h>

class GridEnvDeform {
public:
    GridEnvDeform(const torch::Tensor& maze, int l0, int h0, int l1, int h1);

    std::tuple<std::tuple<int, int, int>, std::tuple<int, int>> reset();
    std::tuple<std::tuple<int, int, int>, int, bool, bool, std::string> step(int action);
    void render();
    bool is_done();
    std::vector<int> get_observation();
    
private:
    torch::Tensor original_maze;
    torch::Tensor maze;
    std::vector<int> actions = {0, 1, 2, 3}; // up, right, down, left
    std::vector<int> orientations = {0, 1, 2, 3};
    std::vector<std::tuple<int, int>> deformations;
    std::tuple<int, int> agent_pos;
    int agent_orientation;
    std::tuple<int, int> goal_pos;
    int l0, h0, l1, h1;
    int timestep;

    // Helper functions
    torch::Tensor stretch_maze(const std::tuple<int, int>& thetas);
    void set_deformed_maze(const std::tuple<int, int>& thetas);
};

GridEnvDeform::GridEnvDeform(const torch::Tensor& maze, int l0, int h0, int l1, int h1)
    : original_maze(maze.clone()), maze(maze.clone()), l0(l0), h0(h0), l1(l1), h1(h1), timestep(0) {
    
    // Define the list of possible deformations
    for (int i = l0; i < h0; ++i) {
        for (int j = l1; j < h1; ++j) {
            deformations.emplace_back(i, j);
        }
    }

    // Initialize goal position
    goal_pos = std::make_tuple(maze.size(0) - 2, maze.size(1) - 2);
}

torch::Tensor GridEnvDeform::stretch_maze(const std::tuple<int, int>& thetas) {
    int scale_x = std::get<0>(thetas);
    int scale_y = std::get<1>(thetas);

    int new_height = original_maze.size(0) * scale_y;
    int new_width = original_maze.size(1) * scale_x;
    
    // Create new tensor filled with ones
    auto stretched_maze = torch::ones({new_height, new_width}, original_maze.options());
    
    // Iterate through the original maze and stretch it
    for (int i = 0; i < original_maze.size(0); ++i) {
        for (int j = 0; j < original_maze.size(1); ++j) {
            if (original_maze[i][j].item<int>() == 0) {
                // Create a block of zeros in the stretched maze
                stretched_maze.slice(0, i * scale_y, (i + 1) * scale_y)
                            .slice(1, j * scale_x, (j + 1) * scale_x).fill_(0);
            }
        }
    }
    
    return stretched_maze;
}

void GridEnvDeform::set_deformed_maze(const std::tuple<int, int>& thetas) {
    maze = stretch_maze(thetas);
    goal_pos = std::make_tuple(
        original_maze.size(0) * std::get<1>(thetas) - 2,
        original_maze.size(1) * std::get<0>(thetas) - 2
    );
    std::cout << "Goal position: (" << std::get<0>(goal_pos) << ", " << std::get<1>(goal_pos) << ")" << std::endl;
}

std::tuple<std::tuple<int, int, int>, std::tuple<int, int>> GridEnvDeform::reset() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_x(1, maze.size(0) - 2);
    std::uniform_int_distribution<> dis_y(1, maze.size(1) - 2);
    std::uniform_int_distribution<> dis_orient(0, 3);
    std::uniform_int_distribution<> dis_deform(0, deformations.size() - 1);

    int random_idx = dis_deform(gen);
    auto random_deformation = deformations[random_idx];
    agent_pos = std::make_tuple(dis_x(gen), dis_y(gen));
    agent_orientation = dis_orient(gen);

    set_deformed_maze(random_deformation);
    timestep = 0;

    return {std::make_tuple(std::get<0>(agent_pos), std::get<1>(agent_pos), agent_orientation), random_deformation};
}

std::tuple<std::tuple<int, int, int>, int, bool, bool, std::string> GridEnvDeform::step(int action) {
    int actual_action = (action + agent_orientation) % 4;

    int x = std::get<0>(agent_pos);
    int y = std::get<1>(agent_pos);
    int new_x = x, new_y = y;

    // Move based on action
    if (actual_action == 0) new_x -= 1;      // Up
    else if (actual_action == 1) new_y += 1;  // Right
    else if (actual_action == 2) new_x += 1;  // Down
    else if (actual_action == 3) new_y -= 1;  // Left

    bool terminated = false;
    int reward = -1;
    
    if (maze[new_x][new_y].item<int>() == 0) {
        agent_pos = std::make_tuple(new_x, new_y);
        reward = -0.5;
        if (new_x == std::get<0>(goal_pos) && new_y == std::get<1>(goal_pos)) {
            terminated = true;
            reward = 1;
        }
    } else {
        reward = -2; // penalty for hitting a wall
    }

    agent_orientation = (agent_orientation + action) % 4;

    return {std::make_tuple(new_x, new_y, agent_orientation), reward, terminated, false, ""};
}

void GridEnvDeform::render() {
    std::cout << "Maze:" << std::endl;
    for (int i = 0; i < maze.size(0); ++i) {
        for (int j = 0; j < maze.size(1); ++j) {
            if (i == std::get<0>(agent_pos) && j == std::get<1>(agent_pos)) {
                std::cout << "A ";
            } else if (i == std::get<0>(goal_pos) && j == std::get<1>(goal_pos)) {
                std::cout << "G ";
            } else if (maze[i][j].item<int>() == 0) {
                std::cout << ". ";
            } else {
                std::cout << "# ";
            }
        }
        std::cout << std::endl;
    }

    std::cout << "Agent position: (" << std::get<0>(agent_pos) << ", " << std::get<1>(agent_pos) << ")" << std::endl;
    std::cout << "Agent orientation: " << agent_orientation << std::endl;
    std::cout << "Goal position: (" << std::get<0>(goal_pos) << ", " << std::get<1>(goal_pos) << ")" << std::endl;
}

// Main function for testing
int main() {
    // Initialize a simple maze using PyTorch
    auto options = torch::TensorOptions().dtype(torch::kInt32);
    auto maze = torch::ones({5, 5}, options);
    
    // Set the maze layout (0 for paths, 1 for walls)
    maze.slice(0, 1, 4).slice(1, 1, 2).fill_(0);  // Middle vertical path
    maze.slice(0, 1, 4).slice(1, 3, 4).fill_(0);  // Right vertical path
    maze.slice(0, 3, 4).slice(1, 1, 4).fill_(0);  // Bottom horizontal path

    int l0 = 1, h0 = 3;
    int l1 = 1, h1 = 2;

    // Create GridEnvDeform object
    GridEnvDeform env(maze, l0, h0, l1, h1);
    
    // Test the environment
    env.reset();
    env.render();
    
    // Test a step
    auto [next_state, reward, terminated, truncated, info] = env.step(1);
    env.render();
    
    std::cout << "Step result:" << std::endl;
    std::cout << "Reward: " << reward << std::endl;
    std::cout << "Terminated: " << terminated << std::endl;

    return 0;
}