#include <iostream>
#include <Eigen/Dense>

#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <cmath>
#include <Eigen/Dense>
#include <algorithm>
#include <tuple>
#include <state.cpp>


// g++ env.cpp -o env -I /usr/local/include/eigen3

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

    
private:
    Eigen::MatrixXi original_maze;
    Eigen::MatrixXi maze;
    std::vector<int> actions = {0, 1, 2, 3}; // up, right, down, left
    std::vector<int> orientations = {0, 1, 2, 3};
    std::vector<std::tuple<int, int>> deformations;
    Eigen::Vector2i goal_pos;
    int l0, h0, l1, h1;
    int timestep;

};


GridEnvDeform::GridEnvDeform(const Eigen::MatrixXi& maze, int l0, int h0, int l1, int h1)
    : original_maze(maze), maze(maze), l0(l0), h0(h0), l1(l1), h1(h1), timestep(0) {

    // Define the list of possible deformations
    for (int i = l0; i < h0; ++i) {
        for (int j = l1; j < h1; ++j) {
            deformations.emplace_back(i, j);
        }
    }

    goal_pos = Eigen::Vector2i(maze.rows() - 2, maze.cols() - 2);
    max_shape[0] = original_maze.rows() * (h1-1) + 2;
    max_shape[1] = original_maze.cols() * (h0-1) + 2 ;

}

Eigen::MatrixXi GridEnvDeform::stretch_maze(const std::tuple<int, int>& thetas) {
    int scale_x = std::get<0>(thetas);
    int scale_y = std::get<1>(thetas);

    int new_height = original_maze.rows() * scale_y;
    int new_width = original_maze.cols() * scale_x;
    Eigen::MatrixXi stretched_maze = Eigen::MatrixXi::Ones(new_height, new_width);

    for (int i = 0; i < original_maze.rows(); ++i) {
        for (int j = 0; j < original_maze.cols(); ++j) {
            if (original_maze(i, j) == 0) {
                stretched_maze.block(i * scale_y, j * scale_x, scale_y, scale_x).setZero();
            }
        }
    }

    return stretched_maze;
}

void GridEnvDeform::set_deformed_maze(const std::tuple<int, int>& thetas) {
    maze = stretch_maze(thetas);
    goal_pos = Eigen::Vector2i(original_maze.rows() * std::get<1>(thetas)-2, original_maze.cols() * std::get<0>(thetas)-2);
    std::cout << "Goal position: " << goal_pos << std::endl;
}

void GridEnvDeform::set_state(const std::tuple<int, int, int>& state, const std::tuple<int, int>& thetas) {
    agent_pos = std::make_tuple(std::get<0>(state), std::get<1>(state));
    agent_orientation = std::get<2>(state);
    set_deformed_maze(thetas);
}

std::tuple<std::tuple<int, int, int>, std::tuple<int, int>> GridEnvDeform::reset() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_x(1, maze.rows() - 2);
    std::uniform_int_distribution<> dis_y(1, maze.cols() - 2);
    std::uniform_int_distribution<> dis_orient(0, 3);
    std::uniform_int_distribution<> dis_deform(0, deformations.size() - 1);

    int random_idx = dis_deform(gen);
    auto random_deformation = deformations[random_idx];
    agent_pos = std::make_tuple(dis_x(gen), dis_y(gen));
    // goal_pos = Eigen::Vector2i(maze.rows() - 2, maze.cols() - 2);
    agent_orientation = dis_orient(gen);

    set_deformed_maze(random_deformation);
    timestep = 0;

    return {std::make_tuple(std::get<0>(agent_pos), std::get<1>(agent_pos), agent_orientation), random_deformation};
}

std::tuple<std::tuple<int, int, int>, int, bool, bool, std::string> GridEnvDeform::step(int action) {
    // Action mapping based on orientation
    int actual_action = (action + agent_orientation) % 4;

    int x = std::get<0>(agent_pos);
    int y = std::get<1>(agent_pos);
    int new_x = x, new_y = y;

    // Move based on action
    if (actual_action == 0) {  // Move up
        new_x -= 1;
    } else if (actual_action == 1) {  // Move right
        new_y += 1;
    } else if (actual_action == 2) {  // Move down
        new_x += 1;
    } else if (actual_action == 3) {  // Move left
        new_y -= 1;
    }

    // Ensure new position is within bounds and not a wall
    bool terminated = false;
    int reward = -1;
    if (new_x > 0 && new_x < maze.rows() - 1 && new_y > 0 && new_y < maze.cols() - 1) {
        agent_pos = std::make_tuple(new_x, new_y);
        reward = -0.5;
        if (new_x == goal_pos(0) && new_y == goal_pos(1)) {
            terminated = true;
            reward = 1;
        }
    } else {
        reward = -2; // penalty for hitting a borderwall
    }

    // Update orientation
    agent_orientation = (agent_orientation + action) % 4;

    return {std::make_tuple(std::get<0>(agent_pos), std::get<1>(agent_pos), agent_orientation), reward, terminated, false, ""};
}

void GridEnvDeform::render() {
    std::cout << "Maze:" << std::endl;
    for (int i = 0; i < maze.rows(); ++i) {
        for (int j = 0; j < maze.cols(); ++j) {
            if (i == std::get<0>(agent_pos) && j == std::get<1>(agent_pos)) {
                if(agent_orientation == 1){
                    std::cout << "> ";
                }else if(agent_orientation == 2){
                    std::cout << "v ";
                }else if(agent_orientation == 3){
                    std::cout << "< ";
                }else{
                    std::cout << "^ ";
                }
            } else if (i == goal_pos(0) && j == goal_pos(1)) {
                std::cout << "G ";
            } else if (maze(i, j) == 0) {
                std::cout << ". ";
            } else {
                std::cout << "# ";
            }
        }
        std::cout << std::endl;
    }

    std::cout << "Agent position: (" << std::get<0>(agent_pos) << ", " << std::get<1>(agent_pos) << ")" << std::endl;
    std::cout << "Agent orientation: " << agent_orientation << std::endl;
    std::cout << "Goal position: (" << goal_pos(0) << ", " << goal_pos(1) << ")" << std::endl;
}


// void test_get_observation(GridEnvDeform& env) {
//     std::cout << "\nTesting get_observation() function..." << std::endl;
//     auto observation = env.get_observation();

//     std::cout << "Observation around the agent: ";
//     for (int obs : observation) {
//         std::cout << obs << " ";
//     }
//     std::cout << std::endl;
//     std::cout << "Get observation test passed." << std::endl;
// }


int main() {
    // Initialize a simple maze for testing
    Eigen::MatrixXi maze(5, 5);
    maze << 1, 1, 1, 1, 1,
            1, 0, 1, 0, 1,
            1, 0, 1, 0, 1,
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 1;

    int l0 = 1, h0 = 3;
    int l1 = 1, h1 = 2;

    // Create GridEnvDeform object
    GridEnvDeform env(maze, l0, h0, l1, h1);
    env.reset();
    env.render();



    // define states 

    int max_shape_x = env.max_shape[0];  // You'll need to replace with actual values
    int max_shape_y = env.max_shape[1];


    // Generate states
    std::vector<State> states;
    for (int x = 1; x < max_shape_x - 1; x++) {
        for (int y = 1; y < max_shape_y - 1; y++) {
            for (int phi = 0; phi < 4; phi++) {
                for (int i = l0; i < h0; i++) {
                    for (int j = l1; j < h1; j++) {
                        State state;
                        state.pos = std::make_tuple(x, y, phi);
                        state.theta = std::make_pair(i, j);
                        states.push_back(state);
                    }
                }
            }
        }
    }

    std::cout << "States generated: " << states.size() << std::endl;
    // show first 10 states
    for (int i = 0; i < 10; i++) {
        std::cout << "State " << i << ": (" << std::get<0>(states[i].pos) << ", " << std::get<1>(states[i].pos) << ", " << std::get<2>(states[i].pos) << "), (" << states[i].theta.first << ", " << states[i].theta.second << ")" << std::endl;
    }


    // Generate actions
    std::vector<int> actions = {0, 1, 2, 3};


    return 0;
}
