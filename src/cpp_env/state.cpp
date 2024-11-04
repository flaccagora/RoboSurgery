#include <vector>
#include <map>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <env.cpp>

// Define a custom struct for the state tuple
struct State {
    std::tuple<int, int, int> pos;  // (x, y, phi)
    std::pair<int, int> theta;      // (i, j)
    
    // Define operator< for map compatibility
    bool operator<(const State& other) const {
        return std::tie(pos, theta) < std::tie(other.pos, other.theta);
    }
    bool operator==(const State& other) const {
        return pos == other.pos && theta == other.theta;
    }
};

// Define observation type (5-element array of bools)
typedef std::array<bool, 5> Observation;

// Function to generate all possible binary combinations
std::vector<Observation> generateBinaryCombinations(int size) {
    std::vector<Observation> result;
    int total = pow(2, size);
    
    for (int i = 0; i < total; i++) {
        Observation obs;
        for (int j = 0; j < size; j++) {
            obs[j] = (i >> j) & 1;
        }
        result.push_back(obs);
    }
    return result;
}

// Main code
int main() {
    // Assuming these variables are defined somewhere
    int max_shape_x = env.max_shape[0];  // You'll need to replace with actual values
    int max_shape_y = env.max_shape[1];
    int l0 = /* your value */;
    int h0 = /* your value */;
    int l1 = /* your value */;
    int h1 = /* your value */;

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

    // Generate actions
    std::vector<int> actions = {0, 1, 2, 3};

    // Generate observations (all possible combinations of 5 binary values)
    std::vector<Observation> obs = generateBinaryCombinations(5);

    // Generate thetas
    std::vector<std::pair<int, int>> thetas;
    for (int i = l0; i < h0; i++) {
        for (int j = l1; j < h1; j++) {
            thetas.push_back(std::make_pair(i, j));
        }
    }

    // Create state dictionary
    std::map<State, int> state_dict;
    for (size_t i = 0; i < states.size(); i++) {
        state_dict[states[i]] = i;
    }

    // Create observation dictionary
    std::map<Observation, int> obs_dict;
    for (size_t i = 0; i < obs.size(); i++) {
        obs_dict[obs[i]] = i;
    }
}