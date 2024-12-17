// ObservableDeformedGridworld C++ Implementation
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <random>
#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <stdexcept>
#include <chrono>

namespace py = pybind11;

class ObservableDeformedGridworld {
public:
    using Vector2 = std::array<double, 2>;
    using Matrix2x2 = std::array<std::array<double, 2>, 2>;

    ObservableDeformedGridworld(Vector2 grid_size, double step_size, Vector2 goal, 
                                std::vector<std::array<Vector2, 2>> obstacles, Vector2 stretch,
                                Vector2 shear, double observation_radius, 
                                Vector2 shear_range, Vector2 stretch_range)
        : grid_size_(grid_size), step_size_(step_size), goal_(goal), state_({0.1, 0.1}), 
          obstacles_(std::move(obstacles)), observation_radius_(observation_radius),
          shear_range_(shear_range), stretch_range_(stretch_range), timestep_(0) {

        
        set_deformation(stretch, shear);

        corners_ = {Vector2{0, 0},
                    Vector2{grid_size_[0], 0},
                    grid_size_,
                    Vector2{0, grid_size_[1]}};

        corners_array_ = {Vector2{0, 0},
                    grid_size_};

        transformed_corners_ = transform_corners(corners_);

        std::random_device rd;         // Non-deterministic random seed
        rng.seed(rd());               // Mersenne Twister generator
    
    }
    std::mt19937 rng;

    void reset(std::optional<int> seed = std::nullopt) {
        
        if (seed.has_value()) {
            rng.seed(seed.value());
        }

        auto stretch = sample(rng, stretch_range_);
        auto shear = sample(rng, shear_range_);
        set_deformation(stretch, shear);

        transformed_corners_ = transform_corners(corners_);
        state_ = sample_in_parallelogram(transformed_corners_, rng);
        timestep_ = 0;
    }

    std::tuple<std::unordered_map<std::string, std::vector<double>>, double, bool, bool, std::unordered_map<std::string, bool>> step(int action) {
    static const std::array<Vector2, 4> moves = {
        Vector2{0, step_size_},
        Vector2{0, -step_size_},
        Vector2{step_size_, 0},
        Vector2{-step_size_, 0}
    };

    auto move = moves[action];
    Vector2 next_state = {state_[0] + move[0], state_[1] + move[1]};
    std::unordered_map<std::string, bool> info;
    float reward;
    bool terminated;

    if (distance(next_state, transform(goal_)) < observation_radius_) {
        info = {{"goal", true}};
        reward = 1.0;
        terminated = true;
    }else if (!is_point_in_parallelogram(next_state, corners_array_)) {
        info = {{"out", true}};
        reward = -2.0;
        terminated = false;
        next_state = state_;
    }else if (is_collision(next_state)) {
        info = {{"collision", true}};
        reward = -2.0;
        terminated = false;
        // next_state = state_;
    }else {
        info = {};
        reward = -0.25;
        terminated = false;
    }

    state_ = next_state;
    timestep_++;
    return make_step_response(state_, reward, terminated, timestep_ > 500, info);
}

    void set_deformation(Vector2 stretch, Vector2 shear) {
        transformation_matrix_ = {{
            {stretch[0], shear[0]},
            {shear[1], stretch[1]}
        }};
        inverse_transformation_matrix_ = invert(transformation_matrix_);
    }

    Vector2 transform(const Vector2& position) const {
        return matrix_vector_multiply(transformation_matrix_, position);
    }

    Vector2 inverse_transform(const Vector2& position) const {
        return matrix_vector_multiply(inverse_transformation_matrix_, position);
    }


    Vector2 grid_size_;
    Vector2 state_;
    std::vector<std::array<Vector2, 2>> obstacles_;
    Vector2 goal_;
    double observation_radius_;
    Vector2 shear_range_;
    Vector2 stretch_range_;
    Matrix2x2 transformation_matrix_;
    
    
    bool is_collision(const Vector2& position) const {
        for (const auto& obstacle : obstacles_) {
            if (is_point_in_parallelogram(position, obstacle)) {
                return true;
            }
        }
        return false;
    }


    // check if in direction \phi there is an obstacle inside the observation radius
    bool is_collision_in_direction(const Vector2& position, double phi) const {
        Vector2 direction = {std::cos(phi), std::sin(phi)};
        // check at different distances
        std::array<double, 10> distances = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
        for (const auto& distance : distances) {
            Vector2 next_position = {position[0] + direction[0] * distance * observation_radius_,
                                     position[1] + direction[1] * distance * observation_radius_};
            if (is_collision(next_position)) {
                return true;
            }
        }
        return false;
    }
    // check if in direction N S W E there is an obstacle inside the observation radius return an array of 4 bools [E,W,N,S]
    std::array<bool, 4> is_collision_in_cardinal_directions(const Vector2& position) const {
        return {
            is_collision_in_direction(position, 0),
            is_collision_in_direction(position, M_PI),
            is_collision_in_direction(position, M_PI / 2),
            is_collision_in_direction(position, 3 * M_PI / 2)
        };
    }



private:
    std::array<Vector2, 2> corners_array_ ;
    double step_size_;
    Matrix2x2 inverse_transformation_matrix_;
    std::vector<Vector2> corners_;
    std::vector<Vector2> transformed_corners_;
    int timestep_;

    Matrix2x2 invert(const Matrix2x2& matrix) const {
        double det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        if (std::abs(det) < 1e-10) {
            throw std::runtime_error("Matrix inversion failed: Determinant is too small.");
        }

        return {{
            {matrix[1][1] / det, -matrix[0][1] / det},
            {-matrix[1][0] / det, matrix[0][0] / det}
        }};
    }

    Vector2 matrix_vector_multiply(const Matrix2x2& matrix, const Vector2& vec) const {
        return {
            matrix[0][0] * vec[0] + matrix[0][1] * vec[1],
            matrix[1][0] * vec[0] + matrix[1][1] * vec[1]
        };
    }

    double distance(const Vector2& a, const Vector2& b) const {
        return std::sqrt(std::pow(a[0] - b[0], 2) + std::pow(a[1] - b[1], 2));
    }

    bool is_point_in_parallelogram(const Vector2& point, const std::array<Vector2,2>& parallelogram) const {
        // per verificare se il punto è dentro il parallelogramma 
        // basta inverse transform del punto e verificare se è dentro il non deformato, lo stesso vale per gli ostacoli? sì
        // le coordinate del punto sono universali
        // le coordinate del parallelogramma sono quelle non trasformate
        Vector2 inv_point = inverse_transform(point); 

        return parallelogram[0][0] <= inv_point[0] && inv_point[0] <= parallelogram[1][0] &&
               parallelogram[0][1] <= inv_point[1] && inv_point[1] <= parallelogram[1][1];
    }

    Vector2 sample_in_parallelogram(const std::vector<Vector2>& parallelogram, std::mt19937& rng) {
        // Implementation for sampling in parallelogram
        // sample a poitn in box defined by grid_isze and then transform
        Vector2 point = sample(rng, {0, 1});
        return transform({point[0] * grid_size_[0], point[1] * grid_size_[1]});
    }

    template <typename RNG>
    Vector2 sample(RNG& rng, const Vector2& range) {
        std::uniform_real_distribution<double> dist(range[0], range[1]);
        return {dist(rng), dist(rng)};
    }

    std::vector<Vector2> transform_corners(const std::vector<Vector2>& corners) const {
        std::vector<Vector2> transformed;
        for (const auto& corner : corners) {
            transformed.push_back(transform(corner));
        }
        return transformed;
    }

    std::tuple<
    std::unordered_map<std::string, std::vector<double>>,
    double,
    bool,
    bool,
    std::unordered_map<std::string, bool>
    > make_step_response(
    const Vector2& state, double reward, bool terminated, bool truncated,
    const std::unordered_map<std::string, bool>& info) {
    return std::make_tuple(
        std::unordered_map<std::string, std::vector<double>>{
            {"pos", {state[0], state[1]}},
            {"theta", flatten(transformation_matrix_)}
        },
        reward, terminated, truncated, info
    );
}


    std::vector<double> flatten(const Matrix2x2& matrix) const {
        return {matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]};
    }
};

PYBIND11_MODULE(gridworld, m) {
    py::class_<ObservableDeformedGridworld>(m, "ObservableDeformedGridworld")
        .def(py::init<ObservableDeformedGridworld::Vector2, double, ObservableDeformedGridworld::Vector2, 
                      std::vector<std::array<ObservableDeformedGridworld::Vector2, 2>>, 
                      ObservableDeformedGridworld::Vector2, ObservableDeformedGridworld::Vector2, 
                      double, ObservableDeformedGridworld::Vector2, ObservableDeformedGridworld::Vector2>())
        .def("reset", &ObservableDeformedGridworld::reset, py::arg("seed") = py::none())
        .def("step", &ObservableDeformedGridworld::step)
        .def("transform", &ObservableDeformedGridworld::transform)
        .def("is_collision", &ObservableDeformedGridworld::is_collision)
        .def("set_deformation", &ObservableDeformedGridworld::set_deformation)
        .def("is_collision_cardinal", &ObservableDeformedGridworld::is_collision_in_cardinal_directions)
        .def_readonly("transformation_matrix", &ObservableDeformedGridworld::transformation_matrix_)
        .def_readonly("shear_range", &ObservableDeformedGridworld::shear_range_)
        .def_readonly("stretch_range", &ObservableDeformedGridworld::stretch_range_)
        .def_readonly("observation_radius", &ObservableDeformedGridworld::observation_radius_)
        .def_readonly("state", &ObservableDeformedGridworld::state_)
        .def_readonly("goal", &ObservableDeformedGridworld::goal_)
        .def_readonly("obstacles", &ObservableDeformedGridworld::obstacles_)
        .def_readonly("grid_size", &ObservableDeformedGridworld::grid_size_); // Expose public attribute x
}
