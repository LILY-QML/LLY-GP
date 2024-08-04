![GP-removebg-preview](https://github.com/user-attachments/assets/0f7ed5fc-82d4-44ab-91c3-65a7419c602a)
# Introduction

**LLY-DML** is part of the [**LILY Project**](https://www.lilyqml.de) and focuses on optimization parameter-based quantum circuits. It enhances the efficiency of quantum algorithms by fine-tuning parameters of quantum gates. **DML** stands for **Differentiable Machine Learning**, emphasizing the use of gradient-based optimization techniques to improve the performance of quantum circuits.

LLY-DML is available on the [LILY QML platform](https://www.lilyqml.de), making it accessible for researchers and developers.

For inquiries or further information, please contact: [info@lilyqml.de](mailto:info@lilyqml.de).

## Contributors

| Role                     | Name          | Links                                                                                                                |
|--------------------------|---------------|----------------------------------------------------------------------------------------------------------------------|
| Project Lead             | Leon Kaiser   | [ORCID](https://orcid.org/0009-0000-4735-2044), [GitHub](https://github.com/xleonplayz)                              |
| Inquiries and Management | Raul Nieli    | [Email](mailto:raul.nieli@lilyqml.de)                                                                                |
| Supporting Contributors  | Eileen Kühn   | [GitHub](https://github.com/eileen-kuehn), [KIT Profile](https://www-kseta.ttp.kit.edu/fellows/Eileen.Kuehn/)        |
| Supporting Contributors  | Max Kühn      | [GitHub](https://github.com/maxfischer2781)                                                                          |



## Table of Contents

1. [Quantum ML-Gate: L-Gate](#quantum-ml-gate-l-gate)
2. [Objective of Training](#objective-of-training)
3. [Optimization Methods](#optimization-methods)
4. [Public Collaboration](#public-collaboration)


## Quantum ML-Gate: L-Gate

The **L-Gate** is a pivotal component in quantum machine learning circuits, designed to meet specific requirements for effective parameter optimization. It integrates input parameters with optimization parameters, allowing for a seamless flow of data and control. Here are the key properties and design aspects of the L-Gate:

### Key Properties

1. **Parameter Optimization:**  
   The L-Gate must enable the optimization of parameters, allowing for fine-tuning that enhances the performance of quantum algorithms. This optimization is achieved by merging input parameters with optimization parameters to create a dynamic and responsive system.

2. **Full Bloch Sphere Utilization:**  
   The design of the L-Gate ensures that the entire Bloch sphere is accessible. This feature allows for a complete range of quantum state manipulations, providing flexibility and precision in quantum operations.

3. **Integration of Input and Optimization Parameters:**  
   The L-Gate represents a machine learning gate that combines input parameters with optimization parameters. This integration is crucial for adapting to various quantum learning tasks and achieving desired outcomes.

### L-Gate Structure

The structure of the L-Gate is represented as follows:

### L-Gate Structure

| TP0 | IP0 | H  | TP1 | IP1 | H  | TP2 | IP2 |
|-----|-----|----|-----|-----|----|-----|-----|


- **TP**: Tunable Parameters
- **IP**: Input Parameters
- **H**: Hadamard Gate

The sequence of tunable parameters (TP) and input parameters (IP), interspersed with Hadamard gates (H), facilitates the desired operations, ensuring that the L-Gate functions effectively as a machine learning gate.

### Explanation

- **Tunable Parameters (TP):** These are adjustable parameters that allow the quantum circuit to adapt to specific needs and optimize performance dynamically.

- **Input Parameters (IP):** Parameters that represent the input data, feeding the quantum circuit with the necessary information for processing.

- **Hadamard Gate (H):** The Hadamard gate plays a crucial role in creating superpositions, enabling quantum parallelism and interaction between different states.


## Objective of Training

In this system, a multi-qubit system is always used when training these gates. In this case, each qubit has multiple L-Gates. The gates are trained so that, for a given input, combined with the tuning phases, they produce a well-defined state. Thus, the system learns to associate a specific input with a fixed state of the system.

### Visual Representation of a Multi-Qubit System

```plaintext
     ┌───────┐     ┌───────┐     ┌───────┐     ┌───────┐     ┌───────┐
q_0: | TP₀,₀ | --- | IP₀,₀ | --- |  H₀   | --- | TP₀,₁ | --- | IP₀,₁ | 
     └───────┘     └───────┘     └───────┘     └───────┘     └───────┘
     ┌───────┐     ┌───────┐     ┌───────┐ 
     |  H₀   | --- | TP₀,₂ | --- | IP₀,₂ |
     └───────┘     └───────┘     └───────┘

     ┌───────┐     ┌───────┐     ┌───────┐     ┌───────┐     ┌───────┐
q_1: | TP₁,₀ | --- | IP₁,₀ | --- |  H₁   | --- | TP₁,₁ | --- | IP₁,₁ | 
     └───────┘     └───────┘     └───────┘     └───────┘     └───────┘
     ┌───────┐     ┌───────┐     ┌───────┐ 
     |  H₁   | --- | TP₁,₂ | --- | IP₁,₂ |
     └───────┘     └───────┘     └───────┘

     ┌───────┐     ┌───────┐     ┌───────┐     ┌───────┐     ┌───────┐
q_2: | TP₂,₀ | --- | IP₂,₀ | --- |  H₂   | --- | TP₂,₁ | --- | IP₂,₁ | 
     └───────┘     └───────┘     └───────┘     └───────┘     └───────┘
     ┌───────┐     ┌───────┐     ┌───────┐ 
     |  H₂   | --- | TP₂,₂ | --- | IP₂,₂ |
     └───────┘     └───────┘     └───────┘

```
### Explanation

- **Qubits (`q_0`, `q_1`, `q_2`)**: Each line represents a qubit in the multi-qubit system.
- **TP Gates (`TP₀,₀`, `TP₁,₀`, etc.)**: These are the tunable parameter gates, where the first subscript denotes the qubit, and the second subscript denotes the phase index.
- **IP Gates (`IP₀,₀`, `IP₁,₀`, etc.)**: Input parameter gates, where the data matrices are inputted. The subscripts indicate the qubit and phase.
- **Hadamard Gate (`H₀, H₁, H₂`)**: Used for creating superpositions and facilitating quantum parallelism.

### Training Parameter Matrix (TP)

<div align="center">

<table style="border-collapse: collapse; border: none; text-align: center; font-size: 18px;">
  <tr>
    <td style="border: none;">TP<sub>0,0</sub></td>
    <td style="border: none;">TP<sub>0,1</sub></td>
    <td style="border: none;">TP<sub>0,2</sub></td>
  </tr>
  <tr>
    <td style="border: none;">TP<sub>1,0</sub></td>
    <td style="border: none;">TP<sub>1,1</sub></td>
    <td style="border: none;">TP<sub>1,2</sub></td>
  </tr>
  <tr>
    <td style="border: none;">TP<sub>2,0</sub></td>
    <td style="border: none;">TP<sub>2,1</sub></td>
    <td style="border: none;">TP<sub>2,2</sub></td>
  </tr>
</table>

</div>

### Input Parameter Matrix (IP)

<div align="center">

<table style="border-collapse: collapse; border: none; text-align: center; font-size: 18px;">
  <tr>
    <td style="border: none;">IP<sub>0,0</sub></td>
    <td style="border: none;">IP<sub>0,1</sub></td>
    <td style="border: none;">IP<sub>0,2</sub></td>
  </tr>
  <tr>
    <td style="border: none;">IP<sub>1,0</sub></td>
    <td style="border: none;">IP<sub>1,1</sub></td>
    <td style="border: none;">IP<sub>1,2</sub></td>
  </tr>
  <tr>
    <td style="border: none;">IP<sub>2,0</sub></td>
    <td style="border: none;">IP<sub>2,1</sub></td>
    <td style="border: none;">IP<sub>2,2</sub></td>
  </tr>
</table>

</div>

### Process Description

During the training of these gates, data in the form of matrices is applied to the IP gates. The TP gates are then optimized to achieve the desired state transformation. The input matrix feeds specific values into the IP gates, which correspond to the data that the system processes. The training matrix allows the TP gates to adjust their parameters to align with the desired outcomes, effectively learning how to map inputs to specific quantum states.


## Optimizer Classes in the Quantum Circuit System

### 1. Optimizer Class

- **Class Name**: `Optimizer`
- **Purpose**: Serves as a base class for optimizing quantum circuits by adjusting the phases of quantum gates to maximize the probability of achieving a target state.
- **Key Features**:
  - **Loss Function**: Calculates the loss as the negative probability of the target state to maximize its likelihood.
  - **Optimization Loop**: Iteratively updates phases using random perturbations to minimize the loss.
  - **Phase Updates**: Generates small random changes to explore new states.

#### Methods:

- `loss_function(self, counts)`:  
  Calculates the loss based on the probability of reaching the target state.

- `optimize(self)`:  
  Optimizes the training phases to minimize the loss by iterating over several attempts.

- `update_phases(self, current_phases)`:  
  Updates the phases by adding random changes to the current training phases.

- `get_distribution(self, counts)`:  
  Retrieves a sorted probability distribution from the counts obtained from the circuit.

- `plot_distribution(self, counts, title)`:  
  Plots a histogram of the state distribution using matplotlib.

### 2. OptimizerWithMomentum Class

- **Class Name**: `OptimizerWithMomentum`
- **Purpose**: Enhances the basic `Optimizer` class by incorporating momentum to improve convergence speed and stability during phase updates.
- **Key Features**:
  - **Momentum**: Introduces a velocity term to smooth updates and prevent oscillations.
  - **Velocity Calculation**: Utilizes past gradients to refine phase adjustments.

#### Methods:

- `update_phases(self, current_phases)`:  
  Uses momentum to update phases, enhancing the gradient descent process.

### 3. AdamOptimizer Class

- **Class Name**: `AdamOptimizer`
- **Purpose**: Implements the Adam optimization algorithm, which adapts learning rates for each parameter, making it suitable for non-stationary objectives.
- **Key Features**:
  - **Adaptive Learning Rates**: Adjusts learning rates using moving averages of first and second moments of the gradients.
  - **Parameters**: 
    - `beta1` and `beta2`: Control the decay rates of the moving averages.
    - `epsilon`: A small constant to prevent division by zero.

#### Methods:

- `update_phases(self, current_phases)`:  
  Computes adaptive updates for the phases using Adam's algorithm, considering moving averages.

### 4. GeneticOptimizer Class

- **Class Name**: `GeneticOptimizer`
- **Purpose**: Utilizes a genetic algorithm to explore the phase space by simulating natural selection processes.
- **Key Features**:
  - **Population-Based Approach**: Maintains a population of solutions (training phases) that evolve over iterations.
  - **Crossover and Mutation**: Combines solutions and introduces mutations to explore new possibilities.
  - **Parameters**: 
    - `population_size`: Number of individuals in the population.
    - `mutation_rate`: Probability of mutation in each individual.

#### Methods:

- `_initialize_population(self)`:  
  Initializes the population with random variations of the training phases.

- `optimize(self)`:  
  Performs optimization using the genetic algorithm, involving selection, crossover, and mutation.

- `_crossover_and_mutate(self, parents)`:  
  Creates the next generation through crossover and mutation.

- `_crossover(self, parent1, parent2)`:  
  Performs single-point crossover between two parent solutions.

- `_mutate(self, individual)`:  
  Applies random mutations to an individual.

### 5. PSOOptimizer Class (Particle Swarm Optimization)

- **Class Name**: `PSOOptimizer`
- **Purpose**: Implements Particle Swarm Optimization (PSO), a population-based algorithm inspired by social behavior in nature.
- **Key Features**:
  - **Particle Representation**: Models each potential solution as a particle in the search space.
  - **Inertia, Cognitive, and Social Components**: Balances exploration and exploitation using these three components.
  - **Parameters**: 
    - `num_particles`: Number of particles in the swarm.
    - `inertia`, `cognitive`, `social`: Weights for the respective components.

#### Methods:

- `_initialize_particles(self)`:  
  Initializes particles with random variations around the training phases.

- `optimize(self)`:  
  Performs the optimization by updating particle velocities and positions iteratively.

### 6. BayesianOptimizer Class

- **Class Name**: `BayesianOptimizer`
- **Purpose**: Uses Bayesian optimization to tune the phases, particularly useful for expensive-to-evaluate functions.
- **Key Features**:
  - **Gaussian Process (GP)**: Models the objective function and updates it based on observations.
  - **`gp_minimize()`**: Executes Bayesian optimization over the defined bounds.

#### Methods:

- `optimize(self)`:  
  Uses Bayesian optimization to find the optimal training phases by minimizing the objective function over specified bounds.

### 7. SimulatedAnnealingOptimizer Class

- **Class Name**: `SimulatedAnnealingOptimizer`
- **Purpose**: Simulates the annealing process to find the global minimum by gradually reducing temperature.
- **Key Features**:
  - **Temperature Control**: Accepts worse solutions with decreasing probability as temperature lowers.
  - **Cooling Rate**: Determines the rate at which the temperature decreases.

#### Methods:

- `optimize(self)`:  
  Optimizes the phases by exploring the solution space with a controlled cooling schedule.

### 8. QuantumNaturalGradientOptimizer Class

- **Class Name**: `QuantumNaturalGradientOptimizer`
- **Purpose**: Utilizes Quantum Natural Gradient (QNG) to update parameters, accounting for the parameter space's geometry.
- **Key Features**:
  - **Fisher Information Matrix**: Uses the matrix to calculate natural gradient steps.
  - **QNG Step Calculation**: Applies the natural gradient to update phases.

#### Methods:

- `update_phases(self, current_phases)`:  
  Computes updates using the quantum natural gradient, considering the Fisher information matrix.


## Public Collaboration

We welcome and encourage public collaboration on this GitHub project. If you're interested in contributing, there are several ways you can get involved:

### 1. **Contact the Team**

If you have questions or suggestions, feel free to reach out to our team at any time. We're eager to hear your thoughts and are open to discussions about potential improvements or new ideas for the project.

### 2. **Explore the Repository**

Dive into the repository to understand the current state of the project. You'll find detailed documentation and examples that will help you get up to speed quickly. We recommend checking out the following resources:

- **README**: Provides an overview of the project and guides you on getting started.
- **Documentation**: Offers detailed information about the project's architecture, modules, and usage.

### 3. **Pick a Task or Feature**

Identify tasks or features that interest you and feel free to take them on. You can find a list of tasks or features in our [issue tracker](#), where we regularly update the project's current needs and priorities. Here’s how you can proceed:

- **Comment on an Issue**: Let us know which task you're interested in by commenting on the relevant issue.
- **Create a Pull Request**: Once you've made changes, submit a pull request. Our team will review your contributions and provide feedback.

### 4. **Join the Community**

Engage with other contributors in the project's discussions. This is a great way to exchange ideas, ask questions, and collaborate on solutions. You can:

- **Participate in Discussions**: Join ongoing conversations in the [discussion board](#).
- **Collaborate on Projects**: Work with others to tackle complex issues and develop new features.

### 5. **Contributing Guidelines**

We have a set of guidelines to help you contribute effectively:

- **Fork the Repository**: Start by forking the repository to your own GitHub account.
- **Create a Branch**: Make your changes in a dedicated branch.
- **Commit Your Changes**: Use clear and concise commit messages.
- **Push and Pull Request**: Push your changes and create a pull request for review.

By contributing, you become a part of our community, helping us improve and expand the project. We value every contribution and look forward to collaborating with you!

If you're ready to start, head over to our [contributing guide](#) for detailed instructions. Together, we can make this project even better!


