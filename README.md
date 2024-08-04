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

Beim Training dieser Gates in diesem System wird immer ein Multi-Qubit-System verwendet. In diesem Fall hat jedes Qubit mehrere dieser L-Gates. Die Gates werden so trainiert, dass sie bei einem bestimmten Input, kombiniert mit den Tuning-Phasen, einen klaren Zustand zeigen. Somit lernt das System, einem Input einen festen Zustand des Systems zuzuordnen.

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

### Explanation

- **Qubits (`q_0`, `q_1`, `q_2`)**: Each line represents a qubit in the multi-qubit system.
- **TP Gates (`TP₀,₀`, `TP₁,₀`, etc.)**: These are the tunable parameter gates, where the first subscript denotes the qubit, and the second subscript denotes the phase index.
- **IP Gates (`IP₀,₀`, `IP₁,₀`, etc.)**: Input parameter gates, where the data matrices are inputted. The subscripts indicate the qubit and phase.
- **Hadamard Gate (`H₀, H₁, H₂`)**: Used for creating superpositions and facilitating quantum parallelism.

### Input and Training Matrices

```plaintext
Input Matrix (IP):

| IP₀,₀ | IP₀,₁ | IP₀,₂ |
|-------|-------|-------|
| a₀,₀  | a₀,₁  | a₀,₂  |
| a₁,₀  | a₁,₁  | a₁,₂  |
| a₂,₀  | a₂,₁  | a₂,₂  |

Training Matrix (TP):

| TP₀,₀ | TP₀,₁ | TP₀,₂ |
|-------|-------|-------|
| b₀,₀  | b₀,₁  | b₀,₂  |
| b₁,₀  | b₁,₁  | b₁,₂  |
| b₂,₀  | b₂,₁  | b₂,₂  |

### Process Description

During the training of these gates, data in the form of matrices is applied to the IP gates. The TP gates are then optimized to achieve the desired state transformation. The input matrix feeds specific values into the IP gates, which correspond to the data that the system processes. The training matrix allows the TP gates to adjust their parameters to align with the desired outcomes, effectively learning how to map inputs to specific quantum states.


## Optimization Methods

An overview of various optimization methods...

## Public Collaboration

Information on public collaboration efforts...
