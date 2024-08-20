from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile

class Circuit:
    def __init__(self, qubits, depth, training_phases, activation_phases, shots):
        self.qubits = qubits
        self.depth = depth
        self.training_phases = training_phases  # Training phases matrix
        self.activation_phases = activation_phases  # Activation phases matrix
        self.shots = shots  # Number of measurements
        self.circuit = QuantumCircuit(qubits, qubits)
        self.simulation_result = None  # Store the simulation result
        self.initialize_gates()
        self.measure()  # Set up measurement as part of initialization

    def initialize_gates(self):
        """Initialize the gates using training and activation phases."""
        
        # Validate matrix dimensions
        if len(self.training_phases) != self.qubits or len(self.activation_phases) != self.qubits:
            raise ValueError(f"Training and activation phases must each have {self.qubits} rows.")

        if any(len(row) != 3 for row in self.training_phases) or any(len(row) != 3 for row in self.activation_phases):
            raise ValueError("Each phase entry must have 3 columns (3 phases per qubit).")

        # Apply L-Gates for each qubit
        for qubit in range(self.qubits):
            self.apply_l_gate(qubit)

    def apply_l_gate(self, qubit):
        """Apply L-Gate sequence using training and activation phases."""
        # Correct order:
        # Phase, Phase, Hadamard, Phase, Phase, Hadamard, Phase, Phase
        
        # Apply the first two Phase-Gates
        self.circuit.p(self.training_phases[qubit][0], qubit)
        self.circuit.p(self.activation_phases[qubit][0], qubit)
        
        # Apply Hadamard after the first two Phase-Gates
        self.circuit.h(qubit)
        
        # Apply the next two Phase-Gates
        self.circuit.p(self.training_phases[qubit][1], qubit)
        self.circuit.p(self.activation_phases[qubit][1], qubit)
        
        # Apply Hadamard after the next two Phase-Gates
        self.circuit.h(qubit)
        
        # Apply the final two Phase-Gates
        self.circuit.p(self.training_phases[qubit][2], qubit)
        self.circuit.p(self.activation_phases[qubit][2], qubit)

    def measure(self):
        """Add measurement to all qubits."""
        self.circuit.measure(range(self.qubits), range(self.qubits))

    def run(self):
        """Run the quantum circuit simulation and return the result."""
        simulator = Aer.get_backend('aer_simulator')  # Updated to use AerSimulator
        compiled_circuit = transpile(self.circuit, simulator)

        # Directly run the compiled circuit on the simulator
        self.simulation_result = simulator.run(compiled_circuit, shots=self.shots).result()

        # Debugging: Print the simulation result for inspection
        print("Simulation Result:", self.simulation_result)

        return self.simulation_result  # Return the result

    def get_counts(self):
        """Return the counts from the last run simulation."""
        if self.simulation_result is not None:
            try:
                return self.simulation_result.get_counts(self.circuit)
            except Exception as e:
                print("Error in getting counts:", e)
                print("Simulation Result Data:", self.simulation_result.data())
                raise
        else:
            raise RuntimeError("The circuit has not been run yet.")

    def __repr__(self):
        return self.circuit.draw(output='text').__str__()
