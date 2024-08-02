from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer  # Verwenden Sie AerSimulator anstelle von QasmSimulator
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Importieren von tqdm für die Fortschrittsleiste

class Circuit:
    def __init__(self, input_matrix, training_matrix, qubits):
        self.input_matrix = input_matrix
        self.training_matrix = training_matrix
        self.qubits = qubits
        self.circuit = QuantumCircuit(qubits, qubits)  # Zwei Register: Quantum und Classical

    def create_circuit(self):
        # Anwenden von zwei L-Gate-Mustern auf jedes Qubit
        for qubit in range(self.qubits):
            self.apply_l_gate(qubit, self.input_matrix, self.training_matrix)
            self.apply_l_gate(qubit, self.input_matrix, self.training_matrix)

    def apply_l_gate(self, qubit, input_matrix, training_matrix):
        # Anwenden eines L-Gate-Musters (TP-IP-H-TP-IP-H-TP-IP) auf das spezifizierte Qubit
        for phase_idx in range(3):
            # Training Phase Gate (Phasengatter)
            training_phase = training_matrix[qubit][phase_idx]
            self.circuit.p(training_phase, qubit)  # Phasengatter

            # Input Phase Gate (Phasengatter)
            input_phase = input_matrix[qubit][phase_idx]
            self.circuit.p(input_phase, qubit)  # Phasengatter

            # Hadamard Gate nur nach TP-IP
            if phase_idx < 2:  # Das Hadamard-Gate nur für die ersten zwei Paare von TP-IP
                self.circuit.h(qubit)

    def measure(self):
        # Hinzufügen von Messungen an jedes Qubit
        self.circuit.barrier()  # Optional: Füge eine Barriere hinzu, um Messungen von Operationen zu trennen
        for qubit in range(self.qubits):
            self.circuit.measure(qubit, qubit)

    def run(self, shots=1024):
        # Ausführen des Schaltkreises mit einer bestimmten Anzahl von Messungen
        simulator = Aer.get_backend('aer_simulator')  # Verwenden Sie AerSimulator anstelle von QasmSimulator
        transpiled_circuit = transpile(self.circuit, simulator)
        job = simulator.run(transpiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(transpiled_circuit)
        return counts

    def complete_run(self):
        # Führe den Schaltkreis so lange aus, bis jeder Zustand mindestens einmal gemessen wurde
        simulator = Aer.get_backend('aer_simulator')  # Verwenden Sie AerSimulator anstelle von QasmSimulator
        transpiled_circuit = transpile(self.circuit, simulator)

        # Ermitteln der Anzahl der möglichen Zustände
        num_possible_states = 2 ** self.qubits
        all_states = set(format(i, f'0{self.qubits}b') for i in range(num_possible_states))

        measured_states = set()
        total_counts = {}

        # Verwenden von tqdm für die Fortschrittsanzeige
        with tqdm(total=num_possible_states, desc="Messprozess", unit="Zustand") as pbar:
            while measured_states != all_states:
                job = simulator.run(transpiled_circuit, shots=1024)
                result = job.result()
                counts = result.get_counts(transpiled_circuit)
                
                # Aggregiere die Messungen
                for state, count in counts.items():
                    if state not in total_counts:
                        total_counts[state] = count
                    else:
                        total_counts[state] += count

                    # Füge den Zustand zu den gemessenen Zuständen hinzu
                    if state not in measured_states:
                        measured_states.add(state)
                        pbar.update(1)  # Fortschrittsanzeige aktualisieren

                # Fortschrittsanzeige aktualisieren
                pbar.set_postfix({
                    'Gemessen': len(measured_states),
                    'Verbleibend': num_possible_states - len(measured_states),
                    'Fortschritt': f"{len(measured_states) / num_possible_states * 100:.2f}%"
                })

        print("\nAlle Zustände wurden mindestens einmal gemessen.")
        return total_counts

    def draw(self):
        # Zeichnen des Quantenkreises
        print("Zeichne den Quantenkreis:")
        circuit_diagram = self.circuit.draw(output='text')  # Ausgabe als Text
        print(circuit_diagram)
        self.circuit.draw(output='mpl')  # Optional: Matplotlib-Ausgabe
        plt.show()  # Zeigt die Matplotlib-Darstellung an
