import json
import os
import numpy as np
from module.circuit import Circuit
from module.tokenizer import Tokenizer
from module.optimizer import AdamOptimizer
from module.visual import Visual  # Import the Visual class
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def generate_random_matrix(rows, cols):
    return np.random.rand(rows, cols)

def plot_heatmap(matrix, title, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap="viridis")
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def plot_loss_curve(losses, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss During Training")
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_histogram(counts, filename):
    plt.figure(figsize=(10, 6))
    plt.bar(counts.keys(), counts.values())
    plt.xlabel("Quantum States")
    plt.ylabel("Counts")
    plt.title("Probability Distribution of Quantum States")
    plt.savefig(filename)
    plt.close()

def main():
    # Step 1: Load data from the JSON file
    print("Lade JSON-Daten...")
    file_path = os.path.join('var', 'data.json')
    data = load_data(file_path)

    # Extract words from the data
    words = data['words']

    # Initialize the tokenizer
    tokenizer = Tokenizer()

    # Tokenize the first word
    print("Tokenisiere das erste Wort...")
    word = words[0]
    token = tokenizer.tokenize(word)

    # Create input matrix from the token
    input_matrix = np.array(token)
    print(f"Input-Matrix für '{word}' erstellt.")

    # Generate random training matrix
    print("Erzeuge zufällige Trainingsmatrix...")
    training_matrix = generate_random_matrix(25, 3)
    print("Trainingsmatrix erstellt.")

    # Initialize and create the quantum circuit
    print("Erstelle und konfiguriere den Quantenkreis...")
    qubits = 25
    depth = 3
    shots = 1024
    circuit = Circuit(qubits, depth, training_matrix, input_matrix, shots)

    # Draw the quantum circuit as an image
    print("Erstelle Bild des Quantenkreises...")
    circuit_image_path = "quantum_circuit.png"
    circuit_drawer(circuit.circuit, output='mpl', filename=circuit_image_path)

    # Plot the initial training matrix as a heatmap
    print("Erstelle Heatmap der Trainingsmatrix...")
    initial_heatmap_path = "initial_training_matrix.png"
    plot_heatmap(training_matrix, "Initial Training Matrix", initial_heatmap_path)

    # Step 2: Run the optimizer
    print("Starte Optimierung...")
    target_state = '00'
    optimizer = AdamOptimizer(
        circuit=circuit,
        target_state=target_state,
        learning_rate=0.01,
        max_iterations=100
    )
    optimized_phases, losses = optimizer.optimize()

    # Plot the loss curve
    print("Erstelle Loss-Plot...")
    loss_plot_path = "loss_curve.png"
    plot_loss_curve(losses, loss_plot_path)

    # After optimization, plot the new training matrix
    print("Erstelle Heatmap der optimierten Trainingsmatrix...")
    final_heatmap_path = "final_training_matrix.png"
    plot_heatmap(circuit.training_phases, "Optimized Training Matrix", final_heatmap_path)

    # Run the quantum circuit and get the counts
    print("Führe den Quantenkreis aus und erstelle Histogramm...")
    counts = circuit.run()
    histogram_path = "quantum_histogram.png"
    plot_histogram(counts, histogram_path)

    # Create the PDF report
    print("Erstelle PDF-Bericht...")
    visual = Visual()
    visual.generate_report(
        circuit_image_path, 
        input_matrix, 
        training_matrix, 
        initial_heatmap_path, 
        final_heatmap_path, 
        loss_plot_path, 
        histogram_path
    )

if __name__ == '__main__':
    main()
