import json
import matplotlib.pyplot as plt
import seaborn as sns
from visual import Visual  # Import the Visual class

def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

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
    if counts:
        plt.bar(counts.keys(), counts.values())
    else:
        plt.text(0.5, 0.5, "No data available", fontsize=12, ha='center')
    plt.xlabel("Quantum States")
    plt.ylabel("Counts")
    plt.title("Probability Distribution of Quantum States")
    plt.savefig(filename)
    plt.close()

def main():
    # Load the data from the execution phase
    data = load_data('results.json')

    # Extract relevant data
    input_matrix = data["input_matrix"]
    training_matrix = data["training_matrix"]
    optimized_phases = data["optimized_phases"]
    losses = data["losses"]
    counts = data["counts"]

    # Plot the initial training matrix as a heatmap
    initial_heatmap_path = "initial_training_matrix.png"
    plot_heatmap(training_matrix, "Initial Training Matrix", initial_heatmap_path)

    # Plot the optimized training matrix as a heatmap
    final_heatmap_path = "final_training_matrix.png"
    plot_heatmap(optimized_phases, "Optimized Training Matrix", final_heatmap_path)

    # Plot the loss curve
    loss_plot_path = "loss_curve.png"
    plot_loss_curve(losses, loss_plot_path)

    # Plot histogram of the quantum circuit result
    histogram_path = "quantum_histogram.png"
    plot_histogram(counts, histogram_path)

    # Create the PDF report
    print("Erstelle PDF-Bericht...")
    visual = Visual()
    visual.generate_report(
        "quantum_circuit.png", 
        input_matrix, 
        training_matrix, 
        initial_heatmap_path, 
        final_heatmap_path, 
        loss_plot_path, 
        histogram_path
    )

if __name__ == '__main__':
    main()
