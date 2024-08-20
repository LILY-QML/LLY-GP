import json
import numpy as np
from module.circuit import Circuit
from module.optimizer import AdamOptimizer

def load_data(file_path):
    """Lädt die Daten aus einer JSON-Datei."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Datei {file_path} nicht gefunden.")
        return {}

def save_data(file_path, data):
    """Speichert die Daten in einer JSON-Datei."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def generate_random_matrix(rows, cols):
    """Erzeugt eine zufällige Matrix der Größe rows x cols."""
    return np.random.rand(rows, cols)

def save_training_matrix_history(matrix_history, iteration):
    """Speichert die Trainingsmatrix nach jeder Iteration."""
    file_path = f'var/training_matrix_history_iteration_{iteration}.json'
    save_data(file_path, {"training_matrix": matrix_history})

def main():
    # Schritt 1: Laden der Daten
    data_path = 'var/data.json'
    data = load_data(data_path)

    # Falls 'input_matrix' oder 'training_matrix' nicht vorhanden sind, generiere sie
    if 'input_matrix' not in data:
        print("'input_matrix' fehlt, generiere eine neue Matrix...")
        data['input_matrix'] = generate_random_matrix(25, 3).tolist()

    if 'training_matrix' not in data:
        print("'training_matrix' fehlt, generiere eine neue Matrix...")
        data['training_matrix'] = generate_random_matrix(25, 3).tolist()

    # Speichern der aktualisierten Daten
    save_data(data_path, data)

    word = data.get('words', ["dummy"])[0]  # Fallback zu 'dummy' falls 'words' nicht vorhanden
    input_matrix = np.array(data['input_matrix'])
    training_matrix = np.array(data['training_matrix'])

    # Speichern der initialen Trainingsmatrix
    initial_training_matrix = training_matrix.copy()
    save_training_matrix_history(initial_training_matrix.tolist(), "initial")

    # Schritt 2: Quantenkreis initialisieren und ausführen
    qubits = len(input_matrix)
    circuit = Circuit(qubits=qubits, depth=3, training_phases=training_matrix, activation_phases=input_matrix, shots=1024)
    
    print("Führe den initialen Quantenkreis aus...")
    result = circuit.run()
    
    # Extrahiere die Zählungen aus dem Result-Objekt
    initial_counts = result.get_counts(circuit.circuit)

    # Debug-Ausgabe, um den Inhalt von initial_counts zu überprüfen
    print("Initiale Zählungen:", initial_counts)

    # Bestimmen des Zielzustands
    target_state = max(initial_counts, key=initial_counts.get)
    print(f"Zielzustand: {target_state}")

    # Schritt 4: Optimierung der Trainingsphasen
    optimizer = AdamOptimizer(circuit=circuit, target_state=target_state, learning_rate=0.01, max_iterations=100)
    optimized_phases, losses = optimizer.optimize()  # Korrekte Methode


    for iteration in range(optimizer.max_iterations):
        # Optimiere die Trainingsphasen für eine Iteration
        optimized_phases, loss = optimizer.step()

        # Speichern der optimierten Trainingsphasen nach jeder Iteration
        save_training_matrix_history(optimized_phases.tolist(), iteration)
        
        # Überprüfen, ob sich die Trainingsphasen geändert haben
        if not np.array_equal(optimized_phases, training_matrix):
            print(f"Iteration {iteration}: Trainingsphasen wurden geändert.")
        else:
            print(f"Iteration {iteration}: Keine Änderung in den Trainingsphasen.")

        # Setze die aktualisierten Trainingsphasen für den nächsten Schritt
        training_matrix = optimized_phases.copy()
        circuit.update_training_phases(training_matrix)

    # Schritt 5: Ergebnis nach der Optimierung speichern
    result_after_optimization = circuit.run()
    counts_after_optimization = result_after_optimization.get_counts(circuit.circuit)
    
    results = {
        "word": word,
        "input_matrix": input_matrix.tolist(),
        "training_matrix": training_matrix.tolist(),
        "optimized_phases": optimized_phases.tolist(),
        "initial_training_matrix": initial_training_matrix.tolist(),
        "losses": optimizer.losses,
        "initial_counts": initial_counts,
        "counts_after_optimization": counts_after_optimization
    }

    save_data('var/results.json', results)

if __name__ == '__main__':
    main()
