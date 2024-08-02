import json
import os
import numpy as np
from module.circuit import Circuit
from module.tokenizer import Tokenizer

def load_data(file_path):
    """Lädt die Daten aus einer JSON-Datei."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def generate_random_matrix(rows, cols):
    """Erzeugt eine zufällige Matrix der Größe rows x cols."""
    return np.random.rand(rows, cols)

def main():
    # Schritt 1: Laden der Daten aus der JSON-Datei
    print("Lade JSON-Daten...")
    file_path = os.path.join('var', 'data.json')
    data = load_data(file_path)

    # Extrahieren der Wörter aus den Daten
    words = data['words']

    # Initialisieren des Tokenizers
    tokenizer = Tokenizer()

    # Wählen Sie ein Wort zum Tokenisieren
    print("Tokenisiere das erste Wort...")
    word = words[0]  # Annahme: Wir verwenden das erste Wort
    token = tokenizer.tokenize(word)

    # Erzeugen der Input-Matrix aus dem Token
    input_matrix = np.array(token)
    print(f"Input-Matrix für '{word}' erstellt.")

    # Erzeugen der zufälligen Trainingsmatrix der Größe 20x3
    print("Erzeuge zufällige Trainingsmatrix...")
    training_matrix = generate_random_matrix(20, 3)
    print("Trainingsmatrix erstellt.")

    # Initialisieren und Erstellen des Schaltkreises
    print("Erstelle und konfiguriere den Quantenkreis...")
    qubits = 20
    circuit = Circuit(input_matrix, training_matrix, qubits)
    circuit.create_circuit()
    circuit.measure()  # Messpunkte hinzufügen
    print("Quantenkreis erstellt und gemessen.")

    # Ausführen des Schaltkreises und Ergebnisse anzeigen
    print("Führe den Quantenkreis aus...")
    counts = circuit.complete_run()
    print("\nEndgültige Zählungen aller Zustände:")
    for state, count in counts.items():
        print(f"{state}: {count}")

if __name__ == '__main__':
    main()
