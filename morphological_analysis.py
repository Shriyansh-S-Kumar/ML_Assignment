import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import json
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import os
from scipy.ndimage import label, binary_dilation, binary_erosion

SE_PATTERNS = {
    'SE1': np.array([[1, 0, 1],
                     [0, 1, 0],
                     [1, 0, 1]]),
    'SE2': np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]]),
    'SE3': np.array([[0, 1, 0],
                     [1, 0, 1],
                     [0, 1, 0]]),
    'SE4': np.array([[1, 1, 1],
                     [1, 0, 1],
                     [1, 1, 1]]),
    'SE5': np.array([[0, 0, 1],
                     [0, 0, 1],
                     [0, 0, 1]]),
    'SE6': np.array([[1, 0, 0],
                     [1, 0, 0],
                     [1, 0, 0]]),
    'SE7': np.array([[1, 1, 1],
                     [0, 0, 0],
                     [0, 0, 0]]),
    'SE8': np.array([[0, 0, 0],
                     [0, 0, 0],
                     [1, 1, 1]])
}

@dataclass
class TransformationSequence:
    operations: List[Tuple[str, str]]

class MorphologicalProcessor:
    def __init__(self):
        self.se_patterns = SE_PATTERNS
    
    def pad_matrix(self, matrix: np.ndarray, padding: int = 1) -> np.ndarray:
        return np.pad(matrix, pad_width=padding, mode='constant', constant_values=0)
    
    def erode(self, matrix: np.ndarray, se_pattern: str) -> np.ndarray:
        kernel = self.se_patterns[se_pattern]
        padded = self.pad_matrix(matrix)
        result = np.zeros_like(matrix)
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                window = padded[i:i+3, j:j+3]
                result[i, j] = int(np.all(window[kernel == 1] == 1))
        
        return result
    
    def dilate(self, matrix: np.ndarray, se_pattern: str) -> np.ndarray:
        kernel = self.se_patterns[se_pattern]
        padded = self.pad_matrix(matrix)
        result = np.zeros_like(matrix)
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                window = padded[i:i+3, j:j+3]
                result[i, j] = int(np.any(window[kernel == 1] == 1))
        
        return result
    
    def apply_sequence(self, matrix: np.ndarray, sequence: TransformationSequence) -> List[np.ndarray]:
        results = [matrix]
        current = matrix.copy()
        
        for op_type, se_pattern in sequence.operations:
            if op_type == "Dilation":
                current = self.dilate(current, se_pattern)
            else: 
                current = self.erode(current, se_pattern)
            results.append(current.copy())
            
        return results

class ChangeDetector:
    def detect_pixel_changes(self, sequence_results: List[np.ndarray]) -> np.ndarray:
        changes = np.zeros_like(sequence_results[0], dtype=bool)
        
        for i in range(1, len(sequence_results)):
            changes |= (sequence_results[i] != sequence_results[i-1])
            
        return changes
    
    def all_pixels_change(self, sequence_results: List[np.ndarray]) -> bool:
        changes = self.detect_pixel_changes(sequence_results)
        return np.all(changes)

class FeatureExtractor:
    def extract_features(self, matrix: np.ndarray) -> Dict:
        features = {
            'density': np.mean(matrix),
            'edge_density': self._calculate_edge_density(matrix),
            'connected_components': self._count_connected_components(matrix),
            'largest_component_size': self._largest_component_size(matrix),
            'symmetry_score': self._calculate_symmetry(matrix),
            'cluster_count': self._count_clusters(matrix)
        }
        return features
    
    def _calculate_edge_density(self, matrix: np.ndarray) -> float:
        edges_h = np.sum(np.abs(np.diff(matrix, axis=1)))
        edges_v = np.sum(np.abs(np.diff(matrix, axis=0)))
        total_possible = 2 * matrix.size - matrix.shape[0] - matrix.shape[1]
        return (edges_h + edges_v) / total_possible if total_possible > 0 else 0
    
    def _count_connected_components(self, matrix: np.ndarray) -> int:
        labeled, num_features = label(matrix)
        return num_features
    
    def _largest_component_size(self, matrix: np.ndarray) -> int:
        labeled, _ = label(matrix)
        if labeled.max() == 0:
            return 0
        return max(np.bincount(labeled.ravel())[1:])
    
    def _calculate_symmetry(self, matrix: np.ndarray) -> float:
        h_sym = np.mean(matrix == np.fliplr(matrix))
        v_sym = np.mean(matrix == np.flipud(matrix))
        return (h_sym + v_sym) / 2
    
    def _count_clusters(self, matrix: np.ndarray) -> int:
        labeled, num_features = label(matrix)
        return num_features

class MorphologicalPredictor:
    def __init__(self):
        self.processor = MorphologicalProcessor()
        self.detector = ChangeDetector()
        self.feature_extractor = FeatureExtractor()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def prepare_data(self, data_folder: str) -> Tuple[List[Dict], List[bool]]:
        X = []
        y = []
        
        print(f"Looking for files in: {os.path.abspath(data_folder)}")
        
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"Data folder '{data_folder}' does not exist")
        
        for filename in os.listdir(data_folder):
            if filename.startswith('Task') and filename.endswith('.json'):
                task_path = os.path.join(data_folder, filename)
                solution_filename = filename.replace('.json', '_soln.txt')
                solution_path = os.path.join(data_folder, solution_filename)
                
                print(f"Processing task file: {task_path}")
                print(f"Looking for solution file: {solution_path}")
                
                if not os.path.exists(solution_path):
                    print(f"Warning: Solution file not found: {solution_path}")
                    continue
                    
                try:
                    with open(task_path, 'r') as f:
                        task_data = json.load(f)
                    with open(solution_path, 'r') as f:
                        solution_data = f.read()
                        
                    for example in task_data:
                        input_matrix = np.array(example['input'])
                        features = self.feature_extractor.extract_features(input_matrix)
                        X.append(features)
                        
                        sequence = self._parse_sequence(solution_data)
                        results = self.processor.apply_sequence(input_matrix, sequence)
                        changes_all = self.detector.all_pixels_change(results)
                        y.append(changes_all)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue
        
        if not X or not y:
            raise ValueError("No data was loaded. Check if the data files are correctly formatted.")
        
        return X, y
    
    def _parse_sequence(self, solution_data: str) -> TransformationSequence:
        operations = []
        for line in solution_data.strip().split('\n'):
            op_type, se_pattern = line.split(' ')
            operations.append((op_type, se_pattern))
        return TransformationSequence(operations)
    
    def train_with_cross_validation(self, data_folder: str, n_folds: int = 10):
        X, y = self.prepare_data(data_folder)
        X_array = np.array([[x[feature] for feature in sorted(x.keys())] for x in X])
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []
        
        print(f"\nStarting {n_folds}-fold cross-validation...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_array)):
            X_train, X_val = X_array[train_idx], X_array[val_idx]
            y_train, y_val = np.array(y)[train_idx], np.array(y)[val_idx]
            
            self.model.fit(X_train, y_train)
            score = self.model.score(X_val, y_val)
            scores.append(score)
            print(f"Fold {fold + 1}: Accuracy = {score:.4f}")
        
        print(f"\nCross-validation results:")
        print(f"Average accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

        self.model.fit(X_array, y)
        print("\nFinal model trained on all data")
        
    def predict(self, input_matrix: np.ndarray) -> bool:
        features = self.feature_extractor.extract_features(input_matrix)
        feature_vector = np.array([[features[feature] for feature in sorted(features.keys())]])
        return bool(self.model.predict(feature_vector)[0])