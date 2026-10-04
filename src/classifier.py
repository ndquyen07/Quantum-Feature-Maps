import numpy as np

from src.shot_simulation import create_sampler, batch_inversion_test_overlaps


class QuantumClassifier:
    def __init__(self, sampler=None, shots=1024):
        self.sampler = sampler if sampler is not None else create_sampler(shots)
        self.class_samples = None
        self.class_representatives = None
        self.circuit = None
        self.theta_params = None

    def fit(self, class_samples, class_representatives=None, circuit=None, theta=None):
        # Backward-compatible: fit(representatives, circuit, theta)
        if class_representatives is not None and circuit is not None and theta is not None:
            self.class_samples = class_samples
            self.class_representatives = class_representatives
            self.circuit = circuit
            self.theta_params = theta
        else:
            representatives = class_samples
            self.circuit = class_representatives
            self.theta_params = circuit
            self.class_representatives = representatives
            self.class_samples = [[rep] for rep in representatives]

    def _circuit_with_theta(self):
        n_features = self.class_samples[0].shape[1]
        data_params = list(self.circuit.parameters)[:n_features]
        theta_params_list = list(self.circuit.parameters)[n_features:]
        theta_dict = {p: float(v) for p, v in zip(theta_params_list, self.theta_params)}
        return self.circuit.assign_parameters(theta_dict), data_params

    def _score_sample(self, x_test, class_idx, metric):
        circuit_with_theta, data_params = self._circuit_with_theta()
        class_samples = self.class_samples[class_idx]

        pairs = [(x_test, train_sample) for train_sample in class_samples]
        overlaps = batch_inversion_test_overlaps(circuit_with_theta, data_params, pairs, self.sampler)
        mean_overlap = float(np.mean(overlaps))

        if metric == 'overlap':
            return mean_overlap
        if metric == 'hilbert_schmidt':
            return mean_overlap - 1.0
        return -float(np.sqrt(max(0.0, 1.0 - mean_overlap)))

    def predict(self, X_test, metric='trace_distance'):
        predictions = []
        for x_test in X_test:
            scores = [self._score_sample(x_test, c, metric) for c in range(len(self.class_samples))]
            predictions.append(int(np.argmax(scores)))
        return np.array(predictions)

    def score(self, X_test, y_test, metric='trace_distance'):
        return float(np.mean(self.predict(X_test, metric=metric) == y_test))


class ClassicalClassifier:
    @staticmethod
    def evaluate_model(kernel_train, kernel_test, y_train, y_test):
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV

        c_values = [0.001, 0.01, 0.1, 1.0, 10.0, 20.0, 30.0, 40.0, 50.0,
                    60.0, 70.0, 80.0, 90.0, 100.0, 200.0, 300.0, 400.0,
                    500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
        param_grid = {'C': c_values}

        svc = SVC(kernel='precomputed')
        grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(kernel_train, y_train)

        best_model = grid_search.best_estimator_
        best_c = grid_search.best_params_['C']
        best_score = grid_search.best_score_
        test_acc = best_model.score(kernel_test, y_test)

        return best_score, test_acc, best_model, best_c

    @staticmethod
    def calculate_accuracy_fix(kernel_train, kernel_val, y_train, y_val):
        from sklearn.svm import SVC

        svc = SVC(kernel='precomputed', C=1.0)
        svc.fit(kernel_train, y_train)

        return svc.score(kernel_train, y_train), svc.score(kernel_val, y_val)
