import numpy as np
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import brier_score_loss, log_loss
# import matplotlib.pyplot as plt


class VennAbersCalibrator:
    """
    Calibrador Venn-Abers model-agnostic.
    Aceita apenas arrays de probabilidades - não depende de modelos específicos.
    
    Suporta:
    - Classificação binária
    - Multiclasse (one-vs-one ou one-vs-rest)
    """
    
    def __init__(self, multiclass_strategy='one_vs_one'):
        """
        Parameters:
        -----------
        multiclass_strategy : {'one_vs_one', 'one_vs_rest'}
            Estratégia para problemas multiclasse
        """
        self.multiclass_strategy = multiclass_strategy
        self.calibrators = None
        self.classes_ = None
        self.n_classes_ = None
        self.is_binary = None
        
    def fit(self, p_cal, y_cal, precision=None):
        """
        Treina o calibrador com probabilidades e labels.
        
        Parameters:
        -----------
        p_cal : array-like, shape (n_samples, n_classes)
            Probabilidades preditas pelo modelo no conjunto de calibração
            Para binário: pode ser (n_samples,) ou (n_samples, 2)
        y_cal : array-like, shape (n_samples,)
            Labels verdadeiros do conjunto de calibração
        precision : int, optional
            Arredondamento para acelerar computação
        """
        p_cal = np.array(p_cal)
        y_cal = np.array(y_cal)
        
        # Padroniza formato
        if p_cal.ndim == 1:
            # Assume que é probabilidade da classe 1 (binário)
            p_cal_2d = np.zeros((len(p_cal), 2))
            p_cal_2d[:, 1] = p_cal
            p_cal_2d[:, 0] = 1 - p_cal
            p_cal = p_cal_2d
        
        self.classes_ = np.unique(y_cal)
        self.n_classes_ = len(self.classes_)
        self.is_binary = (self.n_classes_ == 2)
        
        if self.is_binary:
            self._fit_binary(p_cal, y_cal, precision)
        elif self.multiclass_strategy == 'one_vs_rest':
            self._fit_ovr(p_cal, y_cal, precision)
        else:  # one_vs_one
            self._fit_ovo(p_cal, y_cal, precision)
        
        return self
    
    def _fit_binary(self, p_cal, y_cal, precision):
        """Treina calibrador binário usando isotonic regression."""
        # Implementação baseada no paper de Vovk et al.
        if precision is not None:
            p_scores = np.round(p_cal[:, 1], precision)
        else:
            p_scores = p_cal[:, 1]
        
        # Ordena por scores
        sorted_idx = np.argsort(p_scores)
        sorted_scores = p_scores[sorted_idx]
        sorted_labels = y_cal[sorted_idx]
        
        # Valores únicos de scores
        unique_scores = np.unique(sorted_scores)
        
        # Calcula p0 e p1 usando isotonic regression
        self.p0, self.p1, self.c = self._calc_isotonic(
            sorted_scores, sorted_labels, unique_scores
        )
    
    def _calc_isotonic(self, sorted_scores, sorted_labels, unique_scores):
        """Calcula vetores de isotonic regression (p0, p1)."""
        # Índices onde cada score único começa
        indices = np.searchsorted(sorted_scores, unique_scores)
        
        # Pesos (contagens)
        weights = np.zeros(len(unique_scores))
        weights[:-1] = np.diff(indices)
        weights[-1] = len(sorted_scores) - indices[-1]
        
        # Cumulative sum diagram (CSD)
        k_dash = len(unique_scores)
        P = np.zeros((k_dash + 2, 2))
        P[0, :] = -1
        P[2:, 0] = np.cumsum(weights)
        
        # Soma cumulativa de labels
        cumsum_labels = np.cumsum(sorted_labels)
        P[2:-1, 1] = cumsum_labels[(indices - 1)[1:]]
        P[-1, 1] = cumsum_labels[-1]
        
        # Calcula p1 (greatest convex minorant)
        p1 = np.zeros((len(unique_scores) + 1, 2))
        p1[1:, 0] = unique_scores
        
        P1 = P[1:] + 1
        c_point = 0
        grad = 0
        
        for i in range(len(p1)):
            P1[i, :] = P1[i, :] - 1
            
            if i == 0:
                with np.errstate(divide='ignore', invalid='ignore'):
                    grads = np.divide(P1[:, 1], P1[:, 0])
                grad = np.nanmin(grads)
                p1[i, 1] = grad
                c_point = 0
            else:
                imp_point = P1[c_point, 1] + (P1[i, 0] - P1[c_point, 0]) * grad
                
                if P1[i, 1] < imp_point:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        grads = np.divide((P1[i:, 1] - P1[i, 1]), (P1[i:, 0] - P1[i, 0]))
                    if np.sum(np.isnan(np.nanmin(grads))) == 0:
                        grad = np.nanmin(grads)
                    c_point = i
                    p1[i, 1] = grad
                else:
                    p1[i, 1] = grad
        
        # Calcula p0 (least concave majorant)
        p0 = np.zeros((len(unique_scores) + 1, 2))
        p0[1:, 0] = unique_scores
        
        P0 = P[1:]
        c_point = len(p1) - 1
        
        for i in range(len(p1) - 1, -1, -1):
            P0[i, 0] = P0[i, 0] + 1
            
            if i == len(p1) - 1:
                with np.errstate(divide='ignore', invalid='ignore'):
                    grads = np.divide((P0[:, 1] - P0[i, 1]), (P0[:, 0] - P0[i, 0]))
                grad = np.nanmax(grads)
                p0[i, 1] = grad
                c_point = i
            else:
                imp_point = P0[c_point, 1] + (P0[i, 0] - P0[c_point, 0]) * grad
                
                if P0[i, 1] < imp_point:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        grads = np.divide((P0[:, 1] - P0[i, 1]), (P0[:, 0] - P0[i, 0]))
                    grads[i:] = 0
                    grad = np.nanmax(grads)
                    c_point = i
                    p0[i, 1] = grad
                else:
                    p0[i, 1] = grad
        
        return p0, p1, unique_scores
    
    def _fit_ovr(self, p_cal, y_cal, precision):
        """One-vs-Rest: treina um calibrador por classe."""
        self.calibrators = []
        
        for i, cls in enumerate(self.classes_):
            # Cria problema binário
            y_binary = (y_cal == cls).astype(int)
            p_binary = np.zeros((len(p_cal), 2))
            p_binary[:, 1] = p_cal[:, i]
            p_binary[:, 0] = 1 - p_binary[:, 1]
            
            # Treina calibrador binário
            cal = VennAbersCalibrator()
            cal._fit_binary(p_binary, y_binary, precision)
            self.calibrators.append(cal)
    
    def _fit_ovo(self, p_cal, y_cal, precision):
        """One-vs-One: treina calibrador para cada par de classes."""
        self.calibrators = []
        self.class_pairs = []
        
        for i in range(self.n_classes_):
            for j in range(i + 1, self.n_classes_):
                cls_i, cls_j = self.classes_[i], self.classes_[j]
                self.class_pairs.append((i, j))
                
                # Filtra exemplos dessas duas classes
                mask = (y_cal == cls_i) | (y_cal == cls_j)
                p_pair = p_cal[mask][:, [i, j]]
                y_pair = y_cal[mask]
                
                # Normaliza probabilidades
                p_pair_norm = p_pair / p_pair.sum(axis=1, keepdims=True)
                
                # Transforma em binário
                y_binary = (y_pair == cls_j).astype(int)
                
                # Treina calibrador
                cal = VennAbersCalibrator()
                cal._fit_binary(p_pair_norm, y_binary, precision)
                self.calibrators.append(cal)
    
    def predict_proba(self, p_test):
        """
        Calibra probabilidades usando Venn-Abers.
        
        Parameters:
        -----------
        p_test : array-like, shape (n_samples, n_classes)
            Probabilidades do modelo a serem calibradas
            
        Returns:
        --------
        p_calibrated : array, shape (n_samples, n_classes)
            Probabilidades calibradas
        """
        p_test = np.array(p_test)
        
        if p_test.ndim == 1:
            p_test_2d = np.zeros((len(p_test), 2))
            p_test_2d[:, 1] = p_test
            p_test_2d[:, 0] = 1 - p_test
            p_test = p_test_2d
        
        if self.is_binary:
            return self._predict_binary(p_test)
        elif self.multiclass_strategy == 'one_vs_rest':
            return self._predict_ovr(p_test)
        else:
            return self._predict_ovo(p_test)
    
    def _predict_binary(self, p_test):
        """Predição binária."""
        scores = p_test[:, 1]
        n_samples = len(scores)
        p_calibrated = np.zeros((n_samples, 2))
        
        for i, score in enumerate(scores):
            # Busca posição no vetor de scores únicos
            idx_right = np.searchsorted(self.c, score, side='right')
            idx_left = np.searchsorted(self.c, score, side='left')
            
            # Obtém p0 e p1
            p0_val = self.p0[idx_right, 1]
            p1_val = self.p1[idx_left, 1]
            
            # Calcula probabilidade calibrada
            p_cal = p1_val / (1 - p0_val + p1_val)
            
            p_calibrated[i, 1] = p_cal
            p_calibrated[i, 0] = 1 - p_cal
        
        return p_calibrated
    
    def _predict_ovr(self, p_test):
        """One-vs-Rest prediction."""
        n_samples = len(p_test)
        p_calibrated = np.zeros((n_samples, self.n_classes_))
        
        for i, cal in enumerate(self.calibrators):
            p_binary = np.zeros((n_samples, 2))
            p_binary[:, 1] = p_test[:, i]
            p_binary[:, 0] = 1 - p_binary[:, 1]
            
            p_cal = cal._predict_binary(p_binary)
            p_calibrated[:, i] = p_cal[:, 1]
        
        # Normaliza
        p_calibrated = p_calibrated / p_calibrated.sum(axis=1, keepdims=True)
        return p_calibrated
    
    def _predict_ovo(self, p_test):
        """One-vs-One prediction usando Wu-Lin-Weng algorithm."""
        n_samples = len(p_test)
        p_calibrated = np.zeros((n_samples, self.n_classes_))
        
        # Calibra cada par
        pair_probs = []
        for idx, (i, j) in enumerate(self.class_pairs):
            p_pair = p_test[:, [i, j]]
            p_pair_norm = p_pair / p_pair.sum(axis=1, keepdims=True)
            
            cal = self.calibrators[idx]
            p_cal = cal._predict_binary(p_pair_norm)
            pair_probs.append(p_cal)
        
        # Agrega usando votação ponderada (Wu-Lin-Weng)
        for sample_idx in range(n_samples):
            for cls_idx in range(self.n_classes_):
                # Coleta todas as probabilidades envolvendo esta classe
                probs_for_class = []
                
                for pair_idx, (i, j) in enumerate(self.class_pairs):
                    if i == cls_idx:
                        probs_for_class.append(pair_probs[pair_idx][sample_idx, 0])
                    elif j == cls_idx:
                        probs_for_class.append(pair_probs[pair_idx][sample_idx, 1])
                
                # Usa inversos para agregação
                if len(probs_for_class) > 0:
                    inv_sum = sum(1/p for p in probs_for_class if p > 0)
                    p_calibrated[sample_idx, cls_idx] = 1 / (inv_sum - (self.n_classes_ - 2))
        
        # Normaliza
        p_calibrated = np.clip(p_calibrated, 1e-10, 1.0)
        p_calibrated = p_calibrated / p_calibrated.sum(axis=1, keepdims=True)
        
        return p_calibrated


# # ==================== EXEMPLOS ====================

# def exemplo_uso_basico():
#     """Demonstra uso model-agnostic."""
#     print("="*70)
#     print("EXEMPLO 1: USO MODEL-AGNOSTIC (Binário)")
#     print("="*70)
    
#     # Gera dados
#     X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
#     # Divide treino em proper train + calibração
#     X_proper, X_cal, y_proper, y_cal = train_test_split(
#         X_train, y_train, test_size=0.2, random_state=42
#     )
    
#     # Treina qualquer modelo (pode ser sklearn, tensorflow, pytorch, etc)
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_proper, y_proper)
    
#     # Obtém probabilidades (funciona com QUALQUER modelo!)
#     p_cal = model.predict_proba(X_cal)
#     p_test = model.predict_proba(X_test)
    
#     # Calibra usando Venn-Abers
#     calibrator = VennAbersCalibrator()
#     calibrator.fit(p_cal, y_cal)
#     p_calibrated = calibrator.predict_proba(p_test)
    
#     # Avalia
#     print(f"\nBrier Score (Original):   {brier_score_loss(y_test, p_test[:, 1]):.4f}")
#     print(f"Brier Score (Calibrado):  {brier_score_loss(y_test, p_calibrated[:, 1]):.4f}")
#     print(f"\nLog Loss (Original):      {log_loss(y_test, p_test):.4f}")
#     print(f"Log Loss (Calibrado):     {log_loss(y_test, p_calibrated):.4f}")


# def exemplo_multiclasse():
#     """Exemplo multiclasse com estratégias diferentes."""
#     print("\n" + "="*70)
#     print("EXEMPLO 2: MULTICLASSE - One-vs-One vs One-vs-Rest")
#     print("="*70)
    
#     X, y = make_classification(
#         n_samples=1000, n_features=20, n_classes=4, 
#         n_informative=15, n_clusters_per_class=1, random_state=42
#     )
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#     X_proper, X_cal, y_proper, y_cal = train_test_split(
#         X_train, y_train, test_size=0.2, random_state=42
#     )
    
#     # Treina modelo
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_proper, y_proper)
    
#     p_cal = model.predict_proba(X_cal)
#     p_test = model.predict_proba(X_test)
    
#     # One-vs-One
#     cal_ovo = VennAbersCalibrator(multiclass_strategy='one_vs_one')
#     cal_ovo.fit(p_cal, y_cal)
#     p_ovo = cal_ovo.predict_proba(p_test)
    
#     # One-vs-Rest
#     cal_ovr = VennAbersCalibrator(multiclass_strategy='one_vs_rest')
#     cal_ovr.fit(p_cal, y_cal)
#     p_ovr = cal_ovr.predict_proba(p_test)
    
#     print(f"\nLog Loss (Original):      {log_loss(y_test, p_test):.4f}")
#     print(f"Log Loss (OvO):           {log_loss(y_test, p_ovo):.4f}")
#     print(f"Log Loss (OvR):           {log_loss(y_test, p_ovr):.4f}")
    
#     # Acurácia
#     acc_orig = np.mean(model.predict(X_test) == y_test)
#     acc_ovo = np.mean(np.argmax(p_ovo, axis=1) == y_test)
#     acc_ovr = np.mean(np.argmax(p_ovr, axis=1) == y_test)
    
#     print(f"\nAcurácia (Original):      {acc_orig:.4f}")
#     print(f"Acurácia (OvO):           {acc_ovo:.4f}")
#     print(f"Acurácia (OvR):           {acc_ovr:.4f}")


# def exemplo_com_arrays_puros():
#     """Demonstra que funciona com arrays puros (sem modelo)."""
#     print("\n" + "="*70)
#     print("EXEMPLO 3: ARRAYS PUROS (sem objeto de modelo)")
#     print("="*70)
    
#     # Simula probabilidades de qualquer fonte
#     np.random.seed(42)
#     n_cal = 200
#     n_test = 100
    
#     # Calibração
#     p_cal = np.random.dirichlet([2, 3, 1], size=n_cal)  # 3 classes
#     y_cal = np.random.choice([0, 1, 2], size=n_cal, p=[0.4, 0.4, 0.2])
    
#     # Teste
#     p_test = np.random.dirichlet([2, 3, 1], size=n_test)
#     y_test = np.random.choice([0, 1, 2], size=n_test, p=[0.4, 0.4, 0.2])
    
#     # Calibra
#     calibrator = VennAbersCalibrator(multiclass_strategy='one_vs_one')
#     calibrator.fit(p_cal, y_cal)
#     p_calibrated = calibrator.predict_proba(p_test)
    
#     print(f"\nLog Loss (Original):      {log_loss(y_test, p_test):.4f}")
#     print(f"Log Loss (Calibrado):     {log_loss(y_test, p_calibrated):.4f}")
    
#     print("\nExemplo de probabilidades:")
#     print("Original:   ", p_test[0])
#     print("Calibrado:  ", p_calibrated[0])


# if __name__ == "__main__":
#     exemplo_uso_basico()
#     exemplo_multiclasse()
#     exemplo_com_arrays_puros()