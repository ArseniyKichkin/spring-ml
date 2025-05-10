from flask import Flask, request, jsonify

app = Flask(__name__)

import torch
import torch.optim as optim
import torch.nn as nn
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def calculate_suspect_ratio(x_batch, threshold=3.0):
    """Suspicion ratio: доля выбросов в окне."""
    outliers = (torch.abs(x_batch) > threshold).float().mean()
    return outliers

def calculate_diff_value(window_data, window_p_values, alpha=0.05):
    """
    Args:
        window_data: Тензор формы (window_size, input_dim) - текущее скользящее окно данных
        window_p_values: Массив p-values для каждой точки в окне
        alpha: Уровень значимости для определения подозрительных точек
    Returns:
        dt: Difference value drift (скаляр)
    """
    # Разделяем точки на нормальные (Nt) и подозрительные (St)
    normal_mask = (window_p_values >= alpha/2) & (window_p_values <= 1 - alpha/2)
    suspicious_mask = ~normal_mask

    normal_points = window_data[normal_mask]  # Множество Nt
    suspicious_points = window_data[suspicious_mask]  # Множество St

    # Для подозрительных точек: среднее абсолютных изменений с соседями (xk-1, xk+1)
    suspicious_diff = 0.0
    for i in range(len(window_data)):
        if suspicious_mask[i]:
            left_diff = right_diff = 0.0
            if i > 0:
                left_diff = torch.abs(window_data[i] - window_data[i-1]).item()
            if i < len(window_data)-1:
                right_diff = torch.abs(window_data[i] - window_data[i+1]).item()

            if i > 0 and i < len(window_data)-1:
                suspicious_diff += (left_diff + right_diff) / 2
            else:
                suspicious_diff += left_diff or right_diff  # Односторонняя разница

    # Для нормальных точек: среднее разниц с соседями (fl(xi), fr(xi))
    normal_diff = 0.0
    for i in range(len(window_data)):
        if normal_mask[i]:
            # Ищем ближайших нормальных соседей слева (fl) и справа (fr)
            left_neighbor = None
            for k in range(i-1, -1, -1):
                if normal_mask[k]:
                    left_neighbor = window_data[k]
                    break

            right_neighbor = None
            for k in range(i+1, len(window_data)):
                if normal_mask[k]:
                    right_neighbor = window_data[k]
                    break

            # Считаем разницы
            diff_left = torch.abs(window_data[i] - left_neighbor).item() if left_neighbor is not None else 0.0
            diff_right = torch.abs(window_data[i] - right_neighbor).item() if right_neighbor is not None else 0.0

            if left_neighbor is not None and right_neighbor is not None:
                normal_diff += (diff_left + diff_right) / 2
            else:
                normal_diff += diff_left or diff_right

    # Итоговый dt (средневзвешенный)
    numerator = torch.abs(window_data[-1] - window_data[-2]).item()  # (xt - xt-1)
    numerator += suspicious_diff  # Сумма разниц для подозрительных точек
    denominator = normal_diff / max(1, len(normal_points))  # Среднее для нормальных точек
    dt = numerator / (len(suspicious_points) + 1)  # |St| + 1 в знаменателе

    return dt

class WGLOptimizer(optim.Optimizer):
    def __init__(self, params, lr=0.01, lambda_param=0.5, gamma=1.0, alpha=0.05):
        defaults = dict(lr=lr, lambda_param=lambda_param, gamma=gamma, alpha=alpha)
        super(WGLOptimizer, self).__init__(params, defaults)
        self.suspect_ratio = None
        self.diff_value = None
        self.errors_window = []
        self.data_window = []

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                lr = group['lr']
                lambda_param = group['lambda_param']
                gamma = group['gamma']

                beta_t = lambda_param * torch.exp(-self.diff_value) * (self.diff_value >= gamma).float() + \
                         (1 - lambda_param) * self.suspect_ratio

                p.data.add_(p.grad, alpha=-lr * beta_t)
        return None



    def update_suspicion_features(self, new_data, error):
        """Обновляет st и dt на основе нового значения ошибки."""
        self.errors_window.append(error)
        if len(self.errors_window) > self.defaults['window_size']:
            self.errors_window.pop(0)
        self.data_window.append(new_data)
        if len(self.data_window) > self.defaults['window_size']:
            self.data_window.pop(0)

        # Расчет suspicion ratio (st)
        alpha = self.defaults['alpha']
        p_values = self._calculate_p_values()
        st = np.mean((p_values < alpha/2) | (p_values > 1 - alpha/2))

        if len(self.data_window) >= 2:
            window_tensor = torch.stack(self.data_window)
            dt = calculate_diff_value(window_tensor, p_values, alpha)
        else:
            dt = 0.0

        self.suspect_ratio = torch.tensor(st, dtype=torch.float32)
        self.diff_value = torch.tensor(dt, dtype=torch.float32)

    def _calculate_p_values(self):
        """Вычисляет p-values для всех ошибок в окне."""
        if len(self.errors_window) < 2:
            return np.zeros(len(self.errors_window))

        mu = np.mean(self.errors_window)
        std = max(np.std(self.errors_window), 1e-8)
        z_scores = [(e - mu)/std for e in self.errors_window]
        return norm.cdf(z_scores)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=400, output_dim=1, num_layers=3, l2_penalty=0.0001):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.l2_penalty = l2_penalty

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :])

    def l2_reg(self):
        return self.l2_penalty * sum(p.norm(2) for p in self.parameters())

input_dim = 1   # Размерность одной точки данных
window_size = 20 # Размер скользящего окна
model = LSTMModel(input_dim)
optimizer = WGLOptimizer(model.parameters(), lr=0.01, lambda_param=0.5, gamma=1.0, alpha=0.05)
optimizer.defaults['window_size'] = window_size  # Добавляем размер окна в параметры
criterion = nn.MSELoss()
sliding_window = []
step = 0

def initialize_model():
    global model, optimizer
    model = LSTMModel(input_dim)
    optimizer = WGLOptimizer(model.parameters(), lr=0.01, lambda_param=0.5, gamma=1.0, alpha=0.05)
    optimizer.defaults['window_size'] = window_size

@app.route('/predict', methods=['POST'])
def predict():
    global step, sliding_window
    # Получаем данные из POST-запроса
    data = request.get_json()
    print(data)
    # Для примера, просто выводим полученные данные
    if not data or 'created_at' not in data or 'salary' not in data:
        return jsonify({"error": "Invalid data format"}), 400
    salary_tensor = torch.tensor([data['salary']], dtype=torch.float32)
    loss_history = []
    avg_loss = 0
    new_point = salary_tensor
    print(f"Received data: {data}")


    # Обновляем скользящее окно
    sliding_window.append(new_point)
    if len(sliding_window) > window_size:
        sliding_window.pop(0)
    print(len(sliding_window))
    # Пропускаем, пока окно не заполнится
    if len(sliding_window) < window_size:
        return jsonify({"status": "collecting", "window_size": len(sliding_window)})

    # Подготовка данных
    window_tensor = torch.stack(sliding_window).unsqueeze(0)  # [1, window_size, input_dim]

    # Предсказание и ошибка
    prediction = model(window_tensor)
    true_value = salary_tensor # Следующая точка - "правильный" ответ
    error = (prediction - true_value[0]).item()  # Скалярная ошибка
    print(f"True data: {true_value}")

    # Обновление оптимизатора
    optimizer.update_suspicion_features(new_point, error)

    # Шаг обучения
    optimizer.zero_grad()
    loss = criterion(prediction, true_value[0].view(1, 1)) + model.l2_reg()
    loss.backward()
    optimizer.step()
    avg_loss += loss.item()
    step += 1
    if step % 100 == 0:
        avg_loss /= 100
        loss_history.append(avg_loss)
        print(f"Step {step}, Loss: {avg_loss:.4f}, "
              f"Suspect Ratio: {optimizer.suspect_ratio:.3f}, "
              f"Diff Value: {optimizer.diff_value:.3f}")
    print(loss.item(), type(loss))
    return jsonify({
        "status": "processed",
        "step": step,
        "loss": float(loss.item()),  # Явное преобразование в float
        "prediction": float(prediction.detach().numpy()[0][0]),  # Полное преобразование тензора
        "is_anomalous": bool(optimizer.suspect_ratio > 0.5)  # Явное преобразование в bool
    })



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
