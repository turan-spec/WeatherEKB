# Импорт необходимых библиотек
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
from tkinter import *
from tkinter import ttk, filedialog, messagebox

# Пути к файлам модели и данным
MODEL_FILE = "model/weather_model.h5"
DATA_FILE = "data.txt"

# Цветовая схема приложения
BACKGROUND_COLOR = "#E0F2F7"
BUTTON_COLOR = "#81D4FA"
TEXT_COLOR = "#0D47A1"

# Класс для отслеживания прогресса обучения модели
class PrognozObuchen(Callback):
    def __init__(self, bar, total_epochs, update_callback):
        """
        Инициализация класса для отслеживания прогресса обучения модели.
        
        :param bar: Прогресс-бар для отображения прогресса.
        :param total_epochs: Общее количество эпох обучения.
        :param update_callback: Функция для обновления интерфейса.
        """
        super().__init__()
        self.bar = bar
        self.total_epochs = total_epochs
        self.update_callback = update_callback

    def on_epoch_end(self, epoch, logs=None):
        """
        Обновление прогресс-бара после каждой эпохи обучения.
        
        :param epoch: Текущая эпоха.
        :param logs: Дополнительные логи (не используется).
        """
        progress = ((epoch + 1) / self.total_epochs) * 100
        self.bar["value"] = progress
        self.update_callback()

# Класс для приложения прогноза погоды
class Prilozenie:
    def __init__(self, root):
        """
        Инициализация приложения прогноза погоды.
        
        :param root: Корневой элемент интерфейса Tkinter.
        """
        self.root = root
        self.root.title("Прогноз погоды г. Екатеринбург")
        self.root.geometry("500x550")
        self.root.resizable(False, False)
        self.root.configure(bg=BACKGROUND_COLOR)

        # Инициализация переменных для хранения модели, истории обучения, данных и прогноза
        self.model = None
        self.history = None
        self.data = None
        self.prediction = None

        # Создание виджетов интерфейса
        self.create_widgets()

    def create_widgets(self):
        """
        Создание элементов интерфейса приложения.
        """
        # Заголовок приложения
        title_label = Label(self.root, text="Погода в Екатеринбурге", font=("Times New Roman", 14), bg=BACKGROUND_COLOR, fg=TEXT_COLOR)
        title_label.pack(pady=10)

        # Прогресс-бар для отображения прогресса обучения
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=460)
        self.progress.pack(pady=5)

        # Настройка стиля кнопок
        button_style = ttk.Style()
        button_style.configure("TButton", background=BUTTON_COLOR, foreground=TEXT_COLOR, padding=10, font=("Times New Roman", 12))

        # Выбор года для прогноза
        year_label = Label(self.root, text="Выберите год для прогноза:", bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=("Times New Roman", 12))
        year_label.pack(pady=(10, 0))

        self.year_var = StringVar(value="2025")
        year_choices = [str(year) for year in range(2025, 2031)]
        year_menu = ttk.Combobox(self.root, textvariable=self.year_var, values=year_choices, state="readonly")
        year_menu.pack(pady=5)

        # Кнопки для загрузки данных, обучения модели, загрузки существующей модели, построения прогноза, сохранения прогноза и отображения графиков
        load_data_btn = ttk.Button(self.root, text="Загрузить данные", command=self.load_data, style="TButton")
        load_data_btn.pack(fill=X, padx=20, pady=5)

        train_model_btn = ttk.Button(self.root, text="Обучить модель", command=self.train_model, style="TButton")
        train_model_btn.pack(fill=X, padx=20, pady=5)

        load_model_btn = ttk.Button(self.root, text="Загрузить модель", command=self.load_existing_model, style="TButton")
        load_model_btn.pack(fill=X, padx=20, pady=5)

        make_prediction_btn = ttk.Button(self.root, text="Построить прогноз", command=self.make_prediction, style="TButton")
        make_prediction_btn.pack(fill=X, padx=20, pady=5)

        save_prediction_btn = ttk.Button(self.root, text="Сохранить прогноз", command=self.save_prediction, style="TButton")
        save_prediction_btn.pack(fill=X, padx=20, pady=5)

        plot_forecast_btn = ttk.Button(self.root, text="График прогноза", command=self.plot_forecast, style="TButton")
        plot_forecast_btn.pack(fill=X, padx=20, pady=5)

        plot_loss_btn = ttk.Button(self.root, text="График потерь", command=self.plot_loss, style="TButton")
        plot_loss_btn.pack(fill=X, padx=20, pady=5)

    def load_data(self):
        """
        Загрузка данных из файла.
        """
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as file:
                lines = file.readlines()
                self.data = np.array([[float(n) for n in line.strip().split()] for line in lines])
            messagebox.showinfo("Успех", "Данные успешно загружены")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при загрузке: {e}")

    def train_model(self):
        """
        Обучение модели на загруженных данных.
        """
        if self.data is None or len(self.data) < 2:
            messagebox.showwarning("Ошибка", "Недостаточно данных для обучения")
            return

        # Подготовка данных для обучения
        X = self.data[:-1]
        y = self.data[1:]

        # Создание и компиляция модели
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(12,)),
            Dense(64, activation='relu'),
            Dense(12)
        ])
        self.model.compile(optimizer='adam', loss='mse')

        # Создание директории для модели, если она не существует
        os.makedirs("model", exist_ok=True)
        
        # Обучение модели с отслеживанием прогресса
        epochs = 500
        progress_cb = PrognozObuchen(self.progress, epochs, self.root.update_idletasks)
        self.history = self.model.fit(X, y, epochs=epochs, verbose=0, callbacks=[progress_cb])
        
        # Сохранение обученной модели
        self.model.save(MODEL_FILE)
        messagebox.showinfo("Обучение завершено", "Модель сохранена")

    def load_existing_model(self):
        """
        Загрузка существующей модели из файла.
        """
        try:
            if os.path.exists(MODEL_FILE):
                self.model = load_model(MODEL_FILE, compile=True)
                messagebox.showinfo("Модель загружена", "Модель успешно загружена")
            else:
                messagebox.showwarning("Файл не найден", f"Файл модели '{MODEL_FILE}' отсутствует")
        except Exception as e:
            messagebox.showerror("Ошибка при загрузке модели", f"Не удалось загрузить модель:\n{e}")

    def make_prediction(self):
        """
        Построение прогноза на выбранный год.
        """
        if self.model is None or self.data is None:
            messagebox.showerror("Ошибка", "Необходимо загрузить модель и данные")
            return

        try:
            target_year = int(self.year_var.get())
            years_ahead = target_year - 2024

            if years_ahead <= 0:
                messagebox.showwarning("Ошибка", "Выберите год начиная с 2025")
                return

            # Построение прогноза на выбранный год
            current_input = self.data[-1].reshape(1, -1)
            for _ in range(years_ahead):
                next_year = self.model.predict(current_input)[0]
                current_input = next_year.reshape(1, -1)

            self.prediction = current_input.flatten()
            messagebox.showinfo("Прогноз готов", f"Прогноз рассчитан на {target_year} год")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось построить прогноз: {e}")

    def save_prediction(self):
        """
        Сохранение прогноза в файл.
        """
        if self.prediction is None:
            messagebox.showwarning("Нет прогноза", "Сначала постройте прогноз")
            return

        with open(DATA_FILE, "w") as f:
            f.write(" ".join(f"{val:.2f}" for val in self.prediction))
        messagebox.showinfo("Сохранено", f"Прогноз сохранен в {DATA_FILE}")

    def plot_forecast(self):
        """
        Отображение графика прогноза.
        """
        if self.prediction is None:
            messagebox.showwarning("Нет прогноза", "Постройте прогноз перед отображением графика")
            return

        # Подготовка данных для графика
        months = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
        colors = ['red' if t > 0 else 'blue' for t in self.prediction]

        # Создание графика
        plt.bar(months, self.prediction, color=colors)
        plt.title("Прогноз температуры на год")
        plt.xlabel("Месяц")
        plt.ylabel("Температура, °C")
        plt.axhline(0, color='black', linewidth=0.8)
        plt.tight_layout()
        plt.show()

    def plot_loss(self):
        """
        Отображение графика потерь при обучении модели.
        """
        if self.history is None:
            messagebox.showwarning("Нет истории", "Сначала обучите модель")
            return

        # Создание графика потерь
        plt.plot(self.history.history['loss'])
        plt.title("Потери при обучении")
        plt.xlabel("Эпоха")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Запуск приложения
    root = Tk()
    app = Prilozenie(root)
    root.mainloop()
