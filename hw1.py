from matplotlib.figure import Figure
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#GUI介面
window = tk.Tk()
window.title('hw1')
window.geometry(f"{900}x{700}")

label_m = tk.Label(window, text='輸入學習率:')
label_b = tk.Label(window, text='輸入epoch:')
entry_m = tk.Entry(window)
entry_b = tk.Entry(window)
label_f = tk.Label(window, text='')
label_tra = tk.Label(window, text='train accuracy: ')
label_tea = tk.Label(window, text='test accuracy: ')
label_w = tk.Label(window, text='weight: ')
label_tra1 = tk.Label(window, text='')
label_tea1 = tk.Label(window, text='')
label_w1 = tk.Label(window, text='')
label_trp = tk.Label(window, text='train plot: ')
label_tep = tk.Label(window, text='test plot: ')

fig = Figure(figsize=(4, 4))
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=window)

fig2 = Figure(figsize=(4, 4))
ax2 = fig2.add_subplot(111)
canvas2 = FigureCanvasTkAgg(fig2, master=window)

#選取檔案
file_path = ""

def open_file_dialog():
    global file_path
    file_path = filedialog.askopenfilename()
    label_f.config(text = file_path)
    
select_file_button = tk.Button(window, text='選擇檔案', command=open_file_dialog)

#訓練+畫線
def plot_line():
    data = np.loadtxt(file_path)

    #隨機打亂
    np.random.shuffle(data)

    X = data[:, :2]  
    d = data[:, 2] 

    num_samples = len(X)

    train_ratio = 2/3  
    num_train = int(train_ratio * num_samples)

    X_train = X[:num_train]
    d_train = d[:num_train]

    X_test = X[num_train:]
    d_test = d[num_train:]
    
    x_max = X.max(0)[0]
    x_min = X.min(0)[0]
    
    num_features = 2  
    learning_rate = float(entry_m.get())

    #隨機初始鍵節值
    weights = np.random.rand(num_features)
    bias = np.random.rand()

    #測資分類為1和2
    def step_function(x):
        return 2 if x >= 0 else 1

    num_epochs = int(entry_b.get())
    print("w:", weights, "bias:", bias)
    print()

    for epoch in range(num_epochs):
        print(epoch)
        for i in range(num_train):
            #判斷此點對目前鍵節值而言是0還是1
            y = step_function(np.dot(X_train[i], weights) + bias)

            #不合修正
            error = d_train[i] - y
            weights += learning_rate * error * X_train[i]
            bias += learning_rate * error
            
            print(" ", i, "X:", X_train[i], ", d =", d_train[i], ", y =", y)
            print("  w:", weights, "bias:", bias)
            print()
    
    
    #計算train正確率
    correct_predictions = 0

    print("train:")
    print("w:", weights, "bias:", bias)
    label_w1.config(text = (str(weights) + ' ' + str(bias)))
    print()
    
    for i in range(len(X_train)):
        y = step_function(np.dot(X_train[i], weights) + bias)
        print(i, "X:", X_train[i], ", d =", d_train[i], ", y =", y)
        print()
        if y == d_train[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(X_train)
    print(f'Train Accuracy: {accuracy * 100:.2f}%')
    print()
    label_tra1.config(text = (str(accuracy * 100.0) + '%'))
            

    #計算test正確率
    correct_predictions = 0

    print("test:")
    print("w:", weights, "bias:", bias)
    print()
    
    for i in range(len(X_test)):
        y = step_function(np.dot(X_test[i], weights) + bias)
        print(i, "X:", X_test[i], ", d =", d_test[i], ", y =", y)
        print()
        if y == d_test[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(X_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    label_tea1.config(text = (str(accuracy * 100.0) + '%'))
    
    
    #畫圖 紅點為1;藍點為2
    m = weights[0] / weights[1]
    m *= -1
    b = bias / weights[1]
    b *= -1

    tmp = abs(x_max) * 0.2
    x = np.linspace(x_min - tmp, x_max + tmp, 100)
    y = m * x + b

    ax.clear()
    
    ax.plot(x, y, label=f'y = {m}x + {b}', color='black')
    for i, p in enumerate(X_train):
        if d_train[i] == 1:
            ax.scatter(p[0], p[1], color = 'red')
        else:
            ax.scatter(p[0], p[1], color = 'blue')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    
    canvas.draw()
    
    ax2.clear()
    
    ax2.plot(x, y, label=f'y = {m}x + {b}', color='black')
    for i, p in enumerate(X_test):
        if d_test[i] == 1:
            ax2.scatter(p[0], p[1], color = 'red')
        else:
            ax2.scatter(p[0], p[1], color = 'blue')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()
    
    canvas2.draw()

plot_button = tk.Button(window, text='訓練', command=plot_line)


#排版
label_m.place(x=0, y=0)
entry_m.place(x=100, y=0)
label_b.place(x=300, y=0)
entry_b.place(x=400, y=0)
select_file_button.place(x=0, y=50)
label_f.place(x=100, y=50)
plot_button.place(x=750, y=50)
label_tra.place(x=0, y=100)
label_tra1.place(x=100, y=100)
label_tea.place(x=300, y=100)
label_tea1.place(x=400, y=100)
label_w.place(x=0, y=150)
label_w1.place(x=100, y=150)
label_trp.place(x=0, y=200)
canvas.get_tk_widget().place(x=0, y=250)
label_tep.place(x=450, y=200)
canvas2.get_tk_widget().place(x=450, y=250)

window.mainloop()
