# gui_final_low_high.py
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import os

# ==============================
# Load Trained Model & Metadata
# ==============================
if not os.path.exists("random_forest_model.pkl") or not os.path.exists("model_metadata.pkl"):
    raise FileNotFoundError(
        "❌ Model files missing. Run train_model_final.py first to generate random_forest_model.pkl and model_metadata.pkl."
    )

model = joblib.load("random_forest_model.pkl")
meta = joblib.load("model_metadata.pkl")

feature_order = meta["feature_order"]
exercise_map = meta["exercise_map"]
binary_map = meta["binary_map"]

# Reverse mapping (for dropdowns)
inv_ex = {v: k for k, v in {"None": 0, "Moderate": 1, "Regular": 2}.items()}
inv_bin = {0: "No", 1: "Yes"}

# ==============================
# GUI Setup
# ==============================
root = tk.Tk()
root.title("💓 Heart Disease Risk Predictor")
root.geometry("560x760")
root.configure(bg="#f5f7fa")

tk.Label(
    root,
    text="Heart Disease Risk Predictor",
    font=("Helvetica", 20, "bold"),
    bg="#f5f7fa",
    fg="#1976D2",
).pack(pady=18)

form = tk.LabelFrame(
    root,
    text="Enter Patient Details",
    padx=18,
    pady=18,
    bg="#ffffff",
    font=("Helvetica", 12, "bold"),
)
form.pack(padx=20, pady=10, fill="both")

# Numeric inputs
entries = {}
row = 0
for c in ["Age", "Blood Pressure", "Cholesterol Level", "Sleep Hours"]:
    tk.Label(form, text=c + ":", bg="#ffffff", font=("Helvetica", 11)).grid(
        row=row, column=0, sticky="w", pady=6
    )
    e = tk.Entry(form, font=("Helvetica", 11), width=22, relief="solid", bd=1)
    e.grid(row=row, column=1, pady=6, padx=8)
    entries[c] = e
    row += 1

# Dropdown helper
def dropdown(label, var, options, default):
    global row
    tk.Label(form, text=label + ":", bg="#ffffff", font=("Helvetica", 11)).grid(
        row=row, column=0, sticky="w", pady=6
    )
    cb = ttk.Combobox(
        form,
        textvariable=var,
        values=options,
        state="readonly",
        width=22,
        font=("Helvetica", 10),
    )
    cb.grid(row=row, column=1, pady=6, padx=8)
    cb.set(default)
    row += 1

# Dropdown variables
exercise_var = tk.StringVar(value="Moderate")
smoking_var = tk.StringVar(value="No")
high_bp_var = tk.StringVar(value="No")
low_hdl_var = tk.StringVar(value="No")
alcohol_var = tk.StringVar(value="No")

# Dropdown fields
dropdown("Exercise Habits", exercise_var, ["None", "Moderate", "Regular"], "Moderate")
dropdown("Smoking", smoking_var, ["No", "Yes"], "No")
dropdown("High Blood Pressure", high_bp_var, ["No", "Yes"], "No")
dropdown("Low HDL Cholesterol", low_hdl_var, ["No", "Yes"], "No")
dropdown("Alcohol Consumption", alcohol_var, ["No", "Yes"], "No")

# ==============================
# Prediction Logic
# ==============================
def predict():
    try:
        # Get numeric inputs
        age = float(entries["Age"].get().strip())
        bp = float(entries["Blood Pressure"].get().strip())
        chol = float(entries["Cholesterol Level"].get().strip())
        sleep = float(entries["Sleep Hours"].get().strip())

        # Convert categorical inputs to numeric (consistent with training)
        def map_ex(x):
            s = str(x).strip()
            if s in exercise_map:
                return exercise_map[s]
            if s.capitalize() in exercise_map:
                return exercise_map[s.capitalize()]
            return int(float(s))

        def map_bin(x):
            s = str(x).strip()
            if s in binary_map:
                return binary_map[s]
            if s.capitalize() in binary_map:
                return binary_map[s.capitalize()]
            return int(float(s))

        ex = map_ex(exercise_var.get())
        smoke = map_bin(smoking_var.get())
        hbp = map_bin(high_bp_var.get())
        low_hdl = map_bin(low_hdl_var.get())
        alcohol = map_bin(alcohol_var.get())

        # Create feature array in correct order
        user_input = np.array([[age, bp, chol, sleep, ex, smoke, hbp, low_hdl, alcohol]])

        # Predict probability of high risk (class 1)
        prob = model.predict_proba(user_input)[0][1]

        # ===============================
        # 🔹 Only Two Categories (Low / High)
        # ===============================
        if prob < 0.5:
            result = f"💚 Low Risk ({prob*100:.1f}%)"
            color = "#2ecc71"
            style.configure("TProgressbar", troughcolor="#e0e0e0", background="#2ecc71")
        else:
            result = f"💔 High Risk ({prob*100:.1f}%)"
            color = "#e74c3c"
            style.configure("TProgressbar", troughcolor="#e0e0e0", background="#e74c3c")

        # Update GUI
        result_label.config(text=result, fg=color)
        progress_bar["value"] = int(prob * 100)

    except Exception as e:
        messagebox.showerror("Input Error", f"❌ {str(e)}")

# ==============================
# Buttons & Result Display
# ==============================
tk.Button(
    root,
    text="🔍 Predict Risk",
    command=predict,
    font=("Helvetica", 13, "bold"),
    bg="#1976D2",
    fg="white",
    relief="raised",
    bd=3,
    padx=12,
    pady=6,
).pack(pady=18)

result_label = tk.Label(
    root,
    text="Awaiting Input...",
    font=("Helvetica", 14, "bold"),
    bg="#f5f7fa",
    fg="#424242",
)
result_label.pack(pady=8)

style = ttk.Style()
style.theme_use("default")
style.configure(
    "TProgressbar",
    thickness=26,
    troughcolor="#e0e0e0",
    background="#2ecc71",
)
progress_bar = ttk.Progressbar(
    root, style="TProgressbar", orient="horizontal", length=380, mode="determinate"
)
progress_bar.pack(pady=12)

footer = tk.Label(
    root,
    text="Developed using Random Forest (Binary Risk Model)",
    font=("Helvetica", 9, "italic"),
    bg="#f5f7fa",
    fg="#616161",
)
footer.pack(side="bottom", pady=18)

root.mainloop()
