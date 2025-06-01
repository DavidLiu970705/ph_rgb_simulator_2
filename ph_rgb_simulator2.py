# ph_rgb_simulator_extended.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image, ImageDraw

st.set_page_config(layout="wide")
st.title("🌈 pH 值與顏色模擬實驗室")

# -- Helper Functions --
def ph_to_rgb(ph):
    r = int(np.clip(255 - 25 * ph, 0, 255))
    g = int(np.clip(25 * ph, 0, 255))
    b = int(np.clip(255 - abs(7 - ph) * 32, 0, 255))
    return (r, g, b)

def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

def generate_virtual_tube(ph):
    rgb = ph_to_rgb(ph)
    width, height = 100, 300
    img = Image.new("RGB", (width, height), (230, 230, 230))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(20, 20), (80, 280)], fill=rgb, outline="black", width=3)
    return img

def generate_gradient_bar():
    width, height = 256, 50
    img = Image.new("RGB", (width, height))
    for x in range(width):
        ph = 14 * x / width
        rgb = ph_to_rgb(ph)
        for y in range(height):
            img.putpixel((x, y), rgb)
    return img

# -- Sidebar Controls --
st.sidebar.header("🔧 控制項")
ph_value = st.sidebar.slider("pH 值", 0.0, 14.0, 7.0, 0.1)
show_points = st.sidebar.checkbox("顯示數據點", value=True)
use_sigmoid = st.sidebar.checkbox("使用 Sigmoid 擬合", value=True)

# -- Static Data Points --
data_ph = np.array([1, 3, 5, 7, 9, 11, 13])
data_r = np.array([230, 200, 160, 120, 80, 40, 20])
data_g = np.array([20, 60, 100, 160, 200, 220, 230])
data_b = np.array([60, 80, 120, 180, 140, 100, 80])

# -- Plot RGB Curves --
x = np.linspace(0, 14, 300)

fig, ax = plt.subplots(figsize=(8, 4))

if use_sigmoid:
    popt_r, _ = curve_fit(sigmoid, data_ph, data_r, maxfev=10000)
    popt_g, _ = curve_fit(sigmoid, data_ph, data_g, maxfev=10000)
    popt_b, _ = curve_fit(sigmoid, data_ph, data_b, maxfev=10000)

    y_r = sigmoid(x, *popt_r)
    y_g = sigmoid(x, *popt_g)
    y_b = sigmoid(x, *popt_b)
    label_suffix = "(Sigmoid)"
else:
    y_r = np.interp(x, data_ph, data_r)
    y_g = np.interp(x, data_ph, data_g)
    y_b = np.interp(x, data_ph, data_b)
    label_suffix = "(Linear)"

ax.plot(x, y_r, 'r-', label=f"Red {label_suffix}")
ax.plot(x, y_g, 'g-', label=f"Green {label_suffix}")
ax.plot(x, y_b, 'b-', label=f"Blue {label_suffix}")

if show_points:
    ax.scatter(data_ph, data_r, color='red')
    ax.scatter(data_ph, data_g, color='green')
    ax.scatter(data_ph, data_b, color='blue')

ax.set_xlabel("pH 值")
ax.set_ylabel("RGB 強度")
ax.set_title("pH 對 RGB 的變化曲線")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# -- 虛擬試管顯示 --
st.subheader("🧪 虛擬試管視覺化")
col1, col2 = st.columns([1, 2])
with col1:
    st.image(generate_virtual_tube(ph_value), caption=f"pH = {ph_value:.1f} 的顏色")

with col2:
    st.image(generate_gradient_bar(), caption="pH 0 到 14 的顏色漸層條")

# -- 顯示 RGB 值 --
rgb = ph_to_rgb(ph_value)
st.markdown(f"### 🎨 RGB 值: **`{rgb}`**")
