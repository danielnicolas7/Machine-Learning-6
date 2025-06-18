import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.nn.functional as F
from collections import Counter
from sklearn.metrics import classification_report

from kode import denoise, add_ecg_noise, ECGTransformer, evaluate_model

# ========== HEADER ========== #
st.set_page_config(page_title="Deteksi Aritmia ECG", layout="centered")
st.title("🫀 Deteksi Aritmia dari Sinyal ECG")
st.markdown("Unggah file `.csv` (sinyal) dan `.txt` (anotasi) untuk melakukan deteksi aritmia.")

# ========== UNGGAH FILE ========== #
uploaded_csv = st.file_uploader("📂 Upload file ECG (.csv)", type=["csv"], key="csv_upload")
uploaded_txt = st.file_uploader("📄 Upload file anotasi (.txt)", type=["txt"], key="txt_upload")

# ========== KELAS DAN PARAMETER ========== #
class_info = {
    'N': "Huruf \"N\" menunjukkan detak jantung atau kompleks jantung yang normal 👍",
    'L': "Kelas \"L\" menunjukkan adanya blok cabang kiri pada sistem konduksi jantung.",
    'R': "Kelas \"R\" menunjukkan adanya blok cabang kanan pada sistem konduksi jantung.",
    'A': "Kelas \"A\" mewakili detak prematur atrium / kontraksi dini (dari ruang atas jantung).",
    'V': "Kelas \"V\" mewakili detak prematur ventrikel / kontraksi dini (dari ruang bawah jantung)."
}
classes = list(class_info.keys())
window_size = 1000

# ========== PROSES ========== #
if uploaded_csv and uploaded_txt:
    df_csv = pd.read_csv(uploaded_csv)
    signals = df_csv.iloc[:, 1].astype(int).values
    signals = stats.zscore(signals)
    signals = denoise(signals)
    signals = add_ecg_noise(signals)

    # Plot sinyal
    st.subheader("📈 Plot Sinyal ECG")
    fig, ax = plt.subplots()
    ax.plot(signals[:1000])
    st.pyplot(fig)

    # Baca anotasi
    txt_data = uploaded_txt.read().decode('utf-8').splitlines()
    annotations = []
    for line in txt_data[1:]:  # skip header
        parts = line.split()
        if len(parts) >= 3:
            annotations.append((int(parts[1]), parts[2]))

    st.subheader("🧠 Klasifikasi Deteksi Aritmia")
    progress_bar = st.progress(0, text="Inisialisasi model...")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = {
        "input_channels": 2000,
        "mid_channels": 32,
        "final_out_channels": 128,
        "trans_dim": 128,
        "num_heads": 8,
        "num_classes": 5,
        "dropout": 0.5,
        "stride": 1
    }
    hparams = {"feature_dim": 128}
    model = ECGTransformer(configs, hparams).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    results = []
    test_preds = []
    test_targets = []
    total = len(annotations)

    for i, (pos, typ) in enumerate(annotations):
        if typ in classes and window_size <= pos < len(signals) - window_size:
            beat = signals[pos - window_size: pos + window_size]
            input_tensor = torch.tensor(beat, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                pred_class = output.argmax(dim=1).item()
            results.append((pos, typ, classes[pred_class]))
            test_preds.append(pred_class)
            test_targets.append(classes.index(typ))

        if i % 5 == 0 or i == total - 1:
            progress_bar.progress(i / total, text="Mengklasifikasikan beat...")

    progress_bar.empty()

    # HASIL
    if results:
        st.subheader("✅ Kesimpulan Akhir")
        predicted_classes = [pred for _, _, pred in results]
        most_common_class, count = Counter(predicted_classes).most_common(1)[0]
        st.success(f"Hasil: **'{most_common_class}'** — {class_info[most_common_class]}")
        
        # report = evaluate_model(test_targets, test_preds, classes)
        # st.subheader("📋 Classification Report")
        # st.text(report)

    else:
        st.warning("Tidak ditemukan beat valid untuk diklasifikasi dari data yang diunggah.")

