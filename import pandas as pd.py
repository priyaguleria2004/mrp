import pandas as pd
import streamlit as st
import os
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Mood Music App", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f8fafc, #e0f2fe);
}

h1 {
    text-align: center;
    color: #1e293b;
    font-size: 42px !important;
}

/* SONG CARD */
.song-card {
    background: white;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}

.song-card:hover {
    transform: scale(1.05);
}

/* BUTTON */
.stButton>button {
    border-radius: 10px;
}

.stButton>button:hover {
    background-color: #2563eb;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
try:
    df = pd.read_csv("song.csv")
except:
    st.error("❌ song.csv file not found!")
    st.stop()

required_cols = ['energy','tempo','valence','mood','song_name']
for col in required_cols:
    if col not in df.columns:
        st.error(f"❌ Missing column: {col}")
        st.stop()

if "artist" not in df.columns:
    df["artist"] = "Unknown"

# ---------------- FEATURES ----------------
X = df[['energy','tempo','valence']]
y = df['mood']

# ---------------- MODELS ----------------
knn = KNeighborsClassifier(n_neighbors=3)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

knn.fit(X, y)
dt.fit(X, y)
rf.fit(X, y)

# ---------------- SESSION ----------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ================= HOME =================
if st.session_state.page == "home":

    st.markdown("<h1>🎵 Mood Music Recommender</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])

    with col1:

        energy = st.slider("Energy ⚡", 0.0, 1.0, 0.5)
        tempo = st.slider("Tempo 🎵", 50, 180, 100)
        valence = st.slider("Positivity 😊", 0.0, 1.0, 0.5)

        model_choice = st.selectbox("Choose Algorithm", ["KNN", "Decision Tree", "Random Forest"])

        if st.button("🚀 Get Recommendations"):
            st.session_state.energy = energy
            st.session_state.tempo = tempo
            st.session_state.valence = valence
            st.session_state.model = model_choice
            st.session_state.page = "results"

        if st.button("🔥 Latest Songs"):
            st.session_state.page = "latest"

        if st.button("📊 Mood Analysis"):
            st.session_state.page = "analysis"

    with col2:
        st.image("https://media.tenor.com/15YUsMWt4FEAAAAj/music.gif", use_container_width=True)

# ================= RESULTS =================
elif st.session_state.page == "results":

    if st.button("⬅ Back"):
        st.session_state.page = "home"

    energy = st.session_state.energy
    tempo = st.session_state.tempo
    valence = st.session_state.valence
    model_choice = st.session_state.model

    model = knn if model_choice == "KNN" else dt if model_choice == "Decision Tree" else rf

    input_data = pd.DataFrame([[energy, tempo, valence]],
                              columns=['energy','tempo','valence'])

    mood = model.predict(input_data)[0]

    st.markdown("<h1>🎧 Recommended Songs</h1>", unsafe_allow_html=True)

    st.success(f"Detected Mood: {mood} ({model_choice})")

    songs = df[df['mood'] == mood].drop_duplicates(subset=['song_name'])

    songs = songs.sample(frac=1).head(9)

    cols = st.columns(3)

    for i, (_, row) in enumerate(songs.iterrows()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="song-card">
                <b>🎵 {row['song_name']}</b><br>
                👤 {row['artist']}
            </div>
            """, unsafe_allow_html=True)

            if 'file_path' in df.columns and pd.notna(row.get('file_path')):
                if os.path.exists(str(row['file_path'])):
                    st.audio(row['file_path'])

            if st.button(f"❤️ Like {row['song_name']}", key=f"like_{i}"):
                st.success("Added to favorites!")

# ================= LATEST =================
elif st.session_state.page == "latest":

    if st.button("⬅ Back"):
        st.session_state.page = "home"

    st.markdown("<h1>🔥 Latest Songs</h1>", unsafe_allow_html=True)

    latest_songs = df.sample(min(10, len(df)))

    for _, row in latest_songs.iterrows():
        st.markdown(f"""
        <div class="song-card">
            <b>🎵 {row['song_name']}</b><br>
            👤 {row['artist']}
        </div>
        """, unsafe_allow_html=True)

# ================= ANALYSIS =================
elif st.session_state.page == "analysis":

    if st.button("⬅ Back"):
        st.session_state.page = "home"

    st.markdown("<h1>📊 Mood Analysis</h1>", unsafe_allow_html=True)

    st.bar_chart(df['mood'].value_counts())

    st.subheader("Feature Averages")
    st.bar_chart(df.groupby('mood')[['energy','tempo','valence']].mean()
                 
                 st.header("🟢 K-NEAREST NEIGHBORS (KNN) MODEL")

st.write("""
KNN 
""")

# Train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Prediction
knn_pred = knn.predict(X)

# Accuracy
knn_acc = (knn_pred == y).mean()

st.success(f"KNN Accuracy: {knn_acc:.2f}")

# Demo prediction
st.subheader("🎧 KNN Demo Prediction")
st.write("Input Example: Energy=0.6, Tempo=120, Valence=0.7")
st.write("Predicted Mood:", knn.predict([[0.6, 120, 0.7]]))

st.header("🔵 DECISION TREE MODEL")

st.write("""
Decision Tree """)

# Train model
dt = DecisionTreeClassifier(max_depth=6)
dt.fit(X, y)

# Prediction
dt_pred = dt.predict(X)

# Accuracy
dt_acc = (dt_pred == y).mean()

st.info(f"Decision Tree Accuracy: {dt_acc:.2f}")

# Demo prediction
st.subheader("🎧 Decision Tree Demo Prediction")
st.write("Input Example: Energy=0.4, Tempo=100, Valence=0.3")
st.write("Predicted Mood:", dt.predict([[0.4, 100, 0.3]]))

st.header("🔴 RANDOM FOREST MODEL")

st.write("""
Random Forest multiple decision trees group
""")

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Prediction
rf_pred = rf.predict(X)

# Accuracy
rf_acc = (rf_pred == y).mean()

st.error(f"Random Forest Accuracy: {rf_acc:.2f}")

# Demo prediction
st.subheader("🎧 Random Forest Demo Prediction")
st.write("Input Example: Energy=0.8, Tempo=140, Valence=0.9")
st.write("Predicted Mood:", rf.predict([[0.8, 140, 0.9]]))

JBJNBN