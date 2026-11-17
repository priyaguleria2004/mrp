
import pandas as pd
import streamlit as st
import os

# ML Imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Mood Music App",
    page_icon="🎵",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: linear-gradient(
        135deg,
        #020617,
        #172554,
        #1e40af,
        #0e7490
    );
}

/* MAIN CONTAINER */
.block-container {
    max-width: 1100px;
    margin: auto;
    padding-top: 2rem;
}

/* MAIN HEADING */
h1 {
    color: #ffffff !important;
    text-align: center;
    font-size: 55px !important;
    font-weight: 900 !important;
    text-shadow: 0 0 12px #38bdf8;
}

/* ALL OTHER HEADINGS */
h2, h3, h4, h5 {
    color: #ffffff !important;
    font-weight: 800 !important;
}

/* NORMAL TEXT */
.stApp p,
.stApp label,
.stApp .stMarkdown,
.stApp li {
    color: #ffffff !important;
}

/* SLIDER LABELS */
div[data-testid="stSlider"] label {
    color: white !important;
    font-size: 22px !important;
    font-weight: bold !important;
}

/* SLIDER VALUE */
div[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: white !important;
}

/* BUTTON DESIGN */
.stButton > button {
    width: 100%;
    padding: 14px;
    font-size: 18px;
    font-weight: 700;
    border-radius: 12px;
    background: linear-gradient(
        135deg,
        #2563eb,
        #06b6d4
    );
    color: white !important;
    border: 1px solid #38bdf8;
    transition: 0.3s;
}

/* BUTTON HOVER */
.stButton > button:hover {
    background: linear-gradient(
        135deg,
        #0891b2,
        #2563eb
    );
    border: 1px solid white;
    box-shadow: 0 0 15px #38bdf8;
}

/* GLOW LINE */
.glow-line {
    height: 3px;
    width: 220px;
    margin: 10px auto 25px auto;
    background: linear-gradient(
        90deg,
        transparent,
        #38bdf8,
        transparent
    );
    box-shadow: 0 0 10px #38bdf8;
}

/* SONG CARD */
.song-card {
    background: rgba(255,255,255,0.10);
    padding: 16px;
    border-radius: 15px;
    margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.20);
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

/* SONG TITLE */
.song-title {
    color: #ffffff !important;
    font-size: 19px;
    font-weight: 700;
}

/* SONG ARTIST */
.song-artist {
    color: #bae6fd !important;
    font-size: 15px;
}

/* SELECTBOX */
div[data-baseweb="select"] * {
    color: white !important;
}

div[data-baseweb="select"] > div {
    background-color: #1e3a8a !important;
}

/* SUCCESS MESSAGE */
div[data-testid="stAlert"] {
    background-color: rgba(14,165,233,0.18);
    border-radius: 12px;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    background-color: rgba(255,255,255,0.08);
}

/* CHART TEXT */
[data-testid="stVegaLiteChart"] {
    background-color: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 10px;
}

/* DIVIDER */
hr {
    border-color: #38bdf8;
}

/* AUDIO PLAYER */
audio {
    width: 100%;
}

</style>
""", unsafe_allow_html=True)


# ---------------- DATA ----------------

if not os.path.exists("songs.csv"):
    st.error("songs.csv file not found. Please keep it in the project folder.")
    st.stop()

df = pd.read_csv("songs.csv")

# Required columns
required_columns = [
    'energy',
    'tempo',
    'valence',
    'mood',
    'song_name',
    'file_path',
    'video_url'
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"Missing columns: {missing_columns}")
    st.stop()

# Artist column
if "artist" not in df.columns:
    df["artist"] = "Unknown"

# Remove incomplete rows
df = df.dropna(
    subset=['energy', 'tempo', 'valence', 'mood']
)

if len(df) < 3 or df['mood'].nunique() < 2:
    st.error("Dataset must contain at least 3 songs and 2 different moods.")
    st.stop()


# ---------------- FEATURES ----------------

X = df[['energy', 'tempo', 'valence']]
y = df['mood']


# ---------------- SCALING ----------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ---------------- ML MODELS ----------------

knn_model = KNeighborsClassifier(
    n_neighbors=3
)

dt_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ---------------- ENSEMBLE MODEL ----------------

model = VotingClassifier(
    estimators=[
        ('knn', knn_model),
        ('dt', dt_model),
        ('rf', rf_model)
    ],
    voting='soft'
)

model.fit(X_scaled, y)


# ---------------- SESSION STATE ----------------

if "page" not in st.session_state:
    st.session_state.page = "home"


# ---------------- HOME PAGE ----------------

if st.session_state.page == "home":

    st.markdown(
        "<h1>🎵 Mood Music Recommender</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="glow-line"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "<h3 style='text-align:center;'>"
        "Discover Music According To Your Mood 🎧"
        "</h3>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div style='height:40px'></div>",
        unsafe_allow_html=True
    )

    # SLIDERS
    col1, col2, col3 = st.columns(3)

    with col1:
        energy = st.slider(
            "Energy ⚡",
            0.0, 1.0, 0.5
        )

    with col2:
        tempo = st.slider(
            "Tempo 🎵",
            50, 180, 100
        )

    with col3:
        valence = st.slider(
            "Positivity 😊",
            0.0, 1.0, 0.5
        )

    st.markdown("<div style='height:25px'></div>",
                unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        if st.button("🚀 Get Recommendations"):

            st.session_state.energy = energy
            st.session_state.tempo = tempo
            st.session_state.valence = valence

            st.session_state.page = "results"
            st.rerun()

    with colB:
        if st.button("📊 Mood Analysis"):

            st.session_state.page = "analysis"
            st.rerun()


# ---------------- RESULTS PAGE ----------------

elif st.session_state.page == "results":

    col_back = st.columns([3, 1, 3])

    with col_back[1]:

        if st.button("⬅ Back"):
            st.session_state.page = "home"
            st.rerun()

    st.markdown(
        "<h1>🎧 Recommended Songs</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="glow-line"></div>',
        unsafe_allow_html=True
    )

    energy = st.session_state.energy
    tempo = st.session_state.tempo
    valence = st.session_state.valence

    # PREDICTION
    input_data = scaler.transform(
        [[energy, tempo, valence]]
    )

    mood = model.predict(input_data)[0]

    st.success(f"🎭 Detected Mood: {mood}")

    # PLAYBACK MODE
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        play_mode = st.selectbox(
            "Playback Mode",
            ["Audio 🎧", "Video 🎬"]
        )

    # FILTER SONGS
    songs = df[df['mood'] == mood]

    if songs.empty:
        st.warning("No songs found for this mood.")

    # SONG CARDS
    for i in range(0, len(songs), 3):

        cols = st.columns(3)

        for j in range(3):

            if i + j < len(songs):

                row = songs.iloc[i + j]

                with cols[j]:

                    st.markdown(f"""
                    <div class="song-card">
                        <div class="song-title">
                            🎵 {row['song_name']}
                        </div>
                        <div class="song-artist">
                            👤 {row['artist']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if play_mode == "Audio 🎧":

                        file_path = str(row['file_path'])

                        if os.path.exists(file_path):
                            st.audio(file_path)
                        else:
                            st.warning("Audio file not found.")

                    else:

                        video_url = row['video_url']

                        if pd.notna(video_url) and str(video_url).strip():

                            st.video(video_url)

                        else:
                            st.warning("Video URL not available.")


# ---------------- ANALYSIS PAGE ----------------

elif st.session_state.page == "analysis":

    col_back = st.columns([3, 1, 3])

    with col_back[1]:

        if st.button("⬅ Back"):
            st.session_state.page = "home"
            st.rerun()

    st.markdown(
        "<h1>📊 Mood Analysis</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="glow-line"></div>',
        unsafe_allow_html=True
    )

    # MOOD DISTRIBUTION
    st.subheader("🎭 Mood Distribution")

    st.bar_chart(
        df['mood'].value_counts()
    )

    # FEATURE COMPARISON
    st.subheader("📈 Feature Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Energy ⚡")

        st.bar_chart(
            df.groupby('mood')['energy'].mean()
        )

    with col2:
        st.write("Tempo 🎵")

        st.bar_chart(
            df.groupby('mood')['tempo'].mean()
        )

    with col3:
        st.write("Positivity 😊")

        st.bar_chart(
            df.groupby('mood')['valence'].mean()
        )

    # DATASET PREVIEW
    st.subheader("📄 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )
