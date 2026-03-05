import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Basketball Markov Model", layout="wide")
st.title("🏀 Basketball Possession Markov Model")

# --- 1. STATE DEFINITIONS ---
transient_states = ["Early_Clock", "Mid_Clock", "Late_Clock", "OREB_Reset"]
absorbing_states = ["0_pts", "1_pt", "2_pts", "3_pts", "4_pts"]
all_states = transient_states + absorbing_states
num_t = len(transient_states)

point_values = np.array([0, 1, 2, 3, 4])

# --- 2. DEFAULT TRANSITION MATRIX (P) ---
# Rows must sum to 1.
default_P = np.array([
    [0.0, 0.6, 0.0, 0.0,  0.2, 0.0, 0.1, 0.1, 0.0], # Early_Clock
    [0.0, 0.0, 0.5, 0.0,  0.2, 0.1, 0.1, 0.1, 0.0], # Mid_Clock
    [0.0, 0.0, 0.0, 0.1,  0.4, 0.1, 0.2, 0.2, 0.0], # Late_Clock
    [0.4, 0.0, 0.0, 0.0,  0.2, 0.1, 0.3, 0.0, 0.0]  # OREB_Reset (Loops back to Early)
])

# --- UI TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["⚙️ Matrix Setup", "📊 Dual-Engine Validation", "🧮 Fundamental Math", "⛹️ Play-by-Play Visualizer"])

with tab1:
    st.subheader("Interactive Transition Matrix (P)")
    st.markdown("Edit the probabilities below. Ensure each row sums to 1.0.")
    df_P = pd.DataFrame(default_P, index=transient_states, columns=all_states)
    edited_df = st.data_editor(df_P, use_container_width=True)
   
    # Normalize to ensure rows sum to 1
    P = edited_df.values
    row_sums = P.sum(axis=1, keepdims=True)
    P = np.divide(P, row_sums, out=np.zeros_like(P), where=row_sums!=0)

# --- 3. ANALYTICAL ENGINE ---
Q = P[:, :num_t]
R = P[:, num_t:]
I = np.eye(num_t)

try:
    F = np.linalg.inv(I - Q)
    B = np.dot(F, R)
    start_idx = 0 # Default start is Early_Clock
    expected_ppp = np.dot(B, point_values)[start_idx]
    expected_dur = np.sum(F, axis=1)[start_idx]
except np.linalg.LinAlgError:
    st.error("Matrix is singular. Check your probabilities.")
    st.stop()

# --- 4. SIMULATION ENGINE ---
def simulate_possession(start_idx, Q, R, point_vals):
    curr = start_idx
    ticks = 0
    path = [transient_states[curr]]
    while True:
        ticks += 1
        probs = np.concatenate((Q[curr], R[curr]))
        nxt = np.random.choice(len(all_states), p=probs)
        if nxt >= num_t:
            path.append(absorbing_states[nxt - num_t])
            return point_vals[nxt - num_t], ticks, path
        curr = nxt
        path.append(transient_states[curr])

# Sidebar Controls
st.sidebar.header("Simulation Parameters")
trials = st.sidebar.number_input("Monte Carlo Trials", min_value=100, max_value=50000, value=10000, step=1000)

with st.spinner("Running Monte Carlo Simulations..."):
    results = [simulate_possession(start_idx, Q, R, point_values) for _ in range(trials)]
    exp_ppp = np.mean([r[0] for r in results])
    exp_dur = np.mean([r[1] for r in results])

with tab2:
    st.subheader("Reconciliation: Analytical vs. Simulated")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Analytical PPP", f"{expected_ppp:.4f}")
    col2.metric("Simulated PPP", f"{exp_ppp:.4f}", delta=f"{exp_ppp - expected_ppp:.4f}")
    col3.metric("Analytical Duration", f"{expected_dur:.2f} steps")
    col4.metric("Simulated Duration", f"{exp_dur:.2f} steps", delta=f"{exp_dur - expected_dur:.2f}")

with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Fundamental Matrix (F)")
        st.markdown(r"Calculated using $F = (I - Q)^{-1}$")
        df_F = pd.DataFrame(F, index=transient_states, columns=transient_states)
        st.dataframe(df_F.style.background_gradient(cmap='Blues'), use_container_width=True)
    with col_b:
        st.subheader("Absorption Probabilities (B)")
        st.markdown("Calculated using $B = F \times R$")
        df_B = pd.DataFrame(B, index=transient_states, columns=absorbing_states)
        st.dataframe(df_B.style.background_gradient(cmap='Greens'), use_container_width=True)

with tab4:
    st.subheader("Live Play-by-Play Visualizer")
    st.write("Visually trace a single simulated possession through the Markov Chain:")
    if st.button("🔄 Simulate Single Possession"):
        _, _, sample_path = simulate_possession(start_idx, Q, R, point_values)
        path_str = " ➔ ".join([f"`{step}`" for step in sample_path])
        st.info(f"**Path:** {path_str}")
