import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")
st.title("Basketball Possession Markov Chain Validator")

# 1. Define States
time_states = [f"T_{i}" for i in range(35, -1, -1)]
transient_states = time_states + ["Shooting_Foul"]
absorbing_states = ["0_pts", "1_pt", "2_pts", "3_pts"]
all_states = transient_states + absorbing_states

# 2. Sidebar Inputs
st.sidebar.header("Simulation Parameters")
trials = st.sidebar.number_input("Monte Carlo Trials", 100, 50000, 10000)
start_time = st.sidebar.slider("Starting Shot Clock (Seconds)", 0, 35, 35)

# 3. Dummy Probability Matrix Setup (Editable by User)
# In reality, you'd populate this with default tactical probabilities
num_t = len(transient_states)
num_a = len(absorbing_states)
P_matrix = np.zeros((num_t, len(all_states)))

# Auto-fill basic directional decay and resets for convenience
for i in range(36): # T_35 to T_0
    if i < 35: P_matrix[i, i+1] = 0.50 # Time ticks down
    P_matrix[i, 0] = 0.05 # Reset loop (Offensive Rebound / Non-shooting foul)
    P_matrix[i, num_t-1] = 0.10 # Go to Shooting Foul
    P_matrix[i, num_t] = 0.15 # 0 pts (Turnover/Miss)
    P_matrix[i, num_t+2] = 0.15 # 2 pts
    P_matrix[i, num_t+3] = 0.05 # 3 pts
   
# Shooting Foul state logic (goes to 0, 1, or 2 pts, or reset)
P_matrix[-1, 0] = 0.10 # Reset (Miss last FT, get Off Reb)
P_matrix[-1, num_t] = 0.20 # 0 pts (Miss all)
P_matrix[-1, num_t+1] = 0.40 # 1 pt (Make 1, miss 1)
P_matrix[-1, num_t+2] = 0.30 # 2 pts (Make both)

df_P = pd.DataFrame(P_matrix, index=transient_states, columns=all_states)

st.subheader("Transition Probabilities Matrix (Edit Below)")
st.write("Rows must sum to 1. Edit the probabilities to model different tactics.")
edited_df = st.data_editor(df_P)

# Normalize strictly to ensure valid Markov Chain
P = edited_df.values
row_sums = P.sum(axis=1, keepdims=True)
P = np.divide(P, row_sums, out=np.zeros_like(P), where=row_sums!=0)

# 4. Engine 1: Analytical Engine
Q = P[:, :num_t]
R = P[:, num_t:]
I = np.eye(num_t)

try:
    F = np.linalg.inv(I - Q)
    B = np.dot(F, R)
    point_values = np.array([0, 1, 2, 3])
    expected_ppp_array = np.dot(B, point_values)
    expected_duration_array = np.sum(F, axis=1)
   
    start_idx = 35 - start_time
    theo_ppp = expected_ppp_array[start_idx]
    theo_dur = expected_duration_array[start_idx]
except np.linalg.LinAlgError:
    st.error("Matrix is singular. Check your probabilities.")
    st.stop()

# 5. Engine 2: Simulation Engine (Monte Carlo)
def simulate_possession(start_idx, Q, R, point_values):
    current_state = start_idx
    ticks = 0
    while True:
        ticks += 1
        probs = np.concatenate((Q[current_state], R[current_state]))
        next_state = np.random.choice(len(all_states), p=probs)
        if next_state >= num_t: # Absorbed
            return point_values[next_state - num_t], ticks
        current_state = next_state

with st.spinner("Running Monte Carlo Simulation..."):
    sim_results = [simulate_possession(start_idx, Q, R, point_values) for _ in range(trials)]
    sim_pts = [r[0] for r in sim_results]
    sim_dur = [r[1] for r in sim_results]
   
    exp_ppp = np.mean(sim_pts)
    exp_dur = np.mean(sim_dur)

# 6. Reconciliation & Outputs
col1, col2 = st.columns(2)
col1.metric("Analytical PPP", f"{theo_ppp:.4f}")
col2.metric("Experimental PPP", f"{exp_ppp:.4f}", delta=f"{exp_ppp - theo_ppp:.4f}")

col3, col4 = st.columns(2)
col3.metric("Analytical Duration (Ticks)", f"{theo_dur:.2f}")
col4.metric("Experimental Duration", f"{exp_dur:.2f}", delta=f"{exp_dur - theo_dur:.2f}")

st.subheader("Fundamental Matrix (F)")
st.dataframe(pd.DataFrame(F, index=transient_states, columns=transient_states))

st.subheader("Absorption Probabilities (B)")
st.dataframe(pd.DataFrame(B, index=transient_states, columns=absorbing_states))
