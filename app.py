import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")
st.title("Basketball Possession Markov Model")

# 1. Define States
t1_states = ["Scorer_In_T1", "Scorer_Out_T1", "NonScorer_In_T1", "NonScorer_Out_T1"] # Early/Not Stressed
t2_states = ["Scorer_In_T2", "Scorer_Out_T2", "NonScorer_In_T2", "NonScorer_Out_T2"] # Late/Stressed
transient_states = t1_states + t2_states
absorbing_states = ["0_pts", "1.48_pts", "2_pts", "3_pts"]
all_states = transient_states + absorbing_states

num_t, num_a = len(transient_states), len(absorbing_states)

# 2. Setup Inputs
st.sidebar.header("Simulation Parameters")
trials = st.sidebar.number_input("Monte Carlo Trials", 100, 50000, 10000)
start_state = st.sidebar.selectbox("Starting State", transient_states)
start_idx = transient_states.index(start_state)

st.subheader("Transition Matrix (P)")
st.write("Edit probabilities below. Ensure each row sums to 1.0.")

# Initialize an equal-probability matrix (1/12 for all transitions)
default_P = np.ones((num_t, len(all_states))) / len(all_states)
df_P = pd.DataFrame(default_P, index=transient_states, columns=all_states)
edited_df = st.data_editor(df_P)

# Normalize probabilities just in case of slight user editing rounding errors
P = edited_df.values
row_sums = P.sum(axis=1, keepdims=True)
P = np.divide(P, row_sums, out=np.zeros_like(P), where=row_sums!=0)

# 3. Analytical Engine
Q = P[:, :num_t]
R = P[:, num_t:]
I = np.eye(num_t)

try:
    F = np.linalg.inv(I - Q)
    B = np.dot(F, R)
    point_values = np.array([0, 1.48, 2, 3])
    expected_ppp = np.dot(B, point_values)[start_idx]
    expected_dur = np.sum(F, axis=1)[start_idx]
except np.linalg.LinAlgError:
    st.error("Matrix is singular. Adjust probabilities so all paths can reach absorbing states.")
    st.stop()

# 4. Simulation Engine (Monte Carlo)
def simulate(start_idx, Q, R, point_values):
    current = start_idx
    ticks = 0
    while True:
        ticks += 1
        probs = np.concatenate((Q[current], R[current]))
        next_state = np.random.choice(len(all_states), p=probs)
        if next_state >= num_t: return point_values[next_state - num_t], ticks
        current = next_state

with st.spinner("Running Dual-Engine Validation..."):
    results = [simulate(start_idx, Q, R, point_values) for _ in range(trials)]
    sim_pts, sim_dur = [r[0] for r in results], [r[1] for r in results]
    exp_ppp, exp_dur = np.mean(sim_pts), np.mean(sim_dur)

# 5. Output Validation
col1, col2 = st.columns(2)
col1.metric("Analytical PPP", f"{expected_ppp:.4f}")
col2.metric("Experimental PPP", f"{exp_ppp:.4f}", delta=f"{exp_ppp - expected_ppp:.4f}")

col3, col4 = st.columns(2)
col3.metric("Analytical Duration (Steps)", f"{expected_dur:.2f}")
col4.metric("Experimental Duration", f"{exp_dur:.2f}", delta=f"{exp_dur - expected_dur:.2f}")

st.write("### The Fundamental Matrix (F)")
st.dataframe(pd.DataFrame(F, index=transient_states, columns=transient_states))

st.write("### Absorption Probabilities (B)")
st.dataframe(pd.DataFrame(B, index=transient_states, columns=absorbing_states))
