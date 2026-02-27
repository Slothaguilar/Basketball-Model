import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")
st.title("Basketball Possession Markov Model")

# 1. Define States
t1_states = ["Scorer_In_T1", "Scorer_Out_T1", "NonScorer_In_T1", "NonScorer_Out_T1"]
t2_states = ["Scorer_In_T2", "Scorer_Out_T2", "NonScorer_In_T2", "NonScorer_Out_T2"]
transient_states = t1_states + t2_states
absorbing_states = ["0_pts", "1.3_pts", "2_pts", "3_pts"]
all_states = transient_states + absorbing_states
num_t, num_a = len(transient_states), len(absorbing_states)

# 2. Setup Inputs
st.sidebar.header("Simulation Parameters")
trials = st.sidebar.number_input("Monte Carlo Trials", 100, 50000, 10000)
start_state = st.sidebar.selectbox("Starting State", transient_states)
start_idx = transient_states.index(start_state)

st.subheader("Transition Matrix (P)")
st.write("Adjusted for High School Stats (50% 2PT, 30% 3PT, 65% FT). Turnovers are higher for NonScorers and in T2 (~20% team average target).")

default_P = np.zeros((num_t, len(all_states)))

for i, state in enumerate(transient_states):
    is_scorer = state.startswith("Scorer")
    is_t2 = "T2" in state
    is_in = "In" in state
   
    # Calculate overall Absorb Probability (P_A)
    p_a = 0.15 + (0.15 if is_scorer else 0) + (0.25 if is_t2 else 0) + (0.05 if is_in else 0)
    p_t = 1.0 - p_a
   
    # Distribute transient equally (Passing / Resets to any of the 8 states)
    default_P[i, :num_t] = p_t / num_t
   
    # Turnover Logic: Higher for NonScorers AND higher for T2
    turnover_weight = 0.15 # Base turnover weight
    if not is_scorer: turnover_weight += 0.20
    if is_t2: turnover_weight += 0.15
   
    turnover_prob = p_a * turnover_weight
    shot_prob = p_a - turnover_prob
   
    ft_prob = shot_prob * 0.20 # 20% of shot attempts draw a foul
    clean_shot = shot_prob * 0.80
   
    if is_in:
        make_prob = clean_shot * 0.50 # 50% 2PT
        miss_prob = clean_shot * 0.50
        default_P[i, num_t:] = [turnover_prob + miss_prob, ft_prob, make_prob, 0.0]
    else:
        make_prob = clean_shot * 0.30 # 30% 3PT
        miss_prob = clean_shot * 0.70
        default_P[i, num_t:] = [turnover_prob + miss_prob, ft_prob, 0.0, make_prob]

df_P = pd.DataFrame(default_P, index=transient_states, columns=all_states)
edited_df = st.data_editor(df_P)

# Normalize
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
    point_values = np.array([0, 1.3, 2, 3])
    expected_ppp = np.dot(B, point_values)[start_idx]
    expected_dur = np.sum(F, axis=1)[start_idx]
except np.linalg.LinAlgError:
    st.error("Matrix is singular.")
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
