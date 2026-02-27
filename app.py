import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")
st.title("Basketball Possession Markov Model")

# 1. Define States
t1_states = ["Scorer_In_T1", "Scorer_Out_T1", "NonScorer_In_T1", "NonScorer_Out_T1"]
t2_states = ["Scorer_In_T2", "Scorer_Out_T2", "NonScorer_In_T2", "NonScorer_Out_T2"]
transient_states = t1_states + t2_states
absorbing_states = ["0_pts", "1.3_pts", "2_pts", "2.75_pts", "3_pts", "3.75_pts"]
all_states = transient_states + absorbing_states
num_t, num_a = len(transient_states), len(absorbing_states)

# 2. Setup Inputs
st.sidebar.header("Simulation Parameters")
trials = st.sidebar.number_input("Monte Carlo Trials", 100, 50000, 10000)
start_state = st.sidebar.selectbox("Starting State", transient_states)
start_idx = transient_states.index(start_state)

st.subheader("Transition Matrix (P)")
st.write("Now featuring 38% OREB looping back to T1, and late-clock And-1 probabilities.")

default_P = np.zeros((num_t, len(all_states)))

for i, state in enumerate(transient_states):
    is_scorer = state.startswith("Scorer")
    is_t2 = "T2" in state
    is_in = "In" in state
   
    # Calculate overall Action Probability (P_A)
    p_a = 0.15 + (0.15 if is_scorer else 0) + (0.25 if is_t2 else 0) + (0.05 if is_in else 0)
    p_t = 1.0 - p_a
   
    # Distribute standard transient passing
    default_P[i, :num_t] = p_t / num_t
   
    # Turnovers (0_pts direct)
    turnover_weight = 0.15 + (0.20 if not is_scorer else 0) + (0.15 if is_t2 else 0)
    turnover_prob = p_a * turnover_weight
    shot_prob = p_a - turnover_prob
   
    # Shots & Free Throws
    ft_prob = shot_prob * 0.20 # 2-shot FT trip
    clean_shot = shot_prob * 0.80
    total_make = clean_shot * (0.50 if is_in else 0.30)
    total_miss = clean_shot * (0.50 if is_in else 0.70)
   
    # OREB Reset Loop vs DREB (0_pts)
    oreb_prob = total_miss * 0.38
    dreb_prob = total_miss * 0.62
   
    # Distribute OREB evenly back to the 4 T1 states
    for t1_idx in range(4):
        default_P[i, t1_idx] += (oreb_prob / 4)
       
    # And-1 Logic (Higher probability in T2)
    and1_rate = 0.10 if is_t2 else 0.05
    and1_make = total_make * and1_rate
    clean_make = total_make * (1 - and1_rate)
   
    # Assign Absorbing States
    default_P[i, num_t] = turnover_prob + dreb_prob # 0_pts
    default_P[i, num_t + 1] = ft_prob               # 1.3_pts
   
    if is_in:
        default_P[i, num_t + 2] = clean_make        # 2_pts
        default_P[i, num_t + 3] = and1_make         # 2.75_pts
    else:
        default_P[i, num_t + 4] = clean_make        # 3_pts
        default_P[i, num_t + 5] = and1_make         # 3.75_pts

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
    point_values = np.array([0, 1.3, 2, 2.75, 3, 3.75])
    expected_ppp = np.dot(B, point_values)[start_idx]
    expected_dur = np.sum(F, axis=1)[start_idx]
except np.linalg.LinAlgError:
    st.error("Matrix is singular.")
    st.stop()

# 4. Simulation Engine
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
    exp_ppp, exp_dur = np.mean([r[0] for r in results]), np.mean([r[1] for r in results])

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
