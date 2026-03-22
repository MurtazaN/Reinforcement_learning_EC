"""
Run this AFTER training and evaluation:
  1. python3 Q_learning.py train        (generates Q_table.pickle, episode_rewards.pickle, N_sa.pickle)
  2. python3 generate_results.py        (generates plots + prints stats)
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mdp_gym import CastleEscapeEnv

# ---------- Load saved artifacts ----------
with open('Q_table.pickle', 'rb') as f:
    Q_table = pickle.load(f)

with open('episode_rewards.pickle', 'rb') as f:
    episode_rewards = pickle.load(f)

with open('N_sa.pickle', 'rb') as f:
    N_sa = pickle.load(f)

# ---------- Helper: same hash function from Q_learning.py ----------
def hash(obs):
    health = int(obs.get('player_health', 0))
    window = obs.get('window', {})
    cell_values = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            cell = window.get((dx, dy))
            if cell is None or not cell.get('in_bounds', False):
                cell_values.append(8)
                continue
            if cell.get('is_trap'):
                tile_type = 1
            elif cell.get('is_heal'):
                tile_type = 2
            elif cell.get('is_goal'):
                tile_type = 3
            else:
                tile_type = 0
            has_guard = 1 if cell.get('guards') else 0
            cell_values.append(has_guard * 4 + tile_type)
    window_hash = 0
    base = 1
    for v in cell_values:
        window_hash += v * base
        base *= 9
    guard_in_cell = obs.get('guard_in_cell')
    guard_index = 0
    if guard_in_cell:
        try:
            guard_index = int(str(guard_in_cell)[-1])
        except:
            guard_index = 0
    WINDOW_SPACE = 9 ** 9
    GUARD_SPACE = WINDOW_SPACE
    HEALTH_SPACE = GUARD_SPACE * 5
    return int(health) * HEALTH_SPACE + int(guard_index) * GUARD_SPACE + window_hash

# =====================================================
# 1. PLOT: Training rewards per episode (smoothed)
# =====================================================
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, alpha=0.3, label='Raw reward')
# Rolling average for readability
window_size = 500
if len(episode_rewards) >= window_size:
    rolling_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size - 1, len(episode_rewards)), rolling_avg, color='red', label=f'{window_size}-episode rolling avg')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Rewards per Episode')
plt.legend()
plt.tight_layout()
plt.savefig('training_rewards.png', dpi=150)
plt.close()
print("Saved training_rewards.png")

# =====================================================
# 2. Evaluation: run 10,000 episodes to collect stats
# =====================================================
def softmax(x, temp=1.0):
    e_x = np.exp((x - np.max(x)) / temp)
    return e_x / e_x.sum(axis=0)

env = CastleEscapeEnv()
eval_rewards = []
eval_lengths = []
unseen_states = set()
total_actions = 0
actions_from_unseen = 0

for ep in range(10000):
    obs, _, done, _ = env.reset()
    total_reward = 0
    ep_len = 0
    while not done:
        state = hash(obs)
        total_actions += 1
        if state in Q_table:
            action = int(np.random.choice(env.action_space.n, p=softmax(Q_table[state])))
        else:
            unseen_states.add(state)
            actions_from_unseen += 1
            action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        ep_len += 1
    eval_rewards.append(total_reward)
    eval_lengths.append(ep_len)

avg_reward = np.mean(eval_rewards)
avg_length = np.mean(eval_lengths)
num_unique_states = len(Q_table)
num_unseen = len(unseen_states)
pct_unseen = (actions_from_unseen / total_actions) * 100 if total_actions > 0 else 0

print("\n===== EVALUATION RESULTS (10,000 episodes) =====")
print(f"Average reward:          {avg_reward:.2f}")
print(f"Average episode length:  {avg_length:.2f}")
print(f"Unique states in Q-table: {num_unique_states}")
print(f"Unseen states in eval:   {num_unseen}")
print(f"% actions from unseen:   {pct_unseen:.4f}%")
print(f"Training episodes:       {len(episode_rewards)}")
print("=================================================\n")

# =====================================================
# 3. TABLE: 5x8 weighted-average Q-values
# =====================================================
# Identify states by category using the hash structure
WINDOW_SPACE = 9 ** 9
GUARD_SPACE = WINDOW_SPACE
HEALTH_SPACE = GUARD_SPACE * 5

action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'FIGHT', 'HIDE', 'HEAL', 'WAIT']
row_labels = ['Heal cell', 'Guard G1', 'Guard G2', 'Guard G3', 'Guard G4']

# For each state in Q_table, determine its category
# guard_index: 0=none, 1=G1, 2=G2, 3=G3, 4=G4
# To check if at heal: look at the window_hash center cell

def decode_state(state_id):
    """Return (health, guard_index, window_hash) from a state_id."""
    health = state_id // HEALTH_SPACE
    remainder = state_id % HEALTH_SPACE
    guard_index = remainder // GUARD_SPACE
    window_hash = remainder % GUARD_SPACE
    return health, guard_index, window_hash

def center_cell_from_window_hash(window_hash):
    """Extract the center cell value (index 4 in row-major order) from window_hash."""
    # Cell values are packed in base-9; center cell is at index 4
    temp = window_hash
    for i in range(4):
        temp //= 9
    return temp % 9

# Categories: heal_cell (center cell value = 2 or 6), guard G1-G4 (guard_index 1-4)
# center cell tile_type=2 means heal, cell_value = has_guard*4 + 2
# So center=2 (heal, no guard) or center=6 (heal, with guard)

table_data = np.zeros((5, 8))  # weighted sum of Q-values
table_weights = np.zeros((5, 8))  # total weights

for state_id, q_vals in Q_table.items():
    health, guard_index, window_hash = decode_state(state_id)
    center_val = center_cell_from_window_hash(window_hash)
    
    # Weight = total visits to this state (sum of N_sa for all actions)
    if state_id in N_sa:
        state_visits = N_sa[state_id]  # array of per-action counts
    else:
        continue  # skip states never visited
    
    # Check if at heal cell: tile_type == 2 means center_val in {2, 6}
    tile_type = center_val % 4
    is_heal = (tile_type == 2)
    
    # Row 0: heal cell states
    if is_heal:
        for a in range(8):
            w = state_visits[a]
            table_data[0][a] += q_vals[a] * w
            table_weights[0][a] += w
    
    # Rows 1-4: guard G1-G4 states
    if 1 <= guard_index <= 4:
        row = guard_index  # G1->row 1, G2->row 2, etc.
        for a in range(8):
            w = state_visits[a]
            table_data[row][a] += q_vals[a] * w
            table_weights[row][a] += w

# Compute weighted averages
with np.errstate(divide='ignore', invalid='ignore'):
    table_avg = np.where(table_weights > 0, table_data / table_weights, 0.0)

print("5x8 Weighted Average Q-Value Table:")
header = f"{'State':<12}" + "".join(f"{a:>10}" for a in action_names)
print(header)
print("-" * len(header))
for i, label in enumerate(row_labels):
    row_str = f"{label:<12}" + "".join(f"{table_avg[i][j]:>10.2f}" for j in range(8))
    print(row_str)

# Save table as CSV for LaTeX
with open('q_table_summary.csv', 'w') as f:
    f.write("State," + ",".join(action_names) + "\n")
    for i, label in enumerate(row_labels):
        vals = ",".join(f"{table_avg[i][j]:.2f}" for j in range(8))
        f.write(f"{label},{vals}\n")
print("\nSaved q_table_summary.csv")
