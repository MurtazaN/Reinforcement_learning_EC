import sys
import time
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mdp_gym import CastleEscapeEnv
from vis_gym import *

BOLD = '\033[1m'  # ANSI escape sequence for bold text
RESET = '\033[0m' # ANSI escape sequence to reset text formatting

train_flag = 'train' in sys.argv
gui_flag = 'gui' in sys.argv

setup(GUI=gui_flag)
env = game # Gym environment already initialized within vis_gym.py

env.render() # Uncomment to print game state info

def hash(obs):
	'''
    Compute a unique compact integer ID representing the given observation.

    Encoding scheme:
      - Observation fields:
          * player_health: integer in {0, 1, 2}
          * window: a 3×3 grid of cells, indexed by (dx, dy) with dx, dy ∈ {-1, 0, 1}
          * guard_in_cell: optional identifier of a guard in the player’s cell (e.g. 'G1', 'G2', ...)

      - Each cell contributes a single digit (0–8) to a base-9 number:
          * If the cell is out of bounds → code = 8
          * Otherwise:
                tile_type = 
                    0 → empty
                    1 → trap
                    2 → heal
                    3 → goal
                has_guard = 1 if one or more guards present, else 0
                cell_value = has_guard * 4 + tile_type  # ranges from 0 to 7

        The 9 cell_values (row-major order: top-left → bottom-right) form a 9-digit base-9 integer `window_hash`.

      - The final state_id packs:
            * window_hash  → fine-grained local state
            * guard_index  → identity of guard in player’s cell (0 if none, 1–4 otherwise)
            * player_health → coarse health component

        Specifically:
            WINDOW_SPACE = 9 ** 9
            GUARD_SPACE  = WINDOW_SPACE       # for guard_index (0–4)
            HEALTH_SPACE = GUARD_SPACE * 5    # for health (0–2)

            state_id = (player_health * HEALTH_SPACE) 
                     + (guard_index * GUARD_SPACE) 
                     + window_hash

    Returns:
        int: A unique, compact integer ID suitable for tabular RL (e.g. as a Q-table key).
    '''
	health = int(obs.get('player_health', 0))
	window = obs.get('window', {})

	# Build cell values in a stable order: dx -1..1 (rows), dy -1..1 (cols)
	cell_values = []
	for dx in [-1, 0, 1]:
		for dy in [-1, 0, 1]:
			cell = window.get((dx, dy))
			if cell is None or not cell.get('in_bounds', False):
				cell_values.append(8)
				continue

			# Determine tile type
			if cell.get('is_trap'):
				tile_type = 1
			elif cell.get('is_heal'):
				tile_type = 2
			elif cell.get('is_goal'):
				tile_type = 3
			else:
				tile_type = 0

			has_guard = 1 if cell.get('guards') else 0
			cell_value = has_guard * 4 + tile_type
			cell_values.append(cell_value)

	# Pack into base-9 integer
	window_hash = 0
	base = 1
	for v in cell_values:
		window_hash += v * base
		base *= 9

	# Include guard identity when player is in the center cell.
	# guard_in_cell is a convenience field set by the environment (e.g. 'G1' or None).
	guard_in_cell = obs.get('guard_in_cell')
	if guard_in_cell:
		# map 'G1' -> 1, 'G2' -> 2, etc.
		try:
			guard_index = int(str(guard_in_cell)[-1])
		except Exception:
			guard_index = 0
	else:
		guard_index = 0

	# window_hash uses 9^9 space; reserve an extra multiplier for guard identity (0..4)
	WINDOW_SPACE = 9 ** 9
	GUARD_SPACE = WINDOW_SPACE  # one slot per guard id
	HEALTH_SPACE = GUARD_SPACE * 5  # 5 possible guard_id values (0 = none, 1-4 = guards)

	state_id = int(health) * HEALTH_SPACE + int(guard_index) * GUARD_SPACE + window_hash
	return state_id

'''
Complete the function below to do the following:

		1. Run a specified number of episodes of the game (argument num_episodes). An episode refers to starting in some initial
			 configuration and taking actions until a terminal state is reached.
		2. Maintain and update Q-values for each state-action pair encountered by the agent in a dictionary (Q-table).
		3. Use epsilon-greedy action selection when choosing actions (explore vs exploit).
		4. Update Q-values using the standard Q-learning update rule.

Important notes about the current environment and state representation

		- The environment is partially observable: observations returned by env.get_observation() include a centered 3x3
			"window" around the player plus the player's health. Each observation is a dict with these relevant keys:
					- 'player_position': (x, y)
					- 'player_health': integer (0=Critical, 1=Injured, 2=Full)
					- 'window': a dict keyed by (dx,dy) offsets in {-1,0,1} x {-1,0,1}. Each entry contains:
								{ 'guards': list or None, 'is_trap': bool, 'is_heal': bool, 'is_goal': bool, 'in_bounds': bool }
					- 'at_trap', 'at_heal', 'at_goal', and 'guard_in_cell' are convenience fields for the center cell.

		- To make a compact and consistent state hash for tabular Q-learning, encode the 3x3 window plus player health into a single integer.
			use the provided hash(obs) function above. Note that the player position is not included in the hash, as it is not needed for local decision-making.

		- Your Q-table should be a dict mapping state_id -> np.array of length env.action_space.n. Initialize arrays to zeros
			when you first encounter a state.

		- The actions available in this environment now include movement, combat, healing and waiting. The action indices are:
					0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: FIGHT, 5: HIDE, 6: HEAL, 7: WAIT

		- Remember to call obs, reward, done, info = env.reset() at the start of each episode.

		- Use a learning-rate schedule per (s,a) pair, i.e. eta = 1/(1 + N(s,a)) where N(s,a) is the
			number of updates applied to that pair so far.

Finally, return the dictionary containing the Q-values (called Q_table).

'''

def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
	"""
	Run Q-learning algorithm for a specified number of episodes.

	Parameters:
	- num_episodes (int): Number of episodes to run.
	- gamma (float): Discount factor.
	- epsilon (float): Exploration rate.
	- decay_rate (float): Rate at which epsilon decays.

	Returns:
	- tuple: (Q_table, episode_rewards, N_sa)
	"""
	Q_table = {}
	N_sa = {}  # Track update counts per (state, action) pair
	n_actions = env.action_space.n
	episode_rewards = []  # Track total reward per episode for plotting

	for episode in tqdm(range(num_episodes)):
		obs, reward, done, info = env.reset()
		state = hash(obs)
		total_reward = 0

		while not done:
			# Initialize Q-values for unseen states
			if state not in Q_table:
				Q_table[state] = np.zeros(n_actions)
				N_sa[state] = np.zeros(n_actions)

			# Epsilon-greedy action selection
			if random.random() < epsilon:
				action = env.action_space.sample()
			else:
				action = int(np.argmax(Q_table[state]))

			# Take action
			obs_next, reward, done, info = env.step(action)
			next_state = hash(obs_next)

			# Initialize next state if unseen
			if next_state not in Q_table:
				Q_table[next_state] = np.zeros(n_actions)
				N_sa[next_state] = np.zeros(n_actions)

			# Compute learning rate
			alpha = 1.0 / (1.0 + N_sa[state][action])

			# Q-learning update
			best_next = np.max(Q_table[next_state])
			Q_table[state][action] += alpha * (reward + gamma * best_next - Q_table[state][action])

			# Increment visit count
			N_sa[state][action] += 1

			# Accumulate reward
			total_reward += reward

			# Transition
			state = next_state

		# Decay epsilon after each episode
		epsilon *= decay_rate
		episode_rewards.append(total_reward)

	return Q_table, episode_rewards, N_sa

# Specify number of episodes and decay rate for training and evaluation.

num_episodes = 130000
decay_rate = 0.99998

'''
Run training if train_flag is set; otherwise, run evaluation using saved Q-table.
'''

if train_flag:
	Q_table, episode_rewards, N_sa = Q_learning(num_episodes=num_episodes, gamma=0.9, epsilon=1, decay_rate=decay_rate) # Run Q-learning

	# Save the Q-table dict to a file
	with open('Q_table.pickle', 'wb') as handle:
		pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open('train_stats.pickle', 'wb') as handle:
		pickle.dump((episode_rewards, N_sa), handle, protocol=pickle.HIGHEST_PROTOCOL)


'''
Evaluation mode: play episodes using the saved Q-table. Useful for debugging/visualization.
Based on autograder logic used to execute actions using uploaded Q-tables.
'''

def softmax(x, temp=1.0):
	e_x = np.exp((x - np.max(x)) / temp)
	return e_x / e_x.sum(axis=0)

def generate_results(Q_table, episode_rewards, N_sa):
	rolling_window = 500
	plt.figure(figsize=(10, 5))
	plt.plot(episode_rewards, alpha=0.3, label='Raw reward')
	if len(episode_rewards) >= rolling_window:
		rolling_avg = np.convolve(episode_rewards, np.ones(rolling_window) / rolling_window, mode='valid')
		plt.plot(range(rolling_window - 1, len(episode_rewards)), rolling_avg, color='red', label=f'{rolling_window}-episode rolling avg')
	plt.xlabel('Episode')
	plt.ylabel('Total Reward')
	plt.title('Training Rewards per Episode')
	plt.legend()
	plt.tight_layout()
	plt.savefig('training_rewards.png', dpi=150)
	plt.close()
	print('Saved as training_rewards.png')

	WINDOW_SPACE = 9 ** 9
	GUARD_SPACE = WINDOW_SPACE
	HEALTH_SPACE = GUARD_SPACE * 5
	action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'FIGHT', 'HIDE', 'HEAL', 'WAIT']
	row_labels = ['Heal cell', 'Guard G1', 'Guard G2', 'Guard G3', 'Guard G4']

	def decode_state(state_id):
		remainder = state_id % HEALTH_SPACE
		guard_index = remainder // GUARD_SPACE
		window_hash = remainder % GUARD_SPACE
		return guard_index, window_hash

	def center_cell_from_window_hash(window_hash):
		temp = window_hash
		for _ in range(4):
			temp //= 9
		return temp % 9

	table_data = np.zeros((5, 8))
	table_weights = np.zeros((5, 8))

	for state_id, q_vals in Q_table.items():
		guard_index, window_hash = decode_state(state_id)
		center_val = center_cell_from_window_hash(window_hash)
		state_visits = N_sa[state_id]
		tile_type = center_val % 4
		is_heal = (tile_type == 2)

		if is_heal:
			for a in range(8):
				w = state_visits[a]
				table_data[0][a] += q_vals[a] * w
				table_weights[0][a] += w

		if 1 <= guard_index <= 4:
			row = guard_index
			for a in range(8):
				w = state_visits[a]
				table_data[row][a] += q_vals[a] * w
				table_weights[row][a] += w

	with np.errstate(divide='ignore', invalid='ignore'):
		table_avg = np.where(table_weights > 0, table_data / table_weights, 0.0)

	print('5x8 Weighted Average Q-Value Table:')
	header = f"{'State':<12}" + ''.join(f"{a:>10}" for a in action_names)
	print(header)
	print('-' * len(header))
	for i, label in enumerate(row_labels):
		row_str = f"{label:<12}" + ''.join(f"{table_avg[i][j]:>10.2f}" for j in range(8))
		print(row_str)

if not train_flag:
	
	rewards = []
	episode_lengths = []
	unseen_state_set = set()
	unseen_action_count = 0
	total_action_count = 0

	filename = 'Q_table.pickle'
	input(f"\n{BOLD}Currently loading Q-table from "+filename+f"{RESET}.  \n\nPress Enter to confirm, or Ctrl+C to cancel and load a different Q-table file.\n(set num_episodes and decay_rate in Q_learning.py).")
	Q_table = np.load(filename, allow_pickle=True)

	for episode in tqdm(range(10000)):
		obs, reward, done, info = env.reset()
		total_reward = 0
		episode_steps = 0
		
		while not done:
			state = hash(obs)
			if state not in Q_table:
				unseen_state_set.add(state)
				unseen_action_count += 1
			total_action_count += 1
			try:
				action = np.random.choice(env.action_space.n, p=softmax(Q_table[state]))  # Select action using softmax over Q-values
			except KeyError:
				action = env.action_space.sample()  # Fallback to random action if state not in Q-table
			
			obs, reward, done, info = env.step(action)
			
			total_reward += reward
			episode_steps += 1
			if gui_flag:
				refresh(obs, reward, done, info, delay=.1)  # Update the game screen [GUI only]

		# print("Total reward:", total_reward)
		rewards.append(total_reward)
		episode_lengths.append(episode_steps)

	avg_reward = sum(rewards)/len(rewards)
	avg_length = sum(episode_lengths)/len(episode_lengths)
	unseen_action_pct = (100.0 * unseen_action_count / total_action_count) if total_action_count > 0 else 0.0

	print('\nResults Summary:')
	print(f'- Training episodes: {num_episodes}')
	print(f'- Decay rate: {decay_rate}')
	print(f'- Average episode length (evaluation): {avg_length:.2f}')
	print(f'- Average reward over {len(rewards)} evaluation episodes: {avg_reward:.2f}')
	print(f'- Number of unique states in Q-table: {len(Q_table)}')
	print(f'- Unique evaluation states not in Q-table: {len(unseen_state_set)}')
	print(f'- Percentage of actions from unseen states: {unseen_action_pct:.2f}%')
	with open('train_stats.pickle', 'rb') as handle:
		episode_rewards, N_sa = pickle.load(handle)

	generate_results(Q_table, episode_rewards, N_sa)