import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# Parameters
GRID_SIZE = 10
OBSTACLE_COUNT = 10
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
GAMMA = 0.9
ALPHA = 0.1
EPSILON = 0.1
EPISODES = 500

# Create a grid environment with obstacles
def create_grid():
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    obstacles = random.sample([(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)], OBSTACLE_COUNT)
    for (i, j) in obstacles:
        grid[i, j] = -1  # Mark obstacles with -1
    return grid, obstacles

# Ensure start and goal are far apart
def initialize_positions(grid):
    while True:
        start = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
        goal = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
        if grid[start] == 0 and grid[goal] == 0 and np.abs(start[0] - goal[0]) + np.abs(start[1] - goal[1]) > GRID_SIZE // 2:
            break
    return start, goal

# Step function for the agent's action
def step(state, action, grid, goal):
    next_state = (state[0] + action[0], state[1] + action[1])
    if 0 <= next_state[0] < GRID_SIZE and 0 <= next_state[1] < GRID_SIZE:
        if grid[next_state] == -1:
            return state, -10  # Obstacle penalty
        elif next_state == goal:
            return next_state, 100  # Goal reward
        else:
            return next_state, -1  # Step penalty
    return state, -1  # Boundary penalty

# Q-Learning (Model-Free RL)
def q_learning(grid, start, goal):
    Q = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
    positions = []  # Store positions for visualization
    explored_states = set()  # Track explored states

    start_time = time.time()
    steps = 0

    for episode in range(EPISODES):
        state = start
        while state != goal:
            steps += 1
            positions.append((episode, state))  # Track the agent's movement for visualization
            explored_states.add(state)
            if random.uniform(0, 1) < EPSILON:
                action_idx = random.choice(range(len(ACTIONS)))
            else:
                action_idx = np.argmax(Q[state[0], state[1]])
            action = ACTIONS[action_idx]
            next_state, reward = step(state, action, grid, goal)
            Q[state[0], state[1], action_idx] += ALPHA * (
                reward + GAMMA * np.max(Q[next_state[0], next_state[1]]) - Q[state[0], state[1], action_idx]
            )
            state = next_state
            if state == goal:
                break  # Terminate episode if goal is reached

    elapsed_time = time.time() - start_time
    explored_percentage = len(explored_states) / (GRID_SIZE * GRID_SIZE) * 100
    return Q, positions, elapsed_time, steps, explored_percentage

# SARSA (On-Policy RL)
def sarsa(grid, start, goal):
    Q = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
    positions = []  # Store positions for visualization
    explored_states = set()  # Track explored states

    start_time = time.time()
    steps = 0

    for episode in range(EPISODES):
        state = start
        action_idx = np.argmax(Q[state[0], state[1]]) if random.uniform(0, 1) > EPSILON else random.choice(range(len(ACTIONS)))
        while state != goal:
            steps += 1
            positions.append((episode, state))  # Track the agent's movement for visualization
            explored_states.add(state)
            action = ACTIONS[action_idx]
            next_state, reward = step(state, action, grid, goal)
            next_action_idx = np.argmax(Q[next_state[0], next_state[1]]) if random.uniform(0, 1) > EPSILON else random.choice(range(len(ACTIONS)))
            Q[state[0], state[1], action_idx] += ALPHA * (
                reward + GAMMA * Q[next_state[0], next_state[1], next_action_idx] - Q[state[0], state[1], action_idx]
            )
            state, action_idx = next_state, next_action_idx
            if state == goal:
                break  # Terminate episode if goal is reached

    elapsed_time = time.time() - start_time
    explored_percentage = len(explored_states) / (GRID_SIZE * GRID_SIZE) * 100
    return Q, positions, elapsed_time, steps, explored_percentage

# Visualization function for the agent's movement
def visualize_movement(grid, positions, start, goal, title):
    fig, ax = plt.subplots()
    grid_show = np.ones((GRID_SIZE, GRID_SIZE, 3))
    grid_show[grid == -1] = [1, 1, 1]  # Obstacles in white
    grid_show[start] = [1, 0, 0]  # Start in red
    grid_show[goal] = [1, 0, 0]   # Goal in red
    mat = ax.matshow(grid_show)

    def update(frame):
        episode, (x, y) = positions[frame]
        if (x, y) != start and (x, y) != goal:
            grid_show[x, y] = [0, 1, 0]  # Path traversed in green
        mat.set_array(grid_show)
        ax.set_title(f"{title} - Episode: {episode}")
        return [mat]

    # Assign animation to a variable to keep a reference
    anim = animation.FuncAnimation(fig, update, frames=len(positions), repeat=False, interval=50)
    
    # Save the animation
    anim.save(f"{title}.mp4", writer="ffmpeg", fps=20)
    plt.close(fig)  # Close the figure to avoid display if not needed

# Main function to run Q-Learning and SARSA
def main():
    # Setup grid and positions
    grid, obstacles = create_grid()
    start, goal = initialize_positions(grid)
    print(f"Start: {start}, Goal: {goal}")

    # Run Q-Learning and collect performance metrics
    Q_qlearning, positions_qlearning, time_qlearning, steps_qlearning, explored_qlearning = q_learning(grid, start, goal)
    # print(f"Q-Learning: Time Taken = {time_qlearning:.2f}s, Steps = {steps_qlearning}, Explored = {explored_qlearning:.2f}%")
    # visualize_movement(grid, positions_qlearning, start, goal, "Q-Learning")

    # Run SARSA and collect performance metrics
    Q_sarsa, positions_sarsa, time_sarsa, steps_sarsa, explored_sarsa = sarsa(grid, start, goal)
    print(f"SARSA: Time Taken = {time_sarsa:.2f}s, Steps = {steps_sarsa}, Explored = {explored_sarsa:.2f}%")
    visualize_movement(grid, positions_sarsa, start, goal, "SARSA")

main()
