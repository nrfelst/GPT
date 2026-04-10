import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#size of the grid is 10x10
GRID_SIZE = 10
#8 squares repsent good feeling 
NUM_REWARDS = 8
#8 squares repsent bad feeling 
NUM_PENALTIES = 8
#the life of the ai, one episode = one full navigation of the grid 
EPISODES = 500
#hard limit on how many steps can be taken in one episode of exploing the grid
MAX_STEPS = 200
# How strongly new expirences overwirte old belifs, on a scale of 0-1.
#1.0= new expirences comply replaces old memory, 0.0 means nothign is every learned, 0.1 means each expirences nudges beleif by 10%
LEARNING_RATE = 0.1
#quatitiave repsentation of how much ai values future rewards v immediate one 
#1.0= future rewards matter just as much as right now  
#0.0= only care about the immediate next reward, ignore the future entirely 
#0.95= future rewards are worth 95% as much as present ones  
DISCOUNT = 0.95
#Epsilon controls the explore/exploit balance.
#Starting at 1.0 means agent moves completely random
EPSILON_START = 1.0
#Floor of epsilon
#0.05 means even with learned expirences at episode 500, agent must move at random 5% of the time 
EPSILON_END = 0.05
#After each episode epsilon gets multiplied by this number 
EPSILON_DECAY = 0.5

#the ai agent can move up, down, left, right
ACTIONS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

def build_environment():
    #create grid/ enviroment:10x10 matrix filled with zeros, every cell is neutral- no good, no bad
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    # Set of collections automatically rejects duplicats, acts as a memory 
    positions = set()
    ##function that exist only within the build_enviroment cannot be called elsewhere 
    #Param- value, what to write into the grid cell (1=reward, -1= penalty)
    #Param- count, how many of that item to place 
    def place(value, count):
        #placed starts at zero
        placed = 0
        #loop keeps running until it reaches count
        #using the while ensure non-predetermined runtime 
        #counter only goes up on a successful placement
        while placed < count:
            #r,c are random nums between 0-9 fired simultaneosly.
            # r=row, c=col
            # used for canidate location 
            r, c = np.random.randint(0, GRID_SIZE, size=2)
            #2 condistions before writing into gid spot (r,c)
            #cnd 1: (r,c) not in positions ensuring nothing has been written in yet 
            #cnd2: not (r=5,c=5) this is the starting position for the agent 
            if (r, c) not in positions and not (r == 5 and c == 5):
                #write 1 or -1 in position(r,c)
                # increment counter 
                grid[r][c] = value
                positions.add((r, c))
                placed += 1
    ##call place twice first for rewards, then for penalties
    place(1, NUM_REWARDS)
    place(-1, NUM_PENALTIES)
    #return finished grid for rest of the program 
    return grid

#creates agent brain- 3d matrix of zeros
#for each cell on grid 4 values stored repsenting each direction 
q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))
#initlizes epsilon 
epsilon = EPSILON_START
#empty array the stores total rewards earned at each episode 
episode_rewards = []
# a 10x10 grid acting as a foot tracker 
#each time a grid a cell is visited, that cell goes up by 1 
visit_map = np.zeros((GRID_SIZE, GRID_SIZE))
# an empty list that will store snapshots of agents behavior every 50 episodes 
behavioral_log = []
#builds the grid 
grid = build_environment()
first_episode_path=[]
last_episode_path=[]
# loop 500 times, each iteration is one episode 
for episode in range(EPISODES):
    # each episode these counters reset 
    row, col = 5, 5
    total_reward = 0
    steps = 0
    episode_positions = []
    #track visted episodes 
    visited_rewards = set()
#contiunes to run until agent meets max steps 
    while steps < MAX_STEPS:
        # describes explore/exploit decision
        #number chosen at random between 0 and 1
        #if num < epsilon, random movement 
        #other wise look up the q-table and pick acton with highest value
        if np.random.random() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(q_table[row, col])

        #tracks actions
        #Returns actions change
        #example action 1 returns (-1,0) for one movement up 
        dr, dc = ACTIONS[action]
        #calculate new position by adding direction to current position 
        new_row = np.clip(row + dr, 0, GRID_SIZE - 1)
        new_col = np.clip(col + dc, 0, GRID_SIZE - 1)

        #value in cell
        cell = (new_row,new_col)
        # intitaly set reward to 0
        reward = 0
        if grid[new_row][new_col] == 1 and cell not in visited_rewards:
            reward = 1
            visited_rewards.add(cell)
        elif grid[new_row][new_col] == -1 and cell not in visited_rewards:
            reward = -1 
            visited_rewards.add(cell)
        
        
        
        total_reward += reward

        #Q-Learning update
        #looks up what agent currently belives about taking action form postion, prior action belief
        old_q = q_table[row, col, action]
        #look at new position and find highest Q-value(best possible move)
        future_q = np.max(q_table[new_row, new_col])
        #Q-learning formula 
        #reward: what actually happened 
        #DISCOUNT*future_q: the discounted value of the best future move 
        #reward + DISCOUNT*future_q: total value of this action(now+future)
        #old_q: how wrong the old belief was 
        q_table[row, col, action] = old_q + LEARNING_RATE * (
            reward + DISCOUNT * future_q - old_q
        )

        #adds 1 to food print track for cell
        visit_map[new_row][new_col] += 1
        #logs this position and episdoe 
        #moves agent 
        #increments step counter 
        episode_positions.append((new_row, new_col))
        row, col = new_row, new_col
        steps += 1

    ##track first episode movements 
    if episode == 0:
        first_episode_path = episode_positions.copy()
    
    #track last episode path 
    if episode == EPISODES-1:
        last_episode_path=episode_positions.copy()
    #after each episode shrink epsilon by epsilon decay rate 
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    #save episodes total rewards 
    episode_rewards.append(total_reward)

    # every 50 episodes take snapshot, save it as a dictionary with four pieces of data
    #episode: which episode
    #avg_reward: average reward ove the last 50 episodes 
    #current epsilon 
    #unique_positions: how many differt cells were visted this episode
    if episode % 50 == 0:
        snapshot={
            "episode": episode,
            "avg_reward": np.mean(episode_rewards[-50:]) if episode > 0 else 0,
            "epsilon": round(epsilon, 3),
            "unique_positions": len(set(episode_positions))
        }
        #save snapshot in behavioral_log
        behavioral_log.append(snapshot)


## creat subplot that shows first path and last path 
def draw_path_unique_lines(path, title):
    # Environment colors
    grid_img = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r][c] == 1:            # reward
                grid_img[r][c] = [0, 0.8, 0]      # green
            elif grid[r][c] == -1:         # penalty
                grid_img[r][c] = [0.8, 0, 0]      # red
            else:                           # empty
                grid_img[r][c] = [0.9, 0.9, 0.9]  # light grey
    grid_img[5][5] = [0, 0, 0.8]  # starting position blue

    plt.imshow(grid_img)
    plt.axis('off')

    # Track drawn lines to avoid duplicates
    drawn_lines = set()

    # Draw lines connecting steps
    for i in range(len(path) - 1):
        r, c = path[i]
        next_r, next_c = path[i + 1]

        # Only draw line if this step hasn't been drawn yet
        line_id = (r, c, next_r, next_c)
        if line_id not in drawn_lines:
            intensity = 0.3 + 0.7 * (i / len(path))  # dark → light
            line_color = [intensity]*3  # gray fading
            plt.plot([c, next_c], [r, next_r], color=line_color, linewidth=2)
            drawn_lines.add(line_id)

    plt.title(title)
#Print behavior 
#loops through 10 snapshots prints each on as a row 
print("\n── Behavioral Development Log ──")
print(f"{'Episode':<12}{'Avg Reward':<15}{'Exploration':<15}{'Unique Cells'}")
print("-" * 55)
for entry in behavioral_log:
    print(f"{entry['episode']:<12}{entry['avg_reward']:<15.2f}{entry['epsilon']:<15}{entry['unique_positions']}")
#reward over time 
plt.figure(figsize=(14, 8))
plt.subplot(2, 2, 1)
smoothed = np.convolve(episode_rewards, np.ones(20)/20, mode='valid')
plt.plot(episode_rewards, color='steelblue')
plt.title("Reward Development Over Time")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
#Heatmap: takes the visted_map built up and renders it as a heat map 
#Black->red->yellow->white, repsents the priority order of vists
#brighter colors= more visits 
plt.subplot(2, 2, 2)
plt.imshow(visit_map, cmap='hot', interpolation='nearest')
plt.colorbar(label='Visit Frequency')
plt.title("Emergent Territory Preference")
#creates a new 10x10x3 grid 
#3 resents RGB color channels. assigns each cell a color
# [0,0.8,0] - green reward 
#[0.8,0,0] - red penalty 
#[0.9,0.9,0.9]- light grey empty 
#[0,0,0.8]- blue starting position
plt.subplot(2, 2, 3)
display_grid = np.zeros((GRID_SIZE, GRID_SIZE, 3))
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        if grid[r][c] == 1:
            display_grid[r][c] = [0, 0.8, 0]
        elif grid[r][c] == -1:
            display_grid[r][c] = [0.8, 0, 0]
        else:
            display_grid[r][c] = [0.9, 0.9, 0.9]
display_grid[5][5] = [0, 0, 0.8]
plt.imshow(display_grid)
plt.title("Environment Layout")
green = mpatches.Patch(color='green', label='Reward')
red = mpatches.Patch(color='red', label='Penalty')
blue = mpatches.Patch(color='blue', label='Start')
plt.legend(handles=[green, red, blue], loc='upper right', fontsize=7)




#Data table displaying behavioral_log
table_data=[
    [entry['episode'], f"{entry['avg_reward']:.2f}", entry['epsilon'], entry['unique_positions']]
    for entry in behavioral_log
]
plt.subplot(2,2,4)
plt.axis('off')
#Colums of data table 
columns = ["Episode", "Avg Reward", "Epsilon", "Unique Cells"]
#ax is the axes object for the subplot 
#table() creates a table in that axis 
plt.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
plt.title("Behavioral Snapshots Every 50 Episodes")

# save and finish 
plt.tight_layout()
plt.savefig("emergent_behavior.png", dpi=150)
plt.show()
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
draw_path_unique_lines(first_episode_path, "First Episode Path")

plt.subplot(1, 2, 2)
draw_path_unique_lines(last_episode_path, "Final Episode Path")

plt.tight_layout()
plt.show()
print("\nDone. emergent_behavior.png saved.")