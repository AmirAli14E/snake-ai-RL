import gym
from gym import spaces
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import os
from datetime import datetime

class CompetitiveSnakeEnv(gym.Env):
    def __init__(self, width=600, height=600, n_agents=2):
        super(CompetitiveSnakeEnv, self).__init__()
        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.grid_size = 20
        self.max_steps = 1000
        
        # State: [head_x, head_y, food_x, food_y, body1_x, body1_y, length, steps_since_meal, iq, enemy_dist_x, enemy_dist_y]
        self.observation_space = spaces.Box(low=0, high=max(width, height), shape=(11,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        
        # Initialize PyGame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('AI Snake Competition')
        self.clock = pygame.time.Clock()
        self.colors = [(255, 50, 50), (50, 50, 255)]
        self.font = pygame.font.SysFont('consolas', 16)  # فونت کوچک برای امتیازها
        self.title_font = pygame.font.SysFont('consolas', 24, bold=True)
        
        # Game parameters
        self.growth_per_food = 3  # هر غذا ۳ واحد به طول مار اضافه می‌کند
        self.base_iq = 0.5
        self.iq_increase = 0.05  # افزایش هوش با هر غذا
        
        self.reset()

    def reset(self):
        self.steps = 0
        self.snakes = [{
            'body': [(200 + i*200, 200 + j*self.grid_size) for j in range(3)],
            'direction': random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT']),
            'score': 0,
            'alive': True,
            'steps_since_meal': 0,
            'iq': self.base_iq,
            'kills': 0,
            'color': self.colors[i]
        } for i in range(self.n_agents)]
        
        self.food = self._spawn_food()
        self.last_winner = None
        return self._get_states()

    def _spawn_food(self):
        valid_positions = []
        for x in range(1, (self.width//self.grid_size)-1):
            for y in range(1, (self.height//self.grid_size)-1):
                pos = (x*self.grid_size, y*self.grid_size)
                if all(pos not in snake['body'] for snake in self.snakes):
                    valid_positions.append(pos)
        return random.choice(valid_positions) if valid_positions else (100, 100)

    def _get_states(self):
        states = []
        for i in range(self.n_agents):
            if not self.snakes[i]['alive']:
                states.append(np.zeros(self.observation_space.shape, dtype=np.float32))
                continue
                
            head = self.snakes[i]['body'][0]
            
            # Find closest enemy head
            enemy_dist = [0, 0]
            for j in range(self.n_agents):
                if i != j and self.snakes[j]['alive']:
                    enemy_head = self.snakes[j]['body'][0]
                    enemy_dist = [enemy_head[0] - head[0], enemy_head[1] - head[1]]
                    break
            
            state = [
                head[0], head[1],  # head position (2)
                self.food[0], self.food[1],  # food position (2)
                *self.snakes[i]['body'][1],  # first body segment (2)
                len(self.snakes[i]['body']),  # length (1)
                self.snakes[i]['steps_since_meal'],  # hunger (1)
                self.snakes[i]['iq'],  # intelligence (1)
                *enemy_dist  # enemy direction (2)
            ]  # Total: 11 features
            states.append(np.array(state, dtype=np.float32))
        return states

    def _render_frame(self):
        self.screen.fill((30, 30, 30))
        
        # Draw grid
        for x in range(0, self.width, self.grid_size):
            pygame.draw.line(self.screen, (50, 50, 50), (x, 0), (x, self.height))
        for y in range(0, self.height, self.grid_size):
            pygame.draw.line(self.screen, (50, 50, 50), (0, y), (self.width, y))
        
        # Draw food
        pygame.draw.rect(self.screen, (100, 255, 100), 
                         (*self.food, self.grid_size-2, self.grid_size-2))
        
        # Draw snakes with gradient based on length
        for snake in self.snakes:
            if not snake['alive']:
                continue
                
            for j, segment in enumerate(snake['body']):
                # Longer snakes appear brighter
                alpha = min(255, 100 + j*5 + snake['iq']*50)
                color = (
                    min(255, snake['color'][0] + alpha//3),
                    min(255, snake['color'][1] + alpha//3),
                    min(255, snake['color'][2] + alpha//3)
                )
                pygame.draw.rect(self.screen, color, 
                                 (*segment, self.grid_size-2, self.grid_size-2))
        
        # Draw scoreboard
        score_text = []
        for i, snake in enumerate(self.snakes):
            status = "ALIVE" if snake['alive'] else "DEAD"
            score_text.append(f"Agent {i}: {snake['score']} (IQ: {snake['iq']:.1f}, Kills: {snake['kills']}) {status}")
        
        for i, text in enumerate(score_text):
            text_surface = self.font.render(text, True, self.colors[i])
            self.screen.blit(text_surface, (10, 10 + i*20))
        
        # Show current winner
        if self.last_winner is not None:
            winner_text = self.title_font.render(f"Winner: Agent {self.last_winner}", True, self.colors[self.last_winner])
            self.screen.blit(winner_text, (self.width//2 - 100, 10))
        
        pygame.display.flip()
        self.clock.tick(15)

    def step(self, actions):
        self.steps += 1
        rewards = [0] * self.n_agents
        dones = [False] * self.n_agents
        infos = [{} for _ in range(self.n_agents)]
        
        # Process movements
        for i in range(self.n_agents):
            if not self.snakes[i]['alive']:
                continue
                
            self.snakes[i]['steps_since_meal'] += 1
            
            # Smarter directional change based on IQ
            if random.random() < self.snakes[i]['iq']:
                actions[i] = self._smart_action(i, actions[i])
            
            # Change direction
            directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
            new_dir = directions[actions[i]]
            current_dir = self.snakes[i]['direction']
            
            if not ((current_dir == 'UP' and new_dir == 'DOWN') or
                    (current_dir == 'DOWN' and new_dir == 'UP') or
                    (current_dir == 'LEFT' and new_dir == 'RIGHT') or
                    (current_dir == 'RIGHT' and new_dir == 'LEFT')):
                self.snakes[i]['direction'] = new_dir
        
        # Move snakes and check collisions
        for i in range(self.n_agents):
            if not self.snakes[i]['alive']:
                continue
                
            head = self.snakes[i]['body'][0]
            if self.snakes[i]['direction'] == 'UP':
                new_head = (head[0], head[1] - self.grid_size)
            elif self.snakes[i]['direction'] == 'DOWN':
                new_head = (head[0], head[1] + self.grid_size)
            elif self.snakes[i]['direction'] == 'LEFT':
                new_head = (head[0] - self.grid_size, head[1])
            else:  # RIGHT
                new_head = (head[0] + self.grid_size, head[1])
            
            # Wall collision
            if (new_head[0] < 0 or new_head[0] >= self.width or
                new_head[1] < 0 or new_head[1] >= self.height):
                self._handle_death(i)
                rewards[i] = -30
                self._award_killer(i)
                continue
                
            # Self or enemy collision
            collision = False
            killer = None
            
            for j in range(self.n_agents):
                if new_head in self.snakes[j]['body'][:-1]:  # Skip tail tips
                    collision = True
                    if i != j:  # Killed by enemy
                        killer = j
                    break
            
            if collision:
                self._handle_death(i)
                rewards[i] = -30
                if killer is not None:
                    self._award_killer(i, killer)
                continue
            
            # Food consumption
            if new_head == self.food:
                # Grow snake
                for _ in range(self.growth_per_food):
                    self.snakes[i]['body'].insert(0, new_head)
                
                self.snakes[i]['score'] += 10
                self.snakes[i]['iq'] = min(2.0, self.snakes[i]['iq'] + self.iq_increase)
                self.snakes[i]['steps_since_meal'] = 0
                rewards[i] = 50 + len(self.snakes[i]['body'])  # Longer snakes get more reward
                self.food = self._spawn_food()
            else:
                self.snakes[i]['body'].insert(0, new_head)
                self.snakes[i]['body'].pop()
                
                # Hunger penalty
                if self.snakes[i]['steps_since_meal'] > 100:
                    rewards[i] = -2
        
        # Check end conditions
        alive_snakes = [i for i in range(self.n_agents) if self.snakes[i]['alive']]
        
        if len(alive_snakes) <= 1 or self.steps >= self.max_steps:
            dones = [True] * self.n_agents
            if len(alive_snakes) == 1:
                self.last_winner = alive_snakes[0]
                rewards[alive_snakes[0]] += 100  # Big reward for winning
                self.snakes[alive_snakes[0]]['score'] += 50
            
            # Final rewards based on performance
            for i in range(self.n_agents):
                if self.snakes[i]['alive']:
                    rewards[i] += 5 * self.snakes[i]['score']
        
        self._render_frame()
        return self._get_states(), rewards, dones, infos

    def _handle_death(self, snake_idx):
        self.snakes[snake_idx]['alive'] = False
        # Leave corpse for collision
        self.snakes[snake_idx]['body'] = self.snakes[snake_idx]['body'][:3]  # Keep first 3 segments

    def _award_killer(self, victim_idx, killer_idx=None):
        if killer_idx is None:  # Death by wall
            # Find closest enemy
            victim_head = self.snakes[victim_idx]['body'][0]
            min_dist = float('inf')
            killer_idx = None
            
            for i in range(self.n_agents):
                if i != victim_idx and self.snakes[i]['alive']:
                    dist = ((self.snakes[i]['body'][0][0] - victim_head[0])**2 + 
                           (self.snakes[i]['body'][0][1] - victim_head[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        killer_idx = i
            
            if killer_idx is None:
                return
        
        self.snakes[killer_idx]['kills'] += 1
        self.snakes[killer_idx]['score'] += 30
        self.snakes[killer_idx]['iq'] = min(2.0, self.snakes[killer_idx]['iq'] + 0.1)

    def _smart_action(self, agent_idx, current_action):
        head = self.snakes[agent_idx]['body'][0]
        
        # Food seeking
        dx_food = self.food[0] - head[0]
        dy_food = self.food[1] - head[1]
        
        # Enemy avoidance
        enemy_dist = [0, 0]
        for i in range(self.n_agents):
            if i != agent_idx and self.snakes[i]['alive']:
                enemy_head = self.snakes[i]['body'][0]
                enemy_dist = [head[0] - enemy_head[0], head[1] - enemy_head[1]]
                break
        
        # Combine strategies based on IQ
        if random.random() < self.snakes[agent_idx]['iq']:
            # Smart food seeking
            if abs(dx_food) > abs(dy_food):
                food_action = 1 if dx_food > 0 else 3  # RIGHT or LEFT
            else:
                food_action = 2 if dy_food > 0 else 0  # DOWN or UP
            
            # Enemy avoidance
            if abs(enemy_dist[0]) < 100 and abs(enemy_dist[1]) < 100:
                if abs(enemy_dist[0]) > abs(enemy_dist[1]):
                    avoid_action = 3 if enemy_dist[0] > 0 else 1  # Move away horizontally
                else:
                    avoid_action = 0 if enemy_dist[1] > 0 else 2  # Move away vertically
                
                # Blend strategies based on hunger
                hunger_factor = min(1.0, self.snakes[agent_idx]['steps_since_meal'] / 50)
                return avoid_action if random.random() < hunger_factor else food_action
            
            return food_action
        
        return current_action

class AdvancedDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(AdvancedDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

def train_competitive_snakes():
    env = CompetitiveSnakeEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create models
    agents = [AdvancedDQN(state_size, action_size) for _ in range(env.n_agents)]
    optimizers = [optim.Adam(agent.parameters(), lr=0.0005) for agent in agents]
    criterion = nn.SmoothL1Loss()
    
    # Experience replay
    memories = [deque(maxlen=50000) for _ in range(env.n_agents)]
    batch_size = 128
    gamma = 0.98
    
    # Exploration
    epsilons = [1.0] * env.n_agents
    epsilon_min = 0.05
    epsilon_decay = 0.998
    
    # Training stats
    scores = [[] for _ in range(env.n_agents)]
    wins = [0] * env.n_agents
    learning_progress = []
    
    # Create model directory
    model_dir = "snake_models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Training loop
    for episode in range(2000):
        states = env.reset()
        states = [torch.FloatTensor(state) for state in states]
        dones = [False] * env.n_agents
        episode_rewards = [0] * env.n_agents
        
        while not all(dones):
            actions = []
            for i in range(env.n_agents):
                if dones[i]:
                    actions.append(0)
                    continue
                    
                if random.random() < epsilons[i]:
                    actions.append(random.randrange(action_size))
                else:
                    with torch.no_grad():
                        q_values = agents[i](states[i].unsqueeze(0))
                        actions.append(torch.argmax(q_values).item())
            
            next_states, rewards, dones, _ = env.step(actions)
            next_states = [torch.FloatTensor(state) for state in next_states]
            
            for i in range(env.n_agents):
                episode_rewards[i] += rewards[i]
                if not dones[i]:
                    memories[i].append((states[i], actions[i], rewards[i], next_states[i], dones[i]))
            
            states = next_states
            
            # Train all agents
            for i in range(env.n_agents):
                if len(memories[i]) >= batch_size:
                    minibatch = random.sample(memories[i], batch_size)
                    
                    states_batch = torch.stack([x[0] for x in minibatch])
                    actions_batch = torch.LongTensor([x[1] for x in minibatch])
                    rewards_batch = torch.FloatTensor([x[2] for x in minibatch])
                    next_states_batch = torch.stack([x[3] for x in minibatch])
                    dones_batch = torch.BoolTensor([x[4] for x in minibatch])
                    
                    current_q = agents[i](states_batch).gather(1, actions_batch.unsqueeze(1))
                    
                    with torch.no_grad():
                        next_q = agents[i](next_states_batch).max(1)[0]
                        target = rewards_batch + (~dones_batch) * gamma * next_q
                    
                    loss = criterion(current_q.squeeze(), target)
                    optimizers[i].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agents[i].parameters(), 1.0)
                    optimizers[i].step()
        
        # Update exploration rates
        for i in range(env.n_agents):
            if epsilons[i] > epsilon_min:
                epsilons[i] *= epsilon_decay
            
            scores[i].append(env.snakes[i]['score'])
        
        # Track wins
        if env.last_winner is not None:
            wins[env.last_winner] += 1
        
        # Save learning progress
        learning_progress.append({
            'episode': episode,
            'scores': [s[-1] for s in scores],
            'wins': wins.copy(),
            'epsilons': epsilons.copy()
        })
        
        # Print and save periodically
        if episode % 50 == 0:
            print(f"\nEpisode {episode}:")
            for i in range(env.n_agents):
                print(f"  Agent {i} - Score: {env.snakes[i]['score']} | "
                      f"IQ: {env.snakes[i]['iq']:.1f} | "
                      f"Kills: {env.snakes[i]['kills']} | "
                      f"Epsilon: {epsilons[i]:.2f}")
            
            print(f"  Wins: {wins}")
            
            # Save models
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for i, agent in enumerate(agents):
                torch.save(agent.state_dict(), f"{model_dir}/agent_{i}_ep_{episode}_{timestamp}.pth")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Scores
    plt.subplot(2, 2, 1)
    for i in range(env.n_agents):
        plt.plot(scores[i], label=f'Agent {i}', color=env.colors[i])
    plt.title('Scores per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    
    # Moving average scores
    plt.subplot(2, 2, 2)
    window_size = 50
    for i in range(env.n_agents):
        moving_avg = [np.mean(scores[i][max(0, j-window_size):j+1]) 
                     for j in range(len(scores[i]))]
        plt.plot(moving_avg, label=f'Agent {i} (MA{window_size})', 
                 color=env.colors[i], linestyle='--')
    plt.title(f'Moving Average Scores (Window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    
    # Wins
    plt.subplot(2, 2, 3)
    plt.bar(range(env.n_agents), wins, color=[env.colors[i] for i in range(env.n_agents)])
    plt.title('Total Wins')
    plt.xlabel('Agent')
    plt.ylabel('Wins')
    
    # Epsilon decay
    plt.subplot(2, 2, 4)
    for i in range(env.n_agents):
        epsilons = [x['epsilons'][i] for x in learning_progress]
        plt.plot(epsilons, label=f'Agent {i}', color=env.colors[i])
    plt.title('Exploration Rate Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Final evaluation
    print("\nFinal Results:")
    print(f"Total episodes: {len(scores[0])}")
    print(f"Minimum learning episodes: {int(100 / (1 - epsilon_min))}")  # تخمین حداقل یادگیری
    
    for i in range(env.n_agents):
        print(f"Agent {i}:")
        print(f"  Max score: {max(scores[i])}")
        print(f"  Average last 100 scores: {np.mean(scores[i][-100:])}")
        print(f"  Total wins: {wins[i]}")
        
        # Save final models
        torch.save(agents[i].state_dict(), f"{model_dir}/agent_{i}_final.pth")
    
    print("\nModels saved in 'snake_models' directory")

if __name__ == "__main__":
    train_competitive_snakes()