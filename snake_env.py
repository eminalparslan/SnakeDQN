import numpy as np
import cv2
from PIL import Image
from collections import deque

class SnakeEnv:
    # Cells per row/col
    CELL_COUNT = 10
    # Render screen size
    SCREEN_SIZE = 400
    # Penalty for each move to encourage more efficient paths to food
    MOVE_PENALTY = -1
    # Huge penalty for death
    DEATH_PENALTY = 200
    # Reward for eating food (only way to get rewarded)
    FOOD_REWARD = 50
    # 3 for RGB values
    OBSERVATION_SPACE_VALUES = (CELL_COUNT, CELL_COUNT, 3)
    # Number of possible actions
    ACTION_SPACE_SIZE = 4
    # Colors for each type of cell (h: snake head, t: snake tail, a: food)
    colors = {
        "a": (0, 0, 255),
        "h": (255, 0, 0),
        "t": (0, 255, 0)
    }

    def render(self):
        img = self.get_image()
        img = Image.fromarray(img)
        img = img.resize((self.SCREEN_SIZE, self.SCREEN_SIZE), Image.BOX)
        img = cv2.putText(img=np.array(img), text=str(self.score), org=(self.SCREEN_SIZE-50, 50), 
                          fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
                          thickness=2, lineType=2)
        cv2.imshow("Snake", img)
        cv2.waitKey(1)

    def get_image(self):
        img = np.zeros(self.OBSERVATION_SPACE_VALUES, dtype=np.uint8)
        img[self.ax][self.ay] = self.colors["a"]
        img[self.hx][self.hy] = self.colors["h"]
        for tx, ty in self.trail:
            img[tx][ty] = self.colors["t"]
        return img

    def reset(self):
        self.hx, self.hy = 4, 4
        self.vx, self.vy = 0, 0
        self.ax = np.random.randint(0, self.CELL_COUNT)
        self.ay = np.random.randint(0, self.CELL_COUNT)
        self.tail = 5
        self.trail = deque()
        self.episode_step = 0
        self.score = 0
        return self.get_image()

    def step(self, action):
        # Keep track of steps in episode
        self.episode_step += 1
        # Set velocity based on action
        # left
        if action == 0:
            self.vx = -1
            self.vy = 0
        # down
        elif action == 1:
            self.vx = 0
            self.vy = 1
        # right
        elif action == 2:
            self.vx = 1
            self.vy = 0
        # up
        elif action == 3:
            self.vx = 0
            self.vy = -1
        # Add head to tail and limit length
        self.trail.append((self.hx, self.hy))
        while len(self.trail) > self.tail:
            self.trail.popleft()
        # Move based on action
        self.hx += self.vx
        self.hy += self.vy
        # Done iff dead
        done = False
        # Penalty for every move
        reward =  0 - self.MOVE_PENALTY
        # Check for wall collision
        if not 0 <= self.hx < self.CELL_COUNT or not 0 <= self.hy < self.CELL_COUNT:
            reward -= self.DEATH_PENALTY
            done = True
            self.reset()
        # Check for self collision
        if (self.hx, self.hy) in self.trail:
            reward -= self.DEATH_PENALTY
            done = True
        # Check if food eaten
        if (self.ax, self.ay) == (self.hx, self.hy):
            self.tail += 1
            self.score += 1
            self.ax = np.random.randint(0, self.CELL_COUNT)
            self.ay = np.random.randint(0, self.CELL_COUNT)
            reward += self.FOOD_REWARD
            # Make sure food doesn't spawn on snake
            # If it does, search linearly until open spot found
            while (self.ax, self.ay) == (self.hx, self.hy):
                self.ax += 1
                # Wrap around
                if self.ax >= self.CELL_COUNT:
                    self.ax = 0
                    self.ay += 1
                if self.ay >= self.CELL_COUNT:
                    self.ay = 0
        # Make observation
        newObs = self.get_image()
        return newObs, reward, done

