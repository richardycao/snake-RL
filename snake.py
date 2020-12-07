"""
How to use:

`python snake.py` to run the game

Press 'Q' for slow speed (for observing and debugging)
Press 'W' for normal speed (for playing the game as a human)
Press 'E' for fast speed (for training the agent)
Press 'A' to toggle player/agent (enable/disable AI)

If AI is disabled (default),
Press UP, DOWN, LEFT, RIGHT to control the snake

"""

import numpy as np
import torch
import arcade
from time import sleep
from agent import SnakeAgent

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400

class Snake(arcade.Window):
    def __init__(self, width, height):
        super().__init__(width, height)
        arcade.set_background_color(arcade.color.WHITE)
        # Window params
        self.width = width
        self.height = height
        self.top_margin = 20
        self.left_margin = 20
        self.right_margin = 20
        self.bot_margin = 20
        # Board params
        self.rows = 6
        self.cols = 6
        self.rect_width = (self.width - self.left_margin - self.right_margin) / self.cols
        self.rect_height = (self.height - self.top_margin - self.bot_margin) / self.rows
        self.line_width = 1
        self.colors = {
            "OUTLINE": arcade.color.BLACK,
            "EMPTY": arcade.color.WHITE,
            "BODY": arcade.color.GREEN,
            "FOOD": arcade.color.RED
        }
        # Snake params
        self.body = [(2,0), (1,0), (0,0)]
        self.direction = 2 # 1: left, 2: up, 3: right, 4: down
        # Food params
        self.food_r = np.random.randint(0, high=self.rows)
        self.food_c = np.random.randint(0, high=self.cols)
        self.score = 0
        self.avg_score = [0 for _ in range(30)]
        self.avg_score_count = 0
        # Others
        self.step_key = True
        self.speed = 1
        # AI params
        #self.board = np.zeros((self.rows, self.cols)) # 0: empty, 1: snake, 2: food
        self.board = torch.zeros(self.rows, self.cols, dtype=torch.float32) # 0: empty, 1: snake, 2: food
        #self.prev_board = self.board.copy()
        self.action = torch.zeros((1,1)) # 0: no action, 1: left, 2: up, 3: right, 4: down
        self.ai_enabled = False
        self.t = 0

        self.agent = SnakeAgent(self.cols, self.rows, 5)

    def reset(self):
        # Erase the board (for AI)
        self.board = torch.zeros(self.rows, self.cols)
        self.action = torch.zeros((1,1))
        self.t = 0
        self.score = 0
        # Erase old food
        self.fill_rect(self.food_r, self.food_c, self.colors["EMPTY"])
        # Erase old snake body
        for r, c in self.body:
            self.fill_rect(r, c, self.colors["EMPTY"])

        self.body = [(2,0), (1,0), (0,0)]
        # Fill in snake body
        for r, c in self.body:
            self.board[r, c] = 1
            self.fill_rect(r, c, self.colors["BODY"])
        self.direction = 2
        self.new_food()
        
        #self.prev_board = self.board.copy()

    def new_food(self):
        # TODO
        # if board has no empty spots:
        #     self.reset()

        # Erase food from board
        self.board[self.food_r, self.food_c] = 0

        # Make new food
        self.food_r = np.random.randint(0, high=self.rows)
        self.food_c = np.random.randint(0, high=self.cols)
        if (self.food_r, self.food_c) in self.body:
            self.new_food()
        
        # Add food to board
        self.board[self.food_r, self.food_c] = 2
        # Draw food
        self.fill_rect(self.food_r, self.food_c, self.colors["FOOD"])
        
    def setup(self):
        self.set_update_rate(1/self.speed) # n Hz for value 1/n
        arcade.start_render()

        self.reset()
        
        # Draw board lines
        for r in range(self.rows):
            for c in range(self.cols):
                self.draw_rect(r, c, self.colors["OUTLINE"])

    def on_draw(self):
        # Fill in snake head
        self.fill_rect(self.body[0][0], self.body[0][1], self.colors["BODY"])
    
    def update(self, delta_time):
        ##########################################################################
        if self.ai_enabled:                                                     ##
            state = (self.board.view(1, 1, self.board.size()[0], self.board.size()[1]), 
                     torch.tensor(self.direction, dtype=torch.float32).clone().detach().view(1, 1))                                ##
            self.action = self.agent.action(state, self.t)                      ##
        ##########################################################################

        self.step_key = True

        r, c = self.body[0]

        # if ate food
        ate = False
        if r == self.food_r and c == self.food_c:
            ate = True
            self.score += 1
            self.new_food()
        # move the snake forward
        new_head = (r,c)
        # set movement direction
        if self.action == 1 and (self.direction != 3):
            self.direction = 1
        elif self.action == 2 and (self.direction != 4):
            self.direction = 2
        elif self.action == 3 and (self.direction != 1):
            self.direction = 3
        elif self.action == 4 and (self.direction != 2):
            self.direction = 4
        # set new position
        if self.direction == 1:
            new_head = (r,c-1)
        elif self.direction == 2:
            new_head = (r+1,c)
        elif self.direction == 3:
            new_head = (r,c+1)
        elif self.direction == 4:
            new_head = (r-1,c)
        # if going to be (out of bounds) or (ate itself), then died
        died = False
        if not self.check_bounds(new_head[0], new_head[1]) or (new_head[0], new_head[1]) in self.body[1:]:
            died = True

        if not died:
            self.body.insert(0, new_head)
            self.board[new_head[0], new_head[1]] = 1

            # Erase the tail of the snake
            if not ate:
                self.fill_rect(self.body[-1][0], self.body[-1][1], self.colors["EMPTY"])
                self.board[self.body[-1][0], self.body[-1][1]] = 0
                self.body.pop(len(self.body) - 1)
            self.t += 1
        else:
            #print("final score:", self.score)
            self.avg_score.append(self.score)
            self.avg_score.pop(0)
            self.avg_score_count = (self.avg_score_count + 1) % 50
            if self.avg_score_count == 0:
                print("Average score:", np.average(self.avg_score))
            self.reset()

        ##########################################################################
        if self.ai_enabled:                                                     ##
            reward = torch.tensor(-1 if died else (1 if ate else 0).view(1, 1))
            next_state = (self.board.view(1, 1, self.board.size()[0], self.board.size()[1]), 
                            torch.tensor(self.direction, dtype=torch.float32).clone().detach().view(1, 1)) if not died else None
            self.agent.get_memory().push(state, self.action, next_state, reward)##
                                                                                ##
            self.agent.optimize()                                               ##
        ##########################################################################
    
    def on_key_press(self, key, modifiers):
        if self.step_key and not self.ai_enabled:
            if key == arcade.key.LEFT:
                self.action[0,0] = 1
                self.step_key = False
            elif key == arcade.key.UP:
                self.action[0,0] = 2
                self.step_key = False
            elif key == arcade.key.RIGHT:
                self.action[0,0] = 3
                self.step_key = False
            elif key == arcade.key.DOWN:
                self.action[0,0] = 4
                self.step_key = False

        if key == arcade.key.A:
            self.ai_enabled = not self.ai_enabled
            self.action = self.direction

        if key == arcade.key.Q:
            self.speed = 2
            self.set_update_rate(1/self.speed)
        elif key == arcade.key.W:
            self.speed = 10
            self.set_update_rate(1/self.speed)
        elif key == arcade.key.E:
            self.speed = 200
            self.set_update_rate(1/self.speed)

    def draw_rect(self, r, c, color):
        arcade.draw_rectangle_outline(
            center_x = self.left_margin + c*self.rect_width + self.rect_width/2, 
            center_y = self.bot_margin + r*self.rect_height + self.rect_height/2, 
            width = self.rect_width,
            height = self.rect_height,
            color = color,
            border_width = self.line_width)

    def fill_rect(self, r, c, color):
        arcade.draw_rectangle_filled(
            center_x = self.left_margin + c*self.rect_width + self.rect_width/2, 
            center_y = self.bot_margin + r*self.rect_height + self.rect_height/2, 
            width = self.rect_width - self.line_width - 1,
            height = self.rect_height - self.line_width - 1,
            color = color)

    def check_bounds(self, r, c):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return True
        return False

def main():
    game = Snake(SCREEN_WIDTH, SCREEN_HEIGHT)
    game.setup()
    arcade.run()

if __name__ == "__main__":
    main()