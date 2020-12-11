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
        self.rows = 10
        self.cols = 10
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
        self.body = [(2,2), (2,1), (2,0)]
        self.head_value = 3
        self.decay = 1
        # Food params
        self.food = (0,0)
        self.food_value = 10
        #self.score = 0
        self.avg_score = [0 for _ in range(100)]
        self.avg_score_count = 0
        # Others
        self.step_key = True
        self.speed = 1
        # AI params
        self.ai_enabled = False

        self.agent = SnakeAgent(self.cols, self.rows, 4)

    def set_board(self, r, c, val):
        self.board[self.rows - r - 1, c] = val
    def set_food(self, rc):
        self.food = (self.rows - rc[0] - 1, rc[1])

    def reset(self):
        # Erase the board (for AI)
        self.board = torch.zeros(self.rows, self.cols)
        self.action = torch.full((1,1), 2)
        self.t = 0
        self.score = 0
        # Erase old food
        self.fill_rect(self.food[0], self.food[1], self.colors["EMPTY"])
        # Erase old snake body
        for r, c in self.body:
            self.fill_rect(r, c, self.colors["EMPTY"])

        self.body = [(2,2), (2,1), (2,0)]
        self.direction = np.random.randint(0, high=4) if self.ai_enabled else 2
        # Fill in snake body
        for r, c in self.body:
            self.set_board(r, c, self.decay)
            self.fill_rect(r, c, self.colors["BODY"])
        self.set_board(self.body[0][0], self.body[0][1], self.head_value)
        self.new_food()

    # Doesn't do anything atm
    def new_body(self):
        head = np.array([np.random.choice(self.rows), np.random.choice(self.cols)])
        neck = head + np.array([0, np.random.choice(3)-1])
        while not self.check_bounds(neck[0], neck[1]):
            neck = head + np.array([0, np.random.choice(3)-1])
        hn = head - neck
        if hn[0] == 0 and hn[1] < 0:
            self.direction = 0
        elif hn[0] > 0 and hn[1] == 0:
            self.direction = 1
        elif hn[0] == 0 and hn[1] > 0:
            self.direction = 2
        elif hn[0] < 0 and hn[1] == 0:
            self.direction = 3

        return [head, neck]
        
    def new_food(self):
        empty_spots = (self.board == 0).nonzero()
        if len(empty_spots) == 0: # if there are no empty spots
            self.reset()
            return

        # Make new food
        empty_idx = np.random.choice(np.arange(len(empty_spots)))
        self.set_food(empty_spots[empty_idx])
        
        # Add food to board
        self.set_board(self.food[0], self.food[1], self.food_value)
        self.fill_rect(self.food[0], self.food[1], self.colors["FOOD"])
        
    def setup(self):
        self.set_update_rate(1/self.speed) # n Hz for value 1/n
        arcade.start_render()

        self.reset()
        
        # Draw board lines
        for r in range(self.rows):
            for c in range(self.cols):
                self.draw_rect(r, c, self.colors["OUTLINE"])

    def on_draw(self):
        pass
    
    def update(self, delta_time):
        ##########################################################################
        state = self.board.view(1, 1, self.board.size()[0], self.board.size()[1])
        if self.ai_enabled:                                                     ##
            self.action = self.agent.action(state)                              ##
        ##########################################################################

        self.step_key = True
        r, c = self.body[0]

        # move the snake forward
        new_head = (r,c)
        # set movement direction
        if self.action == 0 and (self.direction != 2):
            self.direction = 0
        elif self.action == 1 and (self.direction != 3):
            self.direction = 1
        elif self.action == 2 and (self.direction != 0):
            self.direction = 2
        elif self.action == 3 and (self.direction != 1):
            self.direction = 3
        # set new position
        if self.direction == 0:
            new_head = (r,c-1)
        elif self.direction == 1:
            new_head = (r+1,c)
        elif self.direction == 2:
            new_head = (r,c+1)
        elif self.direction == 3:
            new_head = (r-1,c)

        # if going to be (out of bounds) or (ate itself), then died
        died = False
        if not self.check_bounds(new_head[0], new_head[1]) or (new_head[0], new_head[1]) in self.body[1:-1]:
            died = True
        # if ate food
        ate = False
        if new_head[0] == self.food[0] and new_head[1] == self.food[1]:
            ate = True
            self.score += 1

        if not died:
            # Add new head
            self.set_board(r, c, 0.5)
            self.body.insert(0, new_head)
            self.set_board(new_head[0], new_head[1], 1)

            # If not eaten, erase the tail of the snake
            if ate:
                self.new_food()
            else:
                self.fill_rect(self.body[-1][0], self.body[-1][1], self.colors["EMPTY"])
                self.set_board(self.body[-1][0], self.body[-1][1], 0)
                self.body.pop(len(self.body) - 1)
            self.fill_rect(new_head[0], new_head[1], self.colors["BODY"])
                
            self.t += 1
        else:
            self.avg_score.append(self.score)
            self.avg_score.pop(0)
            self.avg_score_count +=1
            if self.avg_score_count%len(self.avg_score) == 0:
                print(str(self.avg_score_count) + ":", np.average(self.avg_score))
            self.reset()

        ##########################################################################
        reward = torch.tensor(-1 if died else 1 if ate else 0).view(1, 1)       ##
        next_state = self.board.view(1, 1, self.board.size()[0], self.board.size()[1]) if not died else None
                                                                                ##
        self.agent.optimize(state, self.action, next_state, reward)             ##
        ##########################################################################
    
    def on_key_press(self, key, modifiers):
        if self.step_key and not self.ai_enabled:
            if key == arcade.key.LEFT:
                self.action[0,0] = 0
                self.step_key = False
            elif key == arcade.key.UP:
                self.action[0,0] = 1
                self.step_key = False
            elif key == arcade.key.RIGHT:
                self.action[0,0] = 2
                self.step_key = False
            elif key == arcade.key.DOWN:
                self.action[0,0] = 3
                self.step_key = False

        if key == arcade.key.A:
            self.ai_enabled = not self.ai_enabled
            self.action = self.direction

        if key == arcade.key.Q:
            self.speed = 2
            self.set_update_rate(1/self.speed)
        elif key == arcade.key.W:
            self.speed = self.rows
            self.set_update_rate(1/self.speed)
        elif key == arcade.key.E:
            self.speed = 200
            self.set_update_rate(1/self.speed)

        if key == arcade.key.L:
            self.agent.toggle_logs()

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