import pygame
from board.Pedistal import Pedistal
import random

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

class Board:
    def __init__(self):
        self.board_height = 400
        self.board_width = 800
        self.screen_size = (1000, 600)
        self.pedistal_size = 20
        self.robot_pos = ()
        self.screen = pygame.display.set_mode(self.screen_size)
        self.rect = self.draw_board()
        self.pedistal_list = {
            "white":[],
            "green":[],
            "red":[]
        }
        self.pedistal_types = [
            {
                "type":"white",
                "color_rgb" : WHITE,
                "quantity" : 3
            },
            {
                "type":"green",
                "color_rgb" : GREEN,
                "quantity" : 3
            },
            {
                "type":"red",
                "color_rgb" : RED,
                "quantity" : 1
            }   
        ]

    def format_pedestal_list(self,peds):
        formated_ped = Pedistal()
        formated_list = []
        for ped in peds:
            formated_ped.pos = (abs(ped[2][0])*50, abs(ped[2][1])*50) #format position
            if ped[1] == 'Red_Pedestal':
                formated_ped.color = (255,0,0)  #format label
            elif ped[1] == 'Green_Pedestal':
                formated_ped.color = (0,255,0)  #format label
            elif ped[1] == 'White_Pedestal':
                formated_ped.color = (255,255,255)  #format label
            formated_list.append(formated_ped)
        return formated_list
        

    def update_pedestal_list(self,peds):
        for ped in peds:
            if ped.color == (255,255,255):
                self.pedistal_list['white'].append(ped)
            if ped.color == (0,255,0):
                self.pedistal_list['green'].append(ped)
            if ped.color == (255,0,0):
                self.pedistal_list['red'].append(ped)

    def draw_board(self):
        x, y = (pygame.display.get_window_size())
        x, y = (x-self.board_width)/2, (y-self.board_height)/2
        return pygame.draw.rect(self.screen, BLACK, [x,y,self.board_width,self.board_height])

    def draw_pedistals(self):
        concat_ped_list = []
        for ped_list in self.pedistal_list.values():
            concat_ped_list += ped_list
        for ped in concat_ped_list:
            ped.rect = pygame.draw.circle(self.screen, ped.color, ped.pos, self.pedistal_size)
            for next_ped in ped.next:
                pygame.draw.line(self.screen, next_ped.color, ped.pos, next_ped.pos)

    def populate_pedistals(self):
        self.pedistal_list = {}
        for ped_type in self.pedistal_types:
            for x in range(0, ped_type["quantity"]):
                ped = Pedistal()
                ped_x = random.randint(self.rect.left + self.pedistal_size, self.rect.right - self.pedistal_size)
                ped_y = random.randint(self.rect.top + self.pedistal_size, self.rect.bottom - self.pedistal_size)
                ped.pos = ped_x,ped_y 
                ped.color = ped_type["color_rgb"]
                if ped_type["type"] not in self.pedistal_list:
                    self.pedistal_list[ped_type["type"]] = []
                self.pedistal_list[ped_type["type"]].append(ped)

    def make_ped_graph(self):
        #print(self.pedistal_list)
        print(self.pedistal_list['red'])
        for white_ped in self.pedistal_list['white']:
            white_ped.next = self.pedistal_list['green']
        for green_ped in self.pedistal_list['green']:
            green_ped.next = self.pedistal_list['red']
            green_ped.prev = self.pedistal_list['white']
        for red_ped in self.pedistal_list['red']:
            red_ped.prev = self.pedistal_list['green']    