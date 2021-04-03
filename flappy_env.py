import gym
from gym import spaces

from itertools import cycle
import random
import sys
import os

import pygame
from pygame.locals import *

import flappy
import numpy as np

# GLOBALS
FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
PLAYERS_FILES = ('assets/sprites/redbird-upflap.png', 'assets/sprites/redbird-midflap.png', 'assets/sprites/redbird-downflap.png')
BACKGROUND_FILE= 'assets/sprites/background-day.png'
PIPES_LIST = 'assets/sprites/pipe-green.png'
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

try:
	xrange
except NameError:
	xrange = range

class FlappyEnv(gym.Env):
	"""Custom Environment that follows gym interface"""
	metadata = {'render.modes': ['human']}

	def __init__(self, server):
		super(FlappyEnv, self).__init__()

		if server == True:
			os.environ["SDL_VIDEODRIVER"] = "dummy"

		self.action_space = spaces.Discrete(10) # Weight the flap such that 1/10 action is to flap.
		# self.observation_space = spaces.Box(low = np.array([-np.inf, -np.inf, -np.inf]), high = np.array([np.inf, np.inf, np.inf]), dtype=np.uint8)
		#self.observation_space = spaces.Box(low = np.array([0, 0, 0]), high = np.array([SCREENHEIGHT, SCREENWIDTH, SCREENHEIGHT]), dtype=np.uint8)

		# self.observation_space = gym.spaces.Box(-np.inf, np.inf,
		# 								shape=(2,),
		# 								dtype=np.float32)
		self.observation_space = spaces.Box(low = 0, high = 255, shape = (SCREENWIDTH, SCREENHEIGHT, 3), dtype=np.uint8)

		pygame.init()
		self.FPSCLOCK = pygame.time.Clock()
		self.SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))

		pygame.display.set_caption('Flappy Bird')

		# numbers sprites for score display
		# image, sound and hitmask  dicts
		IMAGES['numbers'] = (
			pygame.image.load('assets/sprites/0.png').convert_alpha(),
			pygame.image.load('assets/sprites/1.png').convert_alpha(),
			pygame.image.load('assets/sprites/2.png').convert_alpha(),
			pygame.image.load('assets/sprites/3.png').convert_alpha(),
			pygame.image.load('assets/sprites/4.png').convert_alpha(),
			pygame.image.load('assets/sprites/5.png').convert_alpha(),
			pygame.image.load('assets/sprites/6.png').convert_alpha(),
			pygame.image.load('assets/sprites/7.png').convert_alpha(),
			pygame.image.load('assets/sprites/8.png').convert_alpha(),
			pygame.image.load('assets/sprites/9.png').convert_alpha()
		)

		IMAGES['player'] = (
			pygame.image.load(PLAYERS_FILES[0]).convert_alpha(),
			pygame.image.load(PLAYERS_FILES[1]).convert_alpha(),
			pygame.image.load(PLAYERS_FILES[2]).convert_alpha(),
		)
		IMAGES['pipe'] = (
			pygame.transform.flip(
				pygame.image.load(PIPES_LIST).convert_alpha(), False, True),
			pygame.image.load(PIPES_LIST).convert_alpha(),
		)

		# game over sprite
		IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
		# message sprite for welcome screen
		IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
		# base (ground) sprite
		IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

		IMAGES['background'] = pygame.image.load(BACKGROUND_FILE).convert()

		# Sounds
		if 'win' in sys.platform:
			soundExt = '.wav'
		else:
			soundExt = '.ogg'
		SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
		SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
		SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
		SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
		SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

		# Hitmasks for pipes
		HITMASKS['pipe'] = (
			self.getHitmask(IMAGES['pipe'][0]),
			self.getHitmask(IMAGES['pipe'][1]),
		)

		# hitmask for player
		HITMASKS['player'] = (
			self.getHitmask(IMAGES['player'][0]),
			self.getHitmask(IMAGES['player'][1]),
			self.getHitmask(IMAGES['player'][2]),
		)

		self.SCREEN.blit(IMAGES['background'], (0,0))
		pygame.display.update()

		# Game Settings
		self.playerIndexGen = cycle([0, 1, 2, 1])
		self.basex = 0
		self.playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2) + 0
		self.playerx = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2) + 0
		self.playerIndex = 0
		self.score = 0
		self.loopIter = 0
		self.pipeVelX = -4
		self.baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()
		self.playerHeight = IMAGES['player'][self.playerIndex].get_height()

		# player velocity, max velocity, downward accleration, accleration on flap
		self.playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
		self.playerMaxVelY =  10   # max vel along Y, max descend speed
		self.playerMinVelY =  -8   # min vel along Y, max ascend speed
		self.playerAccY    =   1   # players downward accleration
		self.playerRot     =  45   # player's rotation
		self.playerVelRot  =   3   # angular speed
		self.playerRotThr  =  20   # rotation threshold
		self.playerFlapAcc =  -9   # players speed on flapping
		self.playerFlapped = False # True when player flaps
		self.running = True

		self.upperPipes = []
		self.lowerPipes = []

	def step(self, action):
		basex = self.basex
		reward = 0.0
		obs = list()

		if action == 1:
			if self.playery > -2 * IMAGES['player'][0].get_height():
				self.playerVelY = self.playerFlapAcc
				self.playerFlapped = True
				SOUNDS['wing'].play()

		# check for crash here
		crashTest = self.checkCrash({'x': self.playerx, 'y': self.playery, 'index': self.playerIndex},
							   self.upperPipes, self.lowerPipes)
		if crashTest[0]:
			self.running = False
			reward -= 100
			# return {
			# 	'y': self.playery,
			# 	'groundCrash': crashTest[1],
			# 	'basex': self.basex,
			# 	'upperPipes': self.upperPipes,
			# 	'lowerPipes': self.lowerPipes,
			# 	'score': self.score,
			# 	'playerVelY': self.playerVelY,
			# 	'playerRot': self.playerRot,
			# 	'done': True
			# }
		else:
			reward += 0.1 # little bit of reward for surviving

		# check for score
		playerMidPos = self.playerx + IMAGES['player'][0].get_width() / 2
		for pipe in self.upperPipes:
			pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
			if pipeMidPos <= playerMidPos < pipeMidPos + 4:
				self.score += 1
				reward += 1
				SOUNDS['point'].play()

		# playerIndex basex change
		if (self.loopIter + 1) % 3 == 0:
			self.playerIndex = next(self.playerIndexGen)
		self.loopIter = (self.loopIter + 1) % 30
		basex = -((-basex + 100) % self.baseShift)

		# rotate the player
		if self.playerRot > -90:
			self.playerRot -= self.playerVelRot

		# player's movement
		if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
			self.playerVelY += self.playerAccY
		if self.playerFlapped:
			self.playerFlapped = False
			# more rotation to cover the threshold (calculated in visible rotation)
			self.playerRot = 45

		self.playerHeight = IMAGES['player'][self.playerIndex].get_height()
		self.playery += min(self.playerVelY, BASEY - self.playery - self.playerHeight)

		# move pipes to left
		for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
			uPipe['x'] += self.pipeVelX
			lPipe['x'] += self.pipeVelX

		# add new pipe when first pipe is about to touch left of screen
		if len(self.upperPipes) > 0 and 0 < self.upperPipes[0]['x'] < 5:
			newPipe = self.getRandomPipe()
			self.upperPipes.append(newPipe[0])
			self.lowerPipes.append(newPipe[1])

		# remove first pipe if its out of the screen
		if len(self.upperPipes) > 0 and self.upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
			self.upperPipes.pop(0)
			self.lowerPipes.pop(0)

		# draw sprites
		self.SCREEN.blit(IMAGES['background'], (0,0))

		for i, (uPipe, lPipe) in enumerate(zip(self.upperPipes, self.lowerPipes)):
			if i == 0:
				obs.insert(1, uPipe['x'])
				obs.insert(2, uPipe['y'])
				obs.insert(3, lPipe['x'])
				obs.insert(4, lPipe['y'])
			self.SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
			self.SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

		self.SCREEN.blit(IMAGES['base'], (basex, BASEY))
		# print score so player overlaps the score
		self.showScore(self.score)

		# Player rotation has a threshold
		visibleRot = self.playerRotThr
		if self.playerRot <= self.playerRotThr:
			visibleRot = self.playerRot
		
		playerSurface = pygame.transform.rotate(IMAGES['player'][self.playerIndex], visibleRot)
		self.SCREEN.blit(playerSurface, (self.playerx, self.playery))

		## Observations; as a human we can see where the pipes are, we can see where the bird is. Return this.
		## [height of bird, (upper pipe coords), (lower pipe coords)]
		## new obs, [height of bird, next upipex, next upipey, next lpipex, next lpipey]
		obs.insert(0, self.playery)
		# pygame.display.update()
		# self.FPSCLOCK.tick(FPS)
		return self.get_observation(), reward, not self.running, {} #obs, reward, done, info

	# def get_observation(self):
	# 	# [current y value, xcoord of mid pipe, y coord of mid pipe]
	# 	for i, (uPipe, lPipe) in enumerate(zip(self.upperPipes, self.lowerPipes)):
	# 		# print(f"{[self.playery, lPipe['y']-self.playery, uPipe['x']]}")
	# 		return[self.playery, lPipe['y']-self.playery, uPipe['x']]
	# 		# return[self.playery, self.playerx, uPipe['x'], uPipe['y'], lPipe['x'], lPipe['y'], lPipe['y']-self.playery]

	def get_observation(self):
		# r_arr = np.empty([SCREENHEIGHT, SCREENHEIGHT])
		# g_arr = np.empty([SCREENWIDTH, SCREENHEIGHT])
		# b_arr = np.empty([SCREENWIDTH, SCREENHEIGHT])
		# for i in range(SCREENWIDTH):
		# 	for j in range(SCREENHEIGHT):
		# 		colour_tuple = self.SCREEN.get_at((i, j))
		# 		r = colour_tuple[0]
		# 		g = colour_tuple[1]
		# 		b = colour_tuple[2]
		# 		r_arr[i][j] = r
		# 		g_arr[i][j] = g
		# 		b_arr[i][j] = b
		# return [r_arr, g_arr, b_arr]
		# 
		# screen = np.empty(shape=(SCREENWIDTH, SCREENHEIGHT, 3))
		# for i in range(SCREENWIDTH):
		# 	for j in range(SCREENHEIGHT):
		# 		colourtuple = self.SCREEN.get_at((i, j))
		# 		screen[i][j][0] = colourtuple[0]
		# 		screen[i][j][1] = colourtuple[1]
		# 		screen[i][j][2] = colourtuple[2]
		# return screen
		surf = pygame.surfarray.array3d(pygame.display.get_surface())
		return cv2.resize(surf, (80, 80)) # resize to 80x80


	def reset(self):
		self.playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2) + 0
		self.playerx = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2) - 200
		self.basex = 0
		self.playerIndex = 0
		self.playerIndexGen = cycle([0, 1, 2, 1])
		self.score = 9
		self.running = True
		obs = [0, 0, 0, 0, 0]

		baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

		# get 2 new pipes to add to upperPipes lowerPipes list
		newPipe1 = self.getRandomPipe()
		newPipe2 = self.getRandomPipe()

		# list of upper pipes
		self.upperPipes = [
			{'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
			{'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
		]

		# list of lowerpipe
		self.lowerPipes = [
			{'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
			{'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
		]
		return self.get_observation()

	def render(self, mode='human'):
		pygame.display.update()
		self.FPSCLOCK.tick(FPS)

	# helper functions

	def getRandomPipe(self):
		"""returns a randomly generated pipe"""
		# y of gap between upper and lower pipe
		gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
		gapY += int(BASEY * 0.2)
		pipeHeight = IMAGES['pipe'][0].get_height()
		pipeX = SCREENWIDTH + 10

		return [
			{'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
			{'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
		]

	def showScore(self, score):
		"""displays score in center of screen"""
		scoreDigits = [int(x) for x in list(str(score))]
		totalWidth = 0 # total width of all numbers to be printed

		for digit in scoreDigits:
			totalWidth += IMAGES['numbers'][digit].get_width()

		Xoffset = (SCREENWIDTH - totalWidth) / 2

		for digit in scoreDigits:
			self.SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
			Xoffset += IMAGES['numbers'][digit].get_width()

	def checkCrash(self, player, upperPipes, lowerPipes):
		"""returns True if player collders with base or pipes."""
		pi = player['index']
		player['w'] = IMAGES['player'][0].get_width()
		player['h'] = IMAGES['player'][0].get_height()

		# if player crashes into ground
		if player['y'] + player['h'] >= BASEY - 1:
			return [True, True]
		else:

			playerRect = pygame.Rect(player['x'], player['y'],
						  player['w'], player['h'])
			pipeW = IMAGES['pipe'][0].get_width()
			pipeH = IMAGES['pipe'][0].get_height()

			for uPipe, lPipe in zip(upperPipes, lowerPipes):
				# upper and lower pipe rects
				uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
				lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

				# player and upper/lower pipe hitmasks
				pHitMask = HITMASKS['player'][pi]
				uHitmask = HITMASKS['pipe'][0]
				lHitmask = HITMASKS['pipe'][1]

				# if bird collided with upipe or lpipe
				uCollide = self.pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
				lCollide = self.pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

				if uCollide or lCollide:
					return [True, False]

		return [False, False]

	def pixelCollision(self, rect1, rect2, hitmask1, hitmask2):
		"""Checks if two objects collide and not just their rects"""
		rect = rect1.clip(rect2)

		if rect.width == 0 or rect.height == 0:
			return False

		x1, y1 = rect.x - rect1.x, rect.y - rect1.y
		x2, y2 = rect.x - rect2.x, rect.y - rect2.y

		for x in xrange(rect.width):
			for y in xrange(rect.height):
				if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
					return True
		return False

	def getHitmask(self, image):
		"""returns a hitmask using an image's alpha."""
		mask = []
		for x in xrange(image.get_width()):
			mask.append([])
			for y in xrange(image.get_height()):
				mask[x].append(bool(image.get_at((x,y))[3]))
		return mask

	def get_actions(self):
		for event in pygame.event.get():
			if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
				pygame.quit()
				sys.exit()
			if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
				return 1