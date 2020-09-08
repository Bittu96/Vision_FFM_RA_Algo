import pygame

width = 600
height= 400

blue = (0,0,255)


pygame.init()

gameDisplay = pygame.display.set_mode((width,height))


def drawBox(gameDisplay):
	pygame.draw.line(gameDisplay,blue,(100,100),(100,300))
	pygame.draw.line(gameDisplay,blue,(100,300),(300,300))
	pygame.draw.line(gameDisplay,blue,(300,300),(300,100))
	pygame.draw.line(gameDisplay,blue,(300,100),(100,100))



def hero(xy):
	pygame.draw.rect(gameDisplay,blue,xy)

def gameloop():
	gameRunning = True
	x=150
	y=150
	while gameRunning:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame,quit()
				quit()
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_s:
					y = y+2
				if event.key == pygame.K_w:
					y = y-2
				if event.key == pygame.K_a:
					x = x-2
				if event.key == pygame.K_d:
					x = x+2
		gameDisplay.fill((0,0,0))
		
		hero((x,y,10,10))
		print x,y

		pygame.display.update()


if __name__ == "__main__":
	gameloop()
