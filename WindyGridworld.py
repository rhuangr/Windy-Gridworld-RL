import random
import copy
import numpy as np

POSSIBLE_DIRECTIONS = ['u','d','l','r']
class GridWorld:

    def __init__(self, rows, columns, start: list[int, int], end: list[int, int]) -> None:
        self.rows = rows
        self.columns = columns
        self.start = start
        self.end = end
        
        try:
            test = np.zeros((rows,columns))
            test[start[0]][0] = 1
            test[0][start[1]] = 1
            test[end[0]][0] = 1
            test[0][end[1]] = 1
        except IndexError:
            raise IndexError("make sure start and end indices are within the bounds of the world...")
            
    def setWind(self, windInfo: dict[int: int]):
        self.windInfo = windInfo
       
    def applyWind(self, agentXPos):
        variance = 0

        # stochastic environment: uncomment to make wind strength have a variance
        
        # if agentXPos in self.windInfo:
        #     x = random.randint(1,3)
        #     if x == 1:
        #         variance = 1
        #     elif x == 2:
        #         variance = 0
        #     else: variance = -1

        upwardForce = -self.windInfo.get(agentXPos, 0) + variance
        return upwardForce

class Agent:

    def __init__(self, gridWorld : GridWorld) -> None:
        self.epsilon = 0.2
        self.stepSize = 0.5
        self.gridWorld = gridWorld
        self.currentPos = copy.deepcopy(gridWorld.start)
        self.reward = 0
        self.stateActionValues = np.zeros((self.gridWorld.rows, self.gridWorld.columns, 4))
        self.episodeCount = 1

    def move(self, direction: str):
        temp = copy.deepcopy(self.currentPos)
        
        if direction == 'u':
            self.currentPos[0] -= 1  

        elif direction == 'd':
            self.currentPos[0] += 1  
            
        elif direction == 'l':
            self.currentPos[1] -= 1  
            
        elif direction == 'r':
            self.currentPos[1] += 1  
        
        self.gridWorld.agentPos = self.currentPos

        if self.currentPos[0] < 0 or self.currentPos[0] >= self.gridWorld.rows or  self.currentPos[1] < 0 or self.currentPos[1] >= self.gridWorld.columns:
            self.currentPos = temp
        
        windforce =self.gridWorld.applyWind(temp[1])
        self.currentPos[0] += windforce
        
        if self.currentPos[0] < 0:
            self.currentPos[0] = 0
            
        elif self.currentPos[0] >= self.gridWorld.rows:
            self.currentPos[0] = self.gridWorld.rows - 1
            
    def getReward(self):
        return 0 if self.currentPos == self.gridWorld.end else -1
    
    def getEpisode(self):
        moves = 0
        reward = 0
        actionIndex = self.getNextAction()
        while True:
            moves += 1
            tempIndex = actionIndex
            prevPos = tuple(self.currentPos)
            self.move(POSSIBLE_DIRECTIONS[tempIndex])
            reward = self.getReward()
            if reward == 0:
                self.setValues(prevPos,self.currentPos, tempIndex, actionIndex, 0)
                return moves
            actionIndex = self.getNextAction()
            self.setValues(prevPos,self.currentPos, tempIndex, actionIndex, -1)
            

    def train(self, episodes):
        i = 0
        movesTaken = 1000
        movesTakenList = []
        while (i<episodes):
            movesTaken = min(movesTaken, self.getEpisode())
            self.currentPos = copy.deepcopy(self.gridWorld.start)
            if i > episodes - 11:
                movesTakenList.append(movesTaken)
            i+= 1
            self.episodeCount += 1
            if self.epsilon > 0.1:
                self.epsilon -= 2/episodes 
        print()
        print(f"Least amount of moves to complete is: {movesTaken}")
        print(f"Average moves taken of last 10 runs: {np.mean(movesTakenList)} {movesTakenList}")
        print()
        printBestActions(self.gridWorld.rows, self.gridWorld.columns, self.gridWorld.end)   
        printActionValues(self.gridWorld.rows, self.gridWorld.columns)
        print(self.epsilon)

    def setValues(self, prevPos, currentPos, actionIndex, actionIndex2, reward):

        prevActionValue = self.stateActionValues[prevPos[0]][prevPos[1]][actionIndex]
        if reward == 0:
            currentActionValue = 0
        else:
            currentActionValue = self.stateActionValues[currentPos[0]][currentPos[1]][actionIndex2]
        newValue = prevActionValue + self.stepSize*(reward + currentActionValue - prevActionValue)
        self.stateActionValues[prevPos[0]][prevPos[1]][actionIndex] = newValue


    def getNextAction(self) -> int:
        epsilonCheck = random.random()
        y,x = self.currentPos
        maxValue = np.max(self.stateActionValues[y][x][:])
        possibleActions = [i for i in range(len(self.stateActionValues[y][x][:])) if self.stateActionValues[y][x][i]==maxValue]
        if epsilonCheck > self.epsilon/self.episodeCount:
            actionIndex = possibleActions[random.randint(0, len(possibleActions) - 1)]
        else:
            actionIndex = random.randint(0,len(POSSIBLE_DIRECTIONS) - 1)
        return actionIndex
        

def printBestActions(rows,columns,end):
    for i in range(rows):
        for j in range(columns):
            direction = POSSIBLE_DIRECTIONS[np.argmax(agent.stateActionValues[i][j])]  
            if i == end[0] and j == end[1]:
               arrow = 'E'
            elif direction == 'u':
                arrow = '↑'
            elif direction == 'd':
                arrow = '↓'   
            elif direction == 'l':
                arrow = '←'        
            elif direction == 'r':
                arrow = '→'
            print(arrow, end=" ")
        print()
    print()
    
def printActionValues(rows, columns):
    for i in range(rows):
        for j in range(columns):
            direction = np.max(agent.stateActionValues[i][j])
            if direction == 0:
                print(f" 0.00", end=" ")
                
            elif direction % 1 == 0:
                print(f"{direction}", end=" ")

            else: 
                print(f"{direction:.3}", end=" ")
        print()
    print()

# start and end position indices of the gridworld
start= [3,0]

# there exists unreachable end areas if wind is enabled, careful..
end = [3,7]

world = GridWorld(7, 10 , start, end)

# key: index of the column, value: strength of the wind
windStrength = {3:1, 4:1, 5:1, 6:2, 7:2, 8:1}
world.setWind(windStrength)

agent = Agent(world)
# increase this to 10000 if stochastic wind is enabled. 
agent.train(300)




