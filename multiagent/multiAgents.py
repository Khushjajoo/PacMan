# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodnear = 100
        for food in newFood.asList():
            dist = manhattanDistance(newPos, food)
            if foodnear > dist:
                foodnear = dist
        foodScore = 1 / foodnear

        ghostnear = 100
        for ghost in newGhostStates:
            dist = manhattanDistance(newPos, ghost.getPosition())
            if ghostnear > dist:
                ghostnear = dist
            if ghost.scaredTimer > 0 and dist < 2:
                return 1000000
            if ghost.scaredTimer == 0 and dist < 2:
                return -1000000
        ghostScore = 1 / ghostnear

        return successorGameState.getScore() + foodScore/ghostScore
    
def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        
        bestAction = None
        maxEval = float('-inf')

        for action in actions:
            succState = gameState.generateSuccessor(0, action)
            eval = self.minimaxHelper(succState, self.depth, 1)

            if eval > maxEval:
                maxEval = eval
                bestAction = action
            
        return bestAction
            
    def minimaxHelper(self, state, depth, agentIndex):

        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        
        actions = state.getLegalActions(agentIndex)

        if agentIndex == 0:
            maxEval = float('-inf')

            for action in actions:
                succState = state.generateSuccessor(agentIndex, action)
                
                eval = self.minimaxHelper(succState, depth, 1)
                maxEval = max(eval, maxEval)

            return maxEval

        minEval = float('inf')
        for action in actions:
                succState = state.generateSuccessor(agentIndex, action)
                
                newDepth = depth - 1 if agentIndex == state.getNumAgents() - 1 else depth
                newAgentIndex = (agentIndex + 1) % state.getNumAgents()
                
                eval = self.minimaxHelper(succState, newDepth, newAgentIndex)
                minEval = min(eval, minEval)
        
        return minEval


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        
        bestAction = None
        maxEval, alpha, beta = float('-inf'), float('-inf'), float('inf')

        for action in actions:
            succState = gameState.generateSuccessor(0, action)
            eval = self.alphabetaHelper(succState, self.depth, 1, alpha, beta)

            if eval > maxEval:
                maxEval = eval
                bestAction = action
            
            alpha = max(alpha, eval)
            
            
        return bestAction
            
    def alphabetaHelper(self, state, depth, agentIndex, alpha, beta):

        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        
        actions = state.getLegalActions(agentIndex)

        if agentIndex == 0:
            maxEval = float('-inf')
            
            for action in actions:
                succState = state.generateSuccessor(agentIndex, action)
                
                eval = self.alphabetaHelper(succState, depth, 1, alpha, beta)
                maxEval = max(eval, maxEval)
                alpha = max(alpha, eval)
                
                if beta < alpha:
                    break
        
            return maxEval
            

        minEval = float('inf')
        for action in actions:
                succState = state.generateSuccessor(agentIndex, action)
                
                newDepth = depth - 1 if agentIndex == state.getNumAgents() - 1 else depth
                newAgentIndex = (agentIndex + 1) % state.getNumAgents()
                
                eval = self.alphabetaHelper(succState, newDepth, newAgentIndex, alpha, beta)
                minEval = min(eval, minEval)
                beta = min(beta, eval)
                
                if beta < alpha:
                    break
                
        return minEval
          
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        x = 0
        agent = 0
        depth = 0
        for move in gameState.getLegalActions(agent): 
            state = gameState.generateSuccessor(agent, move) 
            score = self.expectimax(state, agent+1, depth) 
            if score > x: 
                x = score
                action = move 
        return action

    def expectimax(self, gameState, agent, depth):
        
        if agent == gameState.getNumAgents():  # if agent is last ghost
            agent = 0                          # reset agent to pacman
            depth += 1                         # increment depth
        
        if gameState.isWin() or gameState.isLose() or depth == self.depth: # if game is over or depth is reached
            return self.evaluationFunction(gameState)                      # return evaluation function
        
        
        
        if agent == 0: # if agent is pacman
            x = 0
            legalMoves = gameState.getLegalActions(agent) # get legal moves for pacman
            for move in legalMoves: 
                state = gameState.generateSuccessor(agent, move) # generate successor state
                x = max(x, self.expectimax(state, agent + 1, depth)) # get max value of all successors
            return x # return max value
        
        else:
            x = 0
            legalMoves = gameState.getLegalActions(agent) # get legal moves for ghost
            for move in legalMoves:
                state = gameState.generateSuccessor(agent, move) # generate successor state
                x = x + self.expectimax(state, agent + 1, depth) # get sum of all successors
            return x / len(legalMoves) # return average value as ghost is random and uniform

def betterEvaluationFunction(currentGameState: GameState):
        """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
        """
        "*** YOUR CODE HERE ***"
        newPos = currentGameState.getPacmanPosition()
        newFood = currentGameState.getFood()
        newGhostStates = currentGameState.getGhostStates()
        score = currentGameState.getScore()

        foodnear = 1000000
        for food in newFood.asList():
            dist = manhattanDistance(newPos, food)
            if foodnear > dist:
                foodnear = dist
        foodScore = 1 / foodnear

        ghostnear = 1000000
        for ghost in newGhostStates:
            dist = manhattanDistance(newPos, ghost.getPosition())
            if ghostnear > dist:
                ghostnear = dist
            if ghost.scaredTimer > 0 and dist < 2:
                return 1000000
            if ghost.scaredTimer == 0 and dist < 2:
                return -1000000
        ghostScore = 1 / ghostnear
        
        capsulenear = 1000000
        for capsule in currentGameState.getCapsules():
            dist = manhattanDistance(newPos, capsule)
            if capsulenear > dist:
                capsulenear = dist
        capsuleScore = 1 / capsulenear

        
        return score + foodScore + ghostScore + capsuleScore
# Abbreviation
better = betterEvaluationFunction

# Abbreviation
better = betterEvaluationFunction
