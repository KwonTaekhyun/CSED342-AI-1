# ID: 20180522 NAME: Kwon Taekhyun
######################################################################################
# Problem 2a
# minimax value of the root node: 5
# pruned edges: h, m, t, 1
######################################################################################

from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
        ------------------------------------------------------------------------------
        Description of GameState and helper functions:

        A GameState specifies the full game state, including the food, capsules,
        agent configurations and score changes. In this function, the |gameState| argument
        is an object of GameState class. Following are a few of the helper methods that you
        can use to query a GameState object to gather information about the present state
        of Pac-Man, the ghosts and the maze.

        gameState.getLegalActions():
            Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

        gameState.generateSuccessor(agentIndex, action):
            Returns the successor state after the specified agent takes the action.
            Pac-Man is always agent 0.

        gameState.getPacmanState():
            Returns an AgentState object for pacmanPosition (in game.py)
            state.configuration.pos gives the current position
            state.direction gives the travel vector

        gameState.getGhostStates():
            Returns list of AgentState objects for the ghosts

        gameState.getNumAgents():
            Returns the total number of agents in the game

        gameState.getScore():
            Returns the score corresponding to the current state of the game
            It corresponds to Utility(s)


        The GameState class is defined in pacmanPosition.py and you might want to look into that for
        other helper methods, though you don't need to.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (pacmanPosition.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (oldFood) and pacmanPosition position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of pacmanPosition having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacmanPosition.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the pacmanPosition GUI.

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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # pacmanPosition is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

######################################################################################
# Problem 1a: implementing minimax


class MinimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):

        def value(gameState, currDepth, agentIdx):

            if gameState.isWin() or gameState.isLose():
                return gameState.getScore()
            elif currDepth == self.depth:
                return self.evaluationFunction(gameState)
            elif agentIdx == 0:
                return max_value(gameState=gameState, currDepth=currDepth)
            else:
                return min_value(gameState=gameState, currDepth=currDepth, agentIdx=agentIdx)

        def max_value(gameState, currDepth):

            v = float("-inf")
            legalActions = gameState.getLegalActions()
            for action in legalActions:
                successorGameState = gameState.generatePacmanSuccessor(
                    action)
                v = max([v, value(gameState=successorGameState,
                        currDepth=currDepth, agentIdx=1)])
            return v

        def min_value(gameState, currDepth, agentIdx):

            v = float("inf")
            legalActions = gameState.getLegalActions(agentIndex=agentIdx)

            if agentIdx == gameState.getNumAgents()-1:
                for action in legalActions:
                    nextState = gameState.generateSuccessor(
                        agentIndex=agentIdx, action=action)
                    v = min([v, value(gameState=nextState,
                                      currDepth=currDepth+1, agentIdx=0)])
            else:
                for action in legalActions:
                    nextState = gameState.generateSuccessor(
                        agentIndex=agentIdx, action=action)
                    v = min([v, value(gameState=nextState,
                                      currDepth=currDepth, agentIdx=agentIdx+1)])

            return v

        legalActions = gameState.getLegalActions()
        nextAction = Directions.STOP
        v = float("-inf")
        for action in legalActions:
            successorGameState = gameState.generatePacmanSuccessor(
                action)
            tempV = value(
                gameState=successorGameState, currDepth=0, agentIdx=1)
            if(tempV > v):
                v = tempV
                nextAction = action
        return nextAction

######################################################################################
# Problem 2b: implementing alpha-beta


class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):

        def value(gameState, currDepth, agentIdx, alpha, beta):

            if gameState.isWin() or gameState.isLose():
                return gameState.getScore()
            elif currDepth == self.depth:
                return self.evaluationFunction(gameState)
            elif agentIdx == 0:
                return max_value(gameState, currDepth, alpha, beta)
            else:
                return min_value(gameState, currDepth, agentIdx, alpha, beta)

        def max_value(gameState, currDepth, alpha, beta):

            v = float("-inf")
            legalActions = gameState.getLegalActions()
            for action in legalActions:
                successorGameState = gameState.generatePacmanSuccessor(
                    action)
                v = max([v, value(gameState=successorGameState,
                        currDepth=currDepth, agentIdx=1, alpha=alpha, beta=beta)])
                if v >= beta:
                    return v
                alpha = max([alpha, v])
            return v

        def min_value(gameState, currDepth, agentIdx, alpha, beta):

            v = float("inf")
            legalActions = gameState.getLegalActions(agentIdx)

            if agentIdx == gameState.getNumAgents()-1:
                for action in legalActions:
                    nextState = gameState.generateSuccessor(agentIdx, action)
                    v = min([v, value(nextState,
                                      currDepth+1, 0, alpha, beta)])
                if v <= alpha:
                    return v
                beta = min([beta, v])
            else:
                for action in legalActions:
                    nextState = gameState.generateSuccessor(agentIdx, action)
                    v = min([v, value(nextState,
                                      currDepth, agentIdx+1, alpha, beta)])
                if v <= alpha:
                    return v
                beta = min([beta, v])

            return v

        legalActions = gameState.getLegalActions()
        nextAction = Directions.STOP
        v = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        for action in legalActions:
            successorGameState = gameState.generatePacmanSuccessor(action)
            tempV = value(successorGameState, 0, 1, alpha, beta)
            if tempV > v:
                v = tempV
                nextAction = action
            if v >= beta:
                break
            alpha = max([alpha, v])
        return nextAction

######################################################################################
# Problem 3a: implementing expectimax


class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):

        def value(gameState, currDepth, agentIdx):

            if gameState.isWin() or gameState.isLose():
                return gameState.getScore()
            elif currDepth == self.depth:
                return self.evaluationFunction(gameState)
            elif agentIdx == 0:
                return max_value(gameState=gameState, currDepth=currDepth)
            else:
                return expected_value(gameState=gameState, currDepth=currDepth, agentIdx=agentIdx)

        def max_value(gameState, currDepth):

            v = float("-inf")
            legalActions = gameState.getLegalActions()
            for action in legalActions:
                successorGameState = gameState.generateSuccessor(
                    0, action)
                v = max([v, value(gameState=successorGameState,
                        currDepth=currDepth, agentIdx=1)])
            return v

        def expected_value(gameState, currDepth, agentIdx):

            v = 0.0
            count = 0.0
            legalActions = gameState.getLegalActions(agentIndex=agentIdx)

            if agentIdx == gameState.getNumAgents()-1:
                for action in legalActions:
                    nextState = gameState.generateSuccessor(
                        agentIndex=agentIdx, action=action)
                    v += (value(gameState=nextState,
                                currDepth=currDepth+1, agentIdx=0))
                    count += 1.0
                return v / count
            else:
                for action in legalActions:
                    nextState = gameState.generateSuccessor(
                        agentIndex=agentIdx, action=action)
                    v += (value(gameState=nextState,
                                currDepth=currDepth, agentIdx=agentIdx+1))
                    count += 1.0
                return v / count

        legalActions = gameState.getLegalActions()
        nextAction = Directions.STOP
        v = float("-inf")
        for action in legalActions:
            successorGameState = gameState.generatePacmanSuccessor(
                action)
            tempV = value(
                gameState=successorGameState, currDepth=0, agentIdx=1)
            if(tempV > v):
                v = tempV
                nextAction = action
        return nextAction
######################################################################################
# Problem 4a (extra credit): creating a better evaluation function


def betterEvaluationFunction(currentGameState):
    evaluation = currentGameState.getScore()

    pacmanPosition = currentGameState.getPacmanPosition()
    foodPositions = currentGameState.getFood().asList()

    foodDistances = [manhattanDistance(
        pacmanPosition, foodPosition) for foodPosition in foodPositions]

    closestFoodDistance = float("inf")
    if(len(foodDistances) != 0):
        closestFoodDistance = min(foodDistances)
        evaluation -= 1.0 * closestFoodDistance

    evaluation -= 1.0 * len(foodPositions)

    ghostStates = currentGameState.getGhostStates()

    scaringGhostDistances = list()
    scaredGhostDistances = list()
    for ghostState in ghostStates:
        distance = manhattanDistance(
            pacmanPosition, ghostState.getPosition())
        if ghostState.scaredTimer != 0:
            scaredGhostDistances.append(distance)
        else:
            scaringGhostDistances.append(distance)

    closestScaredGhostDistance = float("inf")
    if len(scaredGhostDistances) != 0:
        closestScaredGhostDistance = min(scaredGhostDistances)
        if(closestScaredGhostDistance <= 2):
            evaluation += 50.0
        else:
            evaluation -= 4.0*closestScaredGhostDistance

    closestScaringGhostDistance = float("inf")
    if len(scaringGhostDistances) != 0:
        closestScaringGhostDistance = min(scaringGhostDistances)
        if(closestScaringGhostDistance <= 2):
            evaluation -= 100.0
        else:
            evaluation -= 3.0/(closestScaringGhostDistance)

    evaluation -= 15.0 * len(currentGameState.getCapsules())

    if currentGameState.isLose():
        evaluation -= 500.0
    if currentGameState.isWin():
        evaluation += 500.0

    capsules = currentGameState.getCapsules()
    capsuleDistances = [manhattanDistance(
        pacmanPosition, capsule) for capsule in capsules]
    if(len(capsuleDistances) != 0):
        if(min(capsuleDistances) == 0):
            evaluation += 15.0
        else:
            evaluation -= 3.0*min(capsuleDistances)

    return evaluation


# Abbreviation
better = betterEvaluationFunction
