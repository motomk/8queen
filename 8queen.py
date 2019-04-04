# coding: utf-8
import pprint
import random
from deap import base
from deap import creator
from deap import tools

creator.create("EightQueen", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.EightQueen)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 7)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual,
    toolbox.attr_bool, 8)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

gVec = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
def getCell(board, pos, ofst):
    posx = pos[0] + ofst[0]
    posy = pos[1] + ofst[1]
    if posx >= 0 and posy >= 0 and posx < 8 and posy < 8:
        val = board[posx][posy]
    else:
        val = -1

    return val


def calcFitness(gene):
    # print("gene:{}".format(gene))
    fitness = 0
    board = []

    for i in range(0, 8):
        line = []
        for j in range(0, 8):
            if j == gene[i]:
                line.append(1)
            else:
                line.append(0)
        board.append(line)

    for i in range(0, 8):
        for j in range(0, 8):
            val = getCell(board, (i,j), (0,0))
            if val == 1:
                for vec in gVec:
                    for k in range(1, 8):
                        valofst = getCell(board, (i,j), (vec[0]*k, vec[1]*k))
                        if valofst == 1:
                            fitness += 1
                        elif valofst == -1:
                            break

    # print("fitness:{}".format(fitness))
    # print(board)
    return fitness,


toolbox.register("evaluate", calcFitness)
toolbox.register("mate", tools.cxTwoPoints)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(64)
    # 初期の個体群を生成
    pop = toolbox.population(n=100)
    # print(pop)
    # 64Ｃ8 = 4426165368通り
    CXPB, MUTPB, NGEN = 0.5, 0.2, 50 # 交差確率、突然変異確率、進化計算のループ回数

    print("Start of evolution")

    # 初期の個体群の評価
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # 進化計算開始
    for g in range(NGEN):
        print("-- Generation %i --" % g)

        # 次世代の個体群を選択
        offspring = toolbox.select(pop, len(pop))
        # 個体群のクローンを生成
        offspring = list(map(toolbox.clone, offspring))

        # 選択した個体群に交差と突然変異を適応する
        # 偶数番目と奇数番目の個体を取り出して交差
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 適合度が計算されていない個体を集めて適合度を計算
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # 次世代群をoffspringにする
        pop[:] = offspring

        # すべての個体の適合度を配列にする
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    # print board
    board = [[0 for x in range(8)] for y in range(8)]
    for i in range(0, 8):
        for j in range(0, 8):
            if i == best_ind[j]:
                board[i][j] = 1
            else:
                board[i][j] = 0
    pprint.pprint(board)

if __name__ == '__main__':
    main()