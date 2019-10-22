import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import random

matplotlib.rcParams['font.family'] = 'STSong'
inputfile = ['eil51.txt','eil76.txt','eil101.txt',
             'str70.txt','kroA100.txt','kroC100.txt',
             'kroD100.txt', 'lin105.txt','pcb442.txt','pr2392.txt']
outputfile = ['eil51.outcome.txt','eil76.outcome.txt','eil101.outcome.txt',
             'str70.outcome.txt','kroA100.outcome.txt','kroC100.outcome.txt',
             'kroD100.outcome.txt', 'lin105.outcome.txt','pcb442.outcome.txt','pr2392.outcome.txt']
inputfile2 = ['pcb442.txt']
outputfile2 = ['pcb442.outcome.txt']


# 改良次数
improve_count = 500

# 进化次数
evolution_time=[5000,10000,20000]

# 种群数
counts = [50]



# 设置强者的定义概率，即种群前30%为强者
retain_rate = 0.3

# 设置弱者的存活概率
random_select_rate = 0.5

# 变异率
mutation_rate = 0.05
class TspHandler:

    def __init__(self,infile,outfile,count):
        # 种群数
        self.count = count
        self.city_name = []
        self.city_condition = []
        self.Distance = []
        self.register = []
        self.read_file(infile)
        i = 0
        population = self.population
        while i < 20001:
            # 选择繁殖个体群
            parents = self.selection(population)
            # 交叉繁殖
            children = self.crossover3(parents)
            # 变异操作
            self.mutation(children)
            # 更新种群
            population = parents + children

            distance, result_path = self.get_result(population)
            self.register.append(distance)
            i += 1
            if (i == 20000):

                X = []
                Y = []
                for index in result_path:
                    X.append(self.city_condition[index-1][0] )
                    Y.append(self.city_condition[index-1][1])

                plt.plot(X, Y, '-o')
                plt.show()

                plt.plot(list(range(len(self.register))), self.register)
                plt.show()
                result_path = [self.origin] + result_path + [self.origin]
            # with open('./outcomes/' + outfile+'1', 'a', encoding='UTF-8') as f:
            #       print(count, i)
            #      print(distance)
            #      print(result_path)
            #      f.write('population:' + str(count) + '  ' + 'generations：' + str(i) + '\n')
            #     f.write('全程路程：' + str(distance) + '\n')
            #    f.write('路线：')
            #     f.write(str(result_path) + '\n')





    # 总距离
    def get_total_distance(self,x):
        distance = 0
        distance += self.Distance[self.origin - 1][x[0] - 1]
        d = len(x)
        for i in range(d):
            if i == d - 1:
                distance += self.Distance[self.origin - 1][x[i] - 1]
            else:
                distance += self.Distance[x[i] - 1][x[i + 1] - 1]
        return distance
    # 改良

    def improve(self,x):
        i = 0
        distance = self.get_total_distance(x)
        d = len(x)
        while i < improve_count:
            # randint [a,b]
            u = random.randint(0, d - 1)
            v = random.randint(0, d - 1)
            if u != v:
                new_x = x.copy()
                t = new_x[u]
                new_x[u] = new_x[v]
                new_x[v] = t
                new_distance = self.get_total_distance(new_x)
                if new_distance < distance:
                    distance = new_distance
                    x = new_x.copy()
            else:
                continue
            i += 1

    # 自然选择

    def selection(self,population):
        """
        选择
        先对适应度从大到小排序，选出存活的染色体
        再进行随机选择，选出适应度虽然小，但是幸存下来的个体
        """
        # 对总距离从小到大进行排序
        graded = [[self.get_total_distance(x), x] for x in population]
        graded = [x[1] for x in sorted(graded)]
        # 选出适应性强的染色体
        retain_length = int(len(graded) * retain_rate)
        parents = graded[:retain_length]
        # 选出适应性不强，但是幸存的染色体
        for chromosome in graded[retain_length:]:
            if random.random() < random_select_rate:
                parents.append(chromosome)
        return parents


    def crossover1(self,parents):

        d = len(parents)
        target_count = self.count - d
        children = []

        while len(children) < target_count:
            male_index = random.randint(0, d - 1)
            female_index = random.randint(0, d - 1)
            if male_index != female_index:

                male = parents[male_index]
                female = parents[female_index]
                d1 = len(male)
                star = random.randint(0, d1- 1)
                if female[star] != male[star]:
                    index = 1
                else:
                    index = 0
                gene1 = []
                gene2 = []
                gene1.append(male[star])
                gene2.append(female[star])
                while (index == 1):
                    for i in range(d1):
                        if len(gene2) == d1:
                            index = 0
                        if male[i] == female[star]:
                            star = i
                            gene1.append(male[star])
                            gene2.append(female[star])
                            if (gene1[0] == female[i]):
                                index = 0
                            break
                exchange_shine = []
                for i in range(len(gene1)):
                    exchange_shine.append([gene1[i], gene2[i]])

                children1 = male.copy()
                children2 = female.copy()
                d2 = len(exchange_shine)
                for i in range(d1):
                    for j in range(d2):
                        if (children1[i] == exchange_shine[j][0]):
                            children1[i] = exchange_shine[j][1]
                            break
                    for m in range(d2):
                        if (children2[i] == exchange_shine[m][1]):
                            children2[i] = exchange_shine[m][0]
                            break
                children.append(children1)
                if (len(children) < target_count):
                    children.append(children2)
        return children


    def crossover2(self,parents):
        d = len(parents)
        target_count = self.count - d

        children = []

        while len(children) < target_count:
            male_index = random.randint(0, d - 1)
            female_index = random.randint(0, d - 1)
            if male_index != female_index:
                male = parents[male_index]
                female = parents[female_index]

                left = random.randint(0, len(male) - 2)
                right = random.randint(left + 1, len(male) - 1)

                gene1 = male[left:right]
                gene2 = []
                children1 = male.copy()
                children2 = female.copy()

                index = 0
                d1 = len(female)
                for i in range(d1):
                    if female[i] in gene1:
                        gene2.append(female[i])
                        children2[i] = gene1[index]
                        index += 1
                children1[left:right] = gene2
                children.append(children1)
                children.append(children2)
        return children

    # 交叉繁殖

    def crossover3(self,parents):
        # 生成子代的个数,以此保证种群稳定
        target_count = self.count - len(parents)
        # 孩子列表
        children = []
        while len(children) < target_count:
            male_index = random.randint(0, len(parents) - 1)
            female_index = random.randint(0, len(parents) - 1)
            if male_index != female_index:
                male = parents[male_index]
                female = parents[female_index]

                left = random.randint(0, len(male) - 2)
                right = random.randint(left + 1, len(male) - 1)

                # 交叉片段
                gene1 = male[left:right]
                gene2 = female[left:right]

                child1_c = male[right:] + male[:right]
                child2_c = female[right:] + female[:right]
                child1 = child1_c.copy()
                child2 = child2_c.copy()

                for o in gene2:
                    child1_c.remove(o)

                for o in gene1:
                    child2_c.remove(o)

                child1[left:right] = gene2
                child2[left:right] = gene1

                d =len(child1)

                child1[right:] = child1_c[0:d - right]
                child1[:left] = child1_c[d - right:]

                child2[right:] = child2_c[0:d - right]
                child2[:left] = child2_c[d - right:]

                children.append(child1)
                children.append(child2)

        return children

    # 变异

    def mutation(slef,children):
        for i in range(len(children)):
            if random.random() < mutation_rate:
                child = children[i]
                d =len(child)
                u = random.randint(1, d - 4)
                v = random.randint(u + 1, d - 3)
                w = random.randint(v + 1, d - 2)
                child = children[i]
                child = child[0:u] + child[v:w] + child[u:v] + child[w:]
                children[i] = child
        return children

    # 得到最佳纯输出结果
    def get_result(self,population):
        graded = [[self.get_total_distance(x), x] for x in population]
        graded = sorted(graded)
        return graded[0][0], graded[0][1]


    def read_file(self,file_path):
        with open(file_path, 'r', encoding='UTF-8') as f:
            city_condition = []
            lines = f.readlines()
            for line in lines:
                line = line.split('\n')[0]
                line = line.split(' ')
                self.city_name.append(line[0])
                city_condition.append([float(line[1]), float(line[2])])
        self.city_condition = np.array(city_condition)
        # 展示地图
        # plt.scatter(city_condition[:,0],city_condition[:,1])
        # plt.show()
        # 距离矩阵
        city_count = len(self.city_name)
        self.Distance = np.zeros([city_count, city_count])

        for i in range(city_count):
            for j in range(city_count):
                self.Distance[i][j] = math.sqrt(
                    (city_condition[i][0] - city_condition[j][0]) ** 2 + (
                                city_condition[i][1] - city_condition[j][1]) ** 2)
        self.population = []
        # 设置起点
        self.origin = 15
        index = [i for i in range(city_count)]
        index.remove(15)
        # 使用改良圈算法初始化种群
        for i in range(self.count):
            # 随机生成个体
            x = index.copy()
            random.shuffle(x)
            self.improve(x)
            self.population.append(x)


if __name__ == '__main__':
    for infile,outfile in zip(inputfile2,outputfile2):
        for  count in counts:
            a = TspHandler(infile,outfile,count)



