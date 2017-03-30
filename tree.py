import random

class Chromosome:
    def __init__(self, gene, fitness):
        self.gene = gene
        self.fitness = fitness

    def mate(self, mate):
        pivot = random.randint(0, len(self.gene) - 1)
        gene1 = self.gene[:pivot] + mate.gene[pivot:]
        gene2 = mate.gene[:pivot] + self.gene[pivot:]
        
        return Chromosome(gene1, 0), Chromosome(gene2, 0)

    def mutate(self, pool):
        
        gene = list(self.gene)
        while True:
            picked = random.choice(pool)
            if picked not in gene:
                break 

        idx = random.randint(0, len(gene) - 1)
        gene[idx] = picked

        return Chromosome(gene, 0)

def generate_random_gene(depth, elements):
    # 요소에서 하나를 랜덤하게 뽑아서 시작 노드를 만들고
    # 겹치지 않게 다음 요소를 뽑아서 뎁쓰만큼 만든다 
    gene = []
    while len(gene) < depth:
        while True:
            e = random.choice(elements)
            if e not in gene:
                gene.append(e)
                break
    
    return gene


    
class Population:
    _tournamentSize = 3

    def __init__(self, elements, size=1024, crossover=0.8, elitism=0.1, mutation=0.03):
        self.elitism = elitism
        self.mutation = mutation
        self.crossover = crossover

        buf = []
        for i in range(size):
            buf.append(generate_random_gene(7, elements))

        tree = ReservationTree(Node(0))
        for c in buf:
            tree.add(c)
        
        chromsome_list = tree.evaluate()
        self.population = list(sorted(chromsome_list, key=lambda x: x.fitness))

        # 출력하기
        node = tree.root
        print(node.item)
        while len(node.children) > 0:
            v = []
            for c in node.children:
                v.append(c.item)
            print(v)
            node = random.choice(node.children)

    def _tournament_selection(self):
        best = random.choice(self.population)
        for i in range(Population._tournamentSize):
            cont = random.choice(self.population)
            if (cont.fitness < best.fitness): best = cont
        return best

    def _selectParents(self):
        return (self._tournament_selection(), self._tournament_selection())

    def evolve(self, pool):

        size = len(self.population)
        idx = int(round(size * self.elitism))
        buf = self.population[:idx]

        while (idx < size):
            if random.random() <= self.crossover:
                (p1, p2) = self._selectParents()
                children = p1.mate(p2)
                for c in children:
                    if random.random() <= self.mutation:
                        buf.append(c.mutate(pool))
                    else:
                        buf.append(c)
                idx += 2
            else:
                if random.random() <= self.mutation:
                    buf.append(self.population[idx].mutate(pool))
                else:
                    buf.append(self.population[idx])
                idx += 1
        
        tree = ReservationTree(Node(0))
        for c in buf:
            tree.add(c.gene)
        chromsome_list = tree.evaluate()
        print('selected value %s' % tree.root.selected.gene)

        self.population = list(sorted(chromsome_list[:size], key=lambda x: x.fitness))


class GenePolicy:

    @staticmethod
    def evaluate(gene):
        sum = 0
        for i in range(len(gene)):
           sum += abs(gene[i] - (i + 1))
        return sum

class Node:
    def __init__(self, item):
        self.parent = None
        self.children = []
        self.item = item
        self.gene = None
        self.isleaf = False

    def find_child(self, item):
        for child in self.children :
            if child.item == item:
                return child
        return None

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def remove_child(self, child):
        new_children = []
        for c in self.children:
            if c != child:
                new_children.append(c)
        self.children = new_children

class ReservationTree:
    def __init__(self, root):
        self.root = root
        self.leaves = []
    
    def add(self, gene):
        node = self.root
        for g in gene:
            child = node.find_child(g)
            if child is None:
                child = Node(g)
                node.add_child(child)
            node = child
        leaf = node

        # evaluate leaf
        leaf.isleaf = True
        leaf.gene = gene
        leaf.selected = leaf
        self.leaves.append(leaf)
        

    def remove(self, chromosome):
        node = self.root
        for g in chromosome.gene:
            child = node.find_child(g)
            node = child

        # leaf 에서부터 children 이 없는 노드를 제거한다 
        while node.parent:
            parent = node.parent
            if len(node.children) == 0:
                parent.remove_child(node)
            node = parent
    
    def evaluate(self):
        buf = []
        ReservationTree.simple(self.root, True)
        for leaf in self.leaves:
            chromosome = Chromosome(leaf.gene, ReservationTree.calculate_fitness(leaf))
            buf.append(chromosome)
        return buf
        
    @staticmethod
    def calculate_fitness(leaf):
        H = 0
        K = 0

        node = leaf
        while node.parent:
            parent = node.parent
            if parent.selected == leaf.selected:
                K += 1
            H += 1
            node = parent

        return H - K + 1


    @staticmethod
    def minmax(node, is_max):
        if node.isleaf:
            return GenePolicy.evaluate(node.gene)
        
        if is_max == True:
            best = -float('Inf')
            best_node = None

            for child in node.children:
                v = ReservationTree.minmax(child, False)
                if v > best:
                    best = v
                    best_node = child
            
            node.selected = best_node.selected
            return best
        else:
            best = float('Inf')
            best_node = None

            for child in node.children:
                v = ReservationTree.minmax(child, True)
                if v < best:
                    best = v
                    best_node = child
            
            node.selected = best_node.selected
            return best

    @staticmethod
    def simple(node, is_max):
        if node.isleaf:
            return GenePolicy.evaluate(node.gene)
        
        best = float('Inf')
        best_node = None

        for child in node.children:
            v = ReservationTree.simple(child, False)
            if v < best:
                best = v
                best_node = child
        
        node.selected = best_node.selected
        return best
        
if __name__ == '__main__':
    maxGenerations = 1000
    pool = [i for i in range(1000)]
    
    pop = Population(elements=pool, size=1024, crossover=0.8, elitism=0.1, mutation=0.3)
    for i in range(1, maxGenerations + 1):
        print("Generation %d: %s" % (i, pop.population[0].gene))
        if GenePolicy.evaluate(pop.population[0].gene) == 0:
            break
        pop.evolve(pool)
