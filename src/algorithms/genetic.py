import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, network_manager):
        self.nm = network_manager
        
    def run(self, source, target, weights):
        pop_size = 30
        generations = 30
        mutation_rate = 0.2
        
        # NetworkX yardımıyla başlangıç yollarını bul
        population = self.nm.find_initial_paths(source, target, limit=pop_size)
        
        if not population:
            return None, float('inf')

        # Popülasyon eksikse tamamla
        while len(population) < pop_size:
            # Rastgele birini kopyala
            population.append(list(population[0]))

        best_path = None
        best_fitness = float('inf')

        for gen in range(generations):
            scores = []
            for path in population:
                fitness = self.nm.calculate_fitness(path, weights)
                scores.append((fitness, path))
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_path = path
            
            scores.sort(key=lambda x: x[0])
            
            # En iyi %50'yi seç
            top_count = int(pop_size / 2)
            selected = [x[1] for x in scores[:top_count]]
            
            new_pop = selected[:]
            
            while len(new_pop) < pop_size:
                # Rastgele ebeveyn seçimi
                parent1 = selected[np.random.randint(0, len(selected))]
                parent2 = selected[np.random.randint(0, len(selected))]
                
                child = self.crossover(parent1, parent2)
                
                if np.random.random() < mutation_rate:
                    child = self.mutate(child)
                
                new_pop.append(child)
            
            population = new_pop
            
        return best_path, best_fitness

    def crossover(self, p1, p2):
        common = [node for node in p1 if node in p2 and node != p1[0] and node != p1[-1]]
        
        if common:
            pivot = random.choice(common)
            idx1 = p1.index(pivot)
            idx2 = p2.index(pivot)
            
            child = p1[:idx1] + p2[idx2:]
            
            if len(child) == len(set(child)):
                return child
                
        return p1

    def mutate(self, path):
        # Basitlik adına mutasyon orijinal yolu döndürüyor
        return path