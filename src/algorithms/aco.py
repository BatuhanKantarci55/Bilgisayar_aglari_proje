import numpy as np

class ACO:
    def __init__(self, network_manager):
        self.nm = network_manager
        self.pheromones = {} 
        
        # NetworkX grafiğindeki kenarları geziyoruz
        for u, v in self.nm.G.edges():
            self.pheromones[(u, v)] = 1.0
            self.pheromones[(v, u)] = 1.0

    def run(self, source, target, weights):
        ant_count = 15
        iterations = 20
        alpha = 1.0
        beta = 2.0
        evaporation = 0.5
        
        best_path = None
        best_fitness = float('inf')

        for i in range(iterations):
            all_ant_paths = []
            
            for ant in range(ant_count):
                path = self.build_path(source, target, weights, alpha, beta)
                if path:
                    fitness = self.nm.calculate_fitness(path, weights)
                    all_ant_paths.append((path, fitness))
                    
                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_path = path

            # Buharlaşma
            for key in self.pheromones:
                self.pheromones[key] *= (1.0 - evaporation)
            
            # Güncelleme
            for path, fitness in all_ant_paths:
                deposit = 100.0 / (fitness + 0.0001)
                for j in range(len(path)-1):
                    u, v = path[j], path[j+1]
                    if (u, v) in self.pheromones:
                        self.pheromones[(u, v)] += deposit
                    if (v, u) in self.pheromones:
                        self.pheromones[(v, u)] += deposit

        return best_path, best_fitness

    def build_path(self, start, end, weights, alpha, beta):
        current_node = start
        path = [start]
        visited = {start}
        
        while current_node != end:
            # NetworkX ile komşuları al
            neighbors = list(self.nm.G.neighbors(current_node))
            candidates = [n for n in neighbors if n not in visited]
            
            if not candidates:
                return None
            
            probs = []
            for neighbor in candidates:
                tau = self.pheromones.get((current_node, neighbor), 1.0) ** alpha
                
                edge = self.nm.G[current_node][neighbor]
                cost = edge['link_delay'] + (1000.0 / edge['bandwidth']) 
                eta = (1.0 / (cost + 0.0001)) ** beta
                
                probs.append(tau * eta)
            
            # --- NumPy ile Profesyonel Seçim ---
            probs = np.array(probs)
            total = probs.sum()
            
            if total == 0:
                # Olasılıklar sıfırsa rastgele seç
                next_node = np.random.choice(candidates)
            else:
                # Olasılıkları normalize et (Toplamı 1 olsun)
                probs = probs / total
                # NumPy'ın ağırlıklı seçim fonksiyonu
                next_node = np.random.choice(candidates, p=probs)
            # -----------------------------------
            
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node
            
            if len(path) > 50: return None
            
        return path