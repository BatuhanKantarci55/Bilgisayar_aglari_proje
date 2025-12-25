import networkx as nx
import numpy as np
import os

class NetworkManager:
    def __init__(self):
        self.data_folder = 'data'
        
        # NetworkX kütüphanesini kullanarak boş bir graf oluşturuyoruz
        self.G = nx.Graph()
        self.demands = [] 
        
        # Fonksiyonları çalıştır
        self.load_nodes()
        self.load_edges()
        self.load_demands()

    def load_nodes(self):
        file_path = os.path.join(self.data_folder, 'BSM307_317_Guz2025_TermProject_NodeData.csv')
        
        # Dosya açımı
        with open(file_path, 'r', encoding='utf-8') as file:
            next(file) # Başlığı atla
            
            for line in file:
                parts = line.strip().split(';')
                if len(parts) < 3: continue

                node_id = int(parts[0])
                # Virgül - Nokta dönüşümü
                proc_delay = float(parts[1].replace(',', '.'))
                reliability = float(parts[2].replace(',', '.'))
                
                # NetworkX ile düğüm ekleme
                self.G.add_node(node_id, 
                                processing_delay=proc_delay, 
                                reliability=reliability)

    def load_edges(self):
        file_path = os.path.join(self.data_folder, 'BSM307_317_Guz2025_TermProject_EdgeData.csv')
        
        with open(file_path, 'r', encoding='utf-8') as file:
            next(file)
            
            for line in file:
                parts = line.strip().split(';')
                if len(parts) < 5: continue

                source_node = int(parts[0])
                destination_node = int(parts[1])
                bw = int(parts[2])
                delay = int(parts[3])
                rel = float(parts[4].replace(',', '.'))
                
                # NetworkX ile kenar (bağlantı) ekleme
                # NetworkX, eğer düğümler yoksa otomatik oluşturur ama özelliklerini eklemez.
                # O yüzden load_nodes önce çalışmalı.
                self.G.add_edge(source_node, destination_node, 
                                bandwidth=bw, 
                                link_delay=delay, 
                                reliability=rel)

    def load_demands(self):
        file_path = os.path.join(self.data_folder, 'BSM307_317_Guz2025_TermProject_DemandData.csv')
        
        with open(file_path, 'r', encoding='utf-8') as file:
            next(file)
            for line in file:
                parts = line.strip().split(';')
                if len(parts) < 3: continue

                self.demands.append({
                    'src': int(parts[0]),
                    'dst': int(parts[1]),
                    'bw_demand': int(parts[2])
                })

    def calculate_fitness(self, path, weights=(0.33, 0.33, 0.34)):
        # Yol boşsa veya geçersizse sonsuz ceza
        if not path or len(path) < 2:
            return float('inf')
            
        w_delay, w_rel, w_res = weights
        
        total_delay = 0
        total_rel_cost = 0
        total_res_cost = 0
        
        # 1. Düğüm Hesapları (NetworkX verisinden)
        for node_id in path[1:-1]:
            # self.G.nodes[id] ile verilere ulaşırız
            node_data = self.G.nodes[node_id]
            total_delay += node_data.get('processing_delay', 0)
            
            rel = node_data.get('reliability', 0.99)
            # NumPy logaritma fonksiyonu
            total_rel_cost += -np.log(rel) if rel > 0 else 100

        # 2. Bağlantı Hesapları
        for i in range(len(path) - 1):
            source_node = path[i]
            destination_node = path[i+1]
            
            # Bağlantı var mı kontrolü
            if not self.G.has_edge(source_node, destination_node):
                return float('inf')
                
            # self.G[u][v] ile kenar verisine ulaşırız
            edge_data = self.G[source_node][destination_node]
            
            total_delay += edge_data.get('link_delay', 0)
            
            link_rel = edge_data.get('reliability', 0.99)
            total_rel_cost += -np.log(link_rel) if link_rel > 0 else 100
            
            bw = edge_data.get('bandwidth', 1)
            total_res_cost += (1000.0 / bw) if bw > 0 else 100

        fitness = (w_delay * total_delay) + \
                  (w_rel * total_rel_cost) + \
                  (w_res * total_res_cost)
                  
        return fitness

    def find_initial_paths(self, start, end, limit=5):
        # NetworkX'in güçlü "shortest_simple_paths" fonksiyonunu kullanıyoruz.
        # Bu fonksiyon en kısadan başlayarak alternatif yollar üretir.
        paths = []
        try:
            # Generator olduğu için listeye çevirip limit kadarını alıyoruz
            raw_paths_generator = nx.shortest_simple_paths(self.G, start, end)
            
            for p in raw_paths_generator:
                paths.append(p)
                if len(paths) >= limit:
                    break
        except nx.NetworkXNoPath:
            return [] # Yol yoksa boş liste dön
            

        return paths
