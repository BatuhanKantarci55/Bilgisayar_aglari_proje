import time
from src.network_manager import NetworkManager
from src.algorithms.genetic import GeneticAlgorithm
from src.algorithms.aco import ACO

def main():
    print("=================================================")
    print("   BSM307 - AĞ OPTİMİZASYON PROJESİ")
    print("=================================================")
    print("NetworkX ve NumPy ile veriler işleniyor...")
    
    nm = NetworkManager()
    
    # NetworkX methodları ile istatistikleri alıyoruz
    node_count = nm.G.number_of_nodes()
    edge_count = nm.G.number_of_edges()
    
    print(f"Başarılı: {node_count} Düğüm ve {edge_count} Bağlantı yüklendi.")
    print(f"Toplam Talep Sayısı: {len(nm.demands)}")
    print("-" * 50)
    
    ga = GeneticAlgorithm(nm)
    aco = ACO(nm)
    
    weights = (0.33, 0.33, 0.34) 
    
    ga_wins = 0
    aco_wins = 0
    
    for i, demand in enumerate(nm.demands):
        src = demand['src']
        dst = demand['dst']
        bw = demand['bw_demand']
        
        print(f"\n[{i+1}/{len(nm.demands)}] Talep: {src} -> {dst} (Hız: {bw} Mbps)")
        
        # --- GA ---
        start_t = time.time()
        ga_path, ga_fit = ga.run(src, dst, weights)
        ga_time = time.time() - start_t
        
        # --- ACO ---
        start_t = time.time()
        aco_path, aco_fit = aco.run(src, dst, weights)
        aco_time = time.time() - start_t
        
        # --- SONUÇLAR ---
        ga_str = f"{ga_fit:.4f}" if ga_fit != float('inf') else "Bulunamadı"
        aco_str = f"{aco_fit:.4f}" if aco_fit != float('inf') else "Bulunamadı"
        
        print(f"   > GA  : Maliyet = {ga_str} | Süre: {ga_time:.4f}s")
        print(f"   > ACO : Maliyet = {aco_str} | Süre: {aco_time:.4f}s")
        
        if ga_fit < aco_fit:
            print("   >>> KAZANAN: GENETİK ALGORİTMA")
            ga_wins += 1
        elif aco_fit < ga_fit:
            print("   >>> KAZANAN: KARINCA KOLONİSİ")
            aco_wins += 1
        else:
            print("   >>> BERABERE")
            
    print("\n" + "="*50)
    print("GENEL PUAN DURUMU")
    print("="*50)
    print(f"Genetik Algoritma  : {ga_wins}")
    print(f"Karınca Kolonisi   : {aco_wins}")
    print("="*50)

if __name__ == "__main__":
    main()