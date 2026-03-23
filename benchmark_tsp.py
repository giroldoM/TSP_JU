import main
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Configurações do Teste
MAPA = 'tsp/berlin52.tsp'
OTIMO_CONHECIDO = 7542  # Valor oficial para Berlin52
ITERACOES = 5

def run_benchmark():
    print(f"Iniciando Benchmark: {datetime.now()}")
    matrix = main.get_matrix(MAPA)
    
    # Baseline: Algoritmo de Prim
    _, prim_dist = main.solve_tsp_prim(matrix, start=0)
    
    results = []
    
    for i in range(ITERACOES):
        start_time = time.time()
        # Usando os hiperparâmetros sugeridos no main.py
        path, dist = main.genetic_algorithm_tsp(
            matrix, 
            num_paths=400, 
            num_generations=1000, 
            mutation_rate=0.01, 
            num_best=4, 
            k_tournament=5
        )
        duration = time.time() - start_time
        
        # Cálculo de Erro Relativo (Gap)
        gap_opt = ((dist - OTIMO_CONHECIDO) / OTIMO_CONHECIDO) * 100
        improvement_over_prim = ((prim_dist - dist) / prim_dist) * 100
        
        results.append({
            'Iteração': i + 1,
            'Distância_GA': dist,
            'Tempo_sec': duration,
            'Gap_Otimo_%': gap_opt,
            'Melhora_vs_Prim_%': improvement_over_prim
        })
        print(f"Execução {i+1} concluída.")

    df = pd.DataFrame(results)
    
    print("\n" + "="*40)
    print("INDICADORES PARA O CURRÍCULO")
    print("="*40)
    print(f"Distância Média: {df['Distância_GA'].mean():.2f}")
    print(f"Melhor Distância Encontrada: {df['Distância_GA'].min():.2f}")
    print(f"Tempo Médio de Execução: {df['Tempo_sec'].mean():.2f}s")
    print(f"Gap Médio em relação ao Ótimo: {df['Gap_Otimo_%'].mean():.2f}%")
    print(f"Melhora Média sobre Baseline (Prim): {df['Melhora_vs_Prim_%'].mean():.2f}%")
    print("="*40)
    
    df.to_csv("metricas_curriculo.csv", index=False)
    print("Métricas salvas em 'metricas_curriculo.csv'")

if __name__ == "__main__":
    run_benchmark()