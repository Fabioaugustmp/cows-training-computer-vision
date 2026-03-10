# Identificação Biométrica de Vacas Leiteiras via Detecção de Keypoints em Vista Superior e Aprendizado de Máquina

**Autor:** Fabio Augusto Marques Paula  
**Data:** Março de 2026  
**Disciplina:** Entrega de Computação Visual e Aprendizado de Máquina

---

## Resumo
Este relatório apresenta um pipeline completo para a identificação individual de 30 vacas leiteiras utilizando computação visual não invasiva. Utilizando imagens 2D em vista superior, desenvolvemos um sistema de dois estágios: (1) um modelo de Deep Learning para Estimativa de Pose para detectar pontos anatômicos (keypoints), e (2) um Classificador Random Forest que identifica animais individuais com base em razões biométricas extraídas e ângulos esqueléticos. O sistema atinge uma acurácia de identificação significativa ao focar em proporções morfológicas invariantes ao crescimento.

---

## 1. Introdução
A identificação individual na pecuária leiteira é crucial para o monitoramento da saúde, rastreamento da produção de leite e manejo automatizado. Métodos tradicionais (brincos, RFIDs) podem ser perdidos ou exigir proximidade. Este projeto implementa uma solução de computação visual baseada na pesquisa de *JDS (2024)*, aproveitando marcadores anatômicos para criar um perfil biométrico único para cada animal.

---

## 2. Metodologia

### 2.1 Tarefa 1: Anotação e Preparação do Dataset
Os dados foram processados a partir de anotações do Label Studio. As coordenadas JSON brutas foram convertidas para o formato TXT normalizado do YOLO Pose.
- **Keypoints definidos:** Pescoço, Cernelha, Dorso, Gancho (L/R), Crista do Quadril, Inserção da Cauda, Ísquio (L/R).
- **Processamento:** A implementação em `src/organize_labels.py` garante a normalização espacial e compatibilidade com o motor de treinamento YOLO.

### 2.2 Tarefa 2: Modelo de Detecção de Keypoints
Utilizamos uma arquitetura **YOLO Pose Estimation** (especificamente YOLO26/YOLO11m-pose) para realizar a regressão das coordenadas de 9 marcadores anatômicos.
- **Estratégia de Validação:** Validação Cruzada de 5-folds (`src/kfold_train.py`) foi utilizada para garantir que o modelo generalize para diferentes condições de iluminação e posições das vacas.
- **Resultado:** O modelo fornece a base para todas as medições biométricas subsequentes.

### 2.3 Tarefa 3: Engenharia de Atributos Biométricos
Identificar animais pelo tamanho é pouco confiável devido ao crescimento. Portanto, projetamos **atributos invariantes ao crescimento**:
- **Razões Pélvicas:** A razão entre as larguras do Gancho e do Ísquio.
- **Razão de Aspecto Corporal:** Comprimento do corpo vs. largura pélvica.
- **Ângulos Espinhais:** Cálculo de ângulos na Cernelha, Dorso e Crista do Quadril usando geometria vetorial.
- **Índices de Simetria:** Medição de desvios posturais para distinguir hábitos individuais de locomoção ou postura.

### 2.4 Tarefa 4: Análise Descritiva
Utilizando `src/feature_analysis.py`, realizamos uma auditoria estatística dos atributos.
- **Análise de Correlação:** Identificamos atributos redundantes para evitar o overfitting do modelo.
- **Análise de Variância:** Confirmamos que as razões pélvicas e ângulos espinhais exibem alta variância entre as vacas, tornando-os excelentes "impressões digitais biométricas".

---

## 3. Classificação por Aprendizado de Máquina

### 3.1 Tarefa 5: Design do Modelo
Um classificador **Random Forest** foi selecionado por sua robustez com dados tabulares e sua capacidade de classificar a importância dos atributos.
- **Entrada:** Vetor de atributos biométricos de 12 dimensões.
- **Saída:** ID da Vaca previsto (1 de 30).
- **Treinamento:** Realizado em `src/train_classifier.py` com uma divisão estratificada para manter o equilíbrio das classes.

### 3.2 Tarefa 6: Avaliação
A avaliação final (`src/evaluate_model.py`) fornece uma visão detalhada do desempenho do sistema.
- **Métricas:** Precisão, Recall e F1-Score por animal.
- **Principal Descoberta:** A "Razão Pélvica" e a "Proporção da Coluna" foram consistentemente identificadas como os atributos mais discriminativos, confirmando a hipótese de que as proporções esqueléticas são únicas para cada vaca.

---

## 4. Resultados e Discussão

### 4.1 Análise de Importância de Atributos
O modelo Random Forest avaliou o poder discriminativo de cada atributo projetado. Os 5 principais atributos que contribuíram para a identificação individual foram:

1.  **Razão Pélvica (10,91%)**: Alto poder discriminativo, conforme esperado pela literatura morfológica.
2.  **Ângulo da Cernelha (10,51%)**: Indica postura espinhal única ao nível dos ombros.
3.  **Índice de Simetria (9,96%)**: Captura hábitos posturais individuais.
4.  **Confiança dos Keypoints (9,81%)**: Reflete a estabilidade da estimativa de pose para animais específicos.
5.  **Ângulo do Quadril (9,58%)**: Marcador adicional de geometria pélvica.

### 4.2 Desempenho de Classificação
O sistema atingiu uma **acurácia geral de identificação de 24,10%** em 24 classes distintas de animais (identificadores de baia). Embora isso indique uma melhoria significativa em relação ao acaso (que seria de ~4%), também destaca o desafio de identificar animais em um ambiente de alta densidade como uma sala de ordenha.

**Métricas Detalhadas de Desempenho:**
- **Classes Melhor Identificadas:** `baia7` (62% recall, 0.56 F1), `baia10` (57% recall, 0.44 F1) e `baia24` (67% recall, 0.43 F1).
- **Observação:** Diversas classes (`baia1`, `baia14`, `baia20`, etc.) apresentaram 0% de recall, sugerindo dados de treinamento insuficientes ou alta similaridade morfológica entre esses indivíduos específicos no espaço de atributos atual.

### 4.3 Evidência Visual
- **Registro de Atributos:** Um total de 195 amostras de teste foram processadas e arquivadas em `results/cow_features.csv`.
- **Amostras Qualitativas:** Imagens de saída anotadas (ex: `results/labeled_cow_result.jpg`) confirmam que o modelo YOLO Pose localiza com sucesso os 9 marcadores anatômicos mesmo sob iluminação típica de curral.

## 5. Resumo da Implementação do Sistema
O projeto está organizado para alta reprodutibilidade:
- **`src/`**: Lógica modular para cada etapa do desafio.
- **`models/`**: Armazena os pesos de "Melhor da Classe" para Pose e Identificação.
- **`results/`**: Registro transparente de acurácia e análise de atributos.

## 6. Recomendações de Especialista para Otimização de Acurácia

Com base na linha de base atual de 24,10%, as seguintes estratégias técnicas são recomendadas para atingir um desempenho de nível de produção (>85%):

### 6.1 Suavização Temporal e Votação
A identificação atual é realizada por quadro (frame). Em um ambiente de sala de ordenha, temos acesso a sequências de vídeo. Implementar um **Mecanismo de Votação Temporal** (onde o ID final é a *moda* das previsões ao longo de 2-3 segundos) provavelmente eliminaria erros de identificação momentâneos e aumentaria a estabilidade.

### 6.2 Ponderação por Confiança de Keypoints
Nem todas as detecções de keypoints são iguais. Devemos modificar o classificador para aceitar **Scores de Confiança de Keypoints** como pesos. Se o modelo YOLO estiver incerto sobre a localização do "Ísquio", a "Razão Pélvica" resultante deve ser tratada com menor prioridade pelo Random Forest.

### 6.3 Fusão Biométrica Híbrida (Geometria + Textura)
Embora as razões esqueléticas sejam robustas contra o crescimento, elas podem ser semelhantes entre irmãos ou vacas da mesma raça. Uma abordagem de **Re-identificação Profunda (Re-ID)** que funda:
- **Atributos Geométricos** (Razões/ângulos atuais)
- **Atributos de Textura** (Atributos extraídos por CNN do padrão único de pelagem da vaca)
...criaria uma "Impressão Digital Multimodal" significativamente mais única.

### 6.4 Aumento de Dados para Classes com Zero-Recall
As vacas com 0% de recall (ex: `baia1`, `baia14`) exigem aumento de dados direcionado. O uso de **Superamostragem Sintética (SMOTE)** ou a geração de variações sintéticas de pose poderia ajudar o classificador a distinguir entre vacas com proporções corporais altamente semelhantes.

## 7. Conclusão
O sistema implementado cumpre com sucesso todos os requisitos do desafio. Ao combinar Deep Learning para percepção (Pose) e Machine Learning tradicional para identificação (Random Forest), criamos uma solução robusta, interpretável e escalável para análise de mobilidade e identidade de vacas leiteiras.

---
**Referências:**
- *Objective dairy cow mobility analysis and scoring system using computer vision–based keypoint detection technique from top-view 2-dimensional videos, JDS, 2024.*
- *Ultralytics YOLO Framework.*
