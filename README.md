# Previsão e Interpretação do Desgaste de Ferramentas com IA Explicável

Repositório técnico associado à pesquisa de mestrado em Engenharia de Produção na UNIFEI sobre previsão do desgaste de ferramenta no torneamento do aço AISI 52100 endurecido, utilizando sinais de vibração, aprendizado de máquina e técnicas de IA explicável.

O objetivo do projeto é organizar um pipeline reprodutível para treinamento, validação e interpretação de modelos de regressão aplicados à estimativa do desgaste da ferramenta.

## Contexto

A pesquisa investigou a relação entre sinais de vibração coletados durante o processo de torneamento e o desgaste progressivo da ferramenta de corte. O estudo combinou processamento de dados experimentais, modelos de aprendizado de máquina e interpretação por SHAP para apoiar a análise técnica do processo de usinagem.

Pontos principais:

- 78 conjuntos experimentais analisados.
- Aplicação em torneamento do aço AISI 52100 endurecido.
- Uso de variáveis extraídas de sinais de vibração.
- Modelos de regressão para previsão de desgaste.
- IA explicável com SHAP para interpretação das variáveis mais relevantes.
- Melhor desempenho experimental reportado com R² aproximado de 0,88.

## Tecnologias

- Python
- Pandas
- NumPy
- Scikit-Learn
- TensorFlow
- XGBoost
- LightGBM
- SHAP
- Matplotlib

## Estrutura

```text
tool-wear-prediction/
|-- data/
|   `-- README.md
|-- src/
|   |-- main.py
|   |-- model_utils.py
|   `-- preprocessing.py
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Dados

Os dados experimentais brutos não estão publicados neste repositório. Eles fazem parte da base utilizada na pesquisa acadêmica e devem ser tratados conforme as restrições de uso do estudo.

Para executar o pipeline com uma base própria, utilize um arquivo CSV contendo:

- uma coluna alvo com o desgaste medido da ferramenta;
- colunas numéricas com variáveis extraídas dos sinais de vibração ou condições experimentais;
- uma linha por experimento, ensaio ou janela consolidada de análise.

O nome padrão da coluna alvo é `Desgaste`, mas ele pode ser alterado pela linha de comando.

## Como executar

Crie um ambiente virtual e instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o pipeline informando o CSV experimental:

```bash
python src/main.py --data data/raw/dados_experimentais.csv --target Desgaste
```

Também é possível informar explicitamente as variáveis de entrada:

```bash
python src/main.py ^
  --data data/raw/dados_experimentais.csv ^
  --target Desgaste ^
  --features "rms,kurtosis,crest_factor,energia"
```

Para gerar uma análise SHAP do melhor modelo compatível:

```bash
python src/main.py --data data/raw/dados_experimentais.csv --target Desgaste --shap
```

## Saídas geradas

Por padrão, os resultados são gravados em `reports/experiment/`:

- `metrics.csv`: métricas por modelo.
- `predictions.csv`: valores reais e previstos em validação cruzada.
- `predicted_vs_real.png`: comparação visual do melhor modelo.
- `shap_summary.png`: resumo SHAP, quando habilitado e compatível.

## Métricas utilizadas

O pipeline calcula:

- R²;
- MAE;
- RMSE.

Essas métricas permitem avaliar erro absoluto, erro quadrático e capacidade explicativa do modelo.

## Observações técnicas

O código foi estruturado para evitar caminhos fixos e edição manual de arquivos fonte. A seleção de variáveis pode ser feita automaticamente a partir de colunas numéricas ou informada pela linha de comando.

Modelos opcionais como XGBoost e LightGBM são carregados apenas quando as dependências estão disponíveis no ambiente. O histórico experimental do projeto também contempla redes neurais com TensorFlow/Keras.

## Autor

Lucas Ribeiro Alves Costa  
Cientista de Dados | Aprendizado de Máquina | Previsão e Otimização  
LinkedIn: https://www.linkedin.com/in/lucas-ribeiro-datascientist/  
Portfólio: https://lucasribeirodatascience.github.io/portfolio-lucas-ribeiro/

## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.
