# Dados experimentais

Os dados brutos utilizados na pesquisa de mestrado não estão publicados neste repositório.

Para usar o pipeline, coloque um arquivo CSV em `data/raw/` com a seguinte estrutura:

```text
Desgaste,rms,kurtosis,crest_factor,energia,...
0.12,0.45,3.10,2.40,18.70,...
0.18,0.51,3.35,2.58,21.10,...
```

Requisitos:

- a coluna alvo deve conter o desgaste medido da ferramenta;
- as variáveis de entrada devem ser numéricas;
- linhas com valores ausentes são tratadas por imputação mediana no pipeline;
- variáveis categóricas devem ser previamente codificadas ou removidas.

Exemplo de execução:

```bash
python src/main.py --data data/raw/dados_experimentais.csv --target Desgaste
```

Se o nome da coluna alvo for diferente, use `--target` para informar o nome correto.
