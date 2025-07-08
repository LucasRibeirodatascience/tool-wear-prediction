# 🔧 tool-wear-prediction

Previsão de desgaste da ferramenta de corte utilizando Machine Learning e sinais de vibração.

---

## 📌 Descrição

Este projeto aplica algoritmos de aprendizado de máquina (incluindo redes neurais e modelos ensemble) para prever a vida útil de ferramentas de corte com base em sinais de vibração capturados durante o processo de torneamento.

---

## 🧠 Modelos Utilizados

- Redes Neurais (Keras)
- Support Vector Regressor (SVR)
- Extra Trees Regressor
- XGBoost Regressor
- LightGBM Regressor
- Gradient Boosting Regressor
- Meta-modelo (Linear Regression com Stacking)

---

## 🗂 Estrutura do Projeto

```bash
tool-wear-prediction/
│
├── src/                    # Código-fonte do projeto
│   ├── main.py            # Pipeline principal de treinamento e avaliação
│   ├── preprocessing.py   # Seleção de variáveis
│   └── model_utils.py     # Criação dos modelos e meta-aprendiz
│
├── requirements.txt       # Dependências
├── README.md              # Este arquivo
└── .gitignore


⚙️ Como Executar

▶️ 1. Instale as dependências

pip install -r requirements.txt

▶️ 2. Coloque seus dados

Adicione seu arquivo de dados CSV (exemplo: meus_dados.csv) no diretório principal.
No src/main.py, altere a linha:

data = pd.read_csv('seu_arquivo.csv')
para
data = pd.read_csv('meus_dados.csv')
Certifique-se de que as colunas estejam com os mesmos nomes usados no projeto.

▶️ 3. Execute o projeto

python src/main.py
Ou use no Google Colab (recomendo criar um notebook com chamadas para os módulos .py).

📈 Resultados Esperados
O projeto irá:

Realizar validação cruzada com os modelos base

Empilhar as previsões (stacking)

Calcular o R² final

Gerar gráficos comparativos com valores reais vs. previstos

👨‍💻 Autor
Lucas Ribeiro
LinkedIn

## 📜 Licença

Este projeto está licenciado sob a Licença MIT - sinta-se livre para usar, modificar e distribuir com atribuição.

Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
