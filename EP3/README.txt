O diretório src contém a seguinte estrutura:

src
   ├── data
   │   ├── bert_human_review_2 - bert_human_review_2.csv
   │   └── bilstm_human_review - bilstm_human_review_Alexandre.csv
   ├── nn
   │   ├── Bert.ipynb
   │   ├── nn_bilstm.ipynb
   │   ├── stats_human_score_bert.ipynb
   │   └── stats_human_score_bilstm.ipynb
   ├── rel
   │   ├── figuras
   │   │   ├── figura10.png
   │   │   ├── figura11.png
   │   │   ├── figura1.png
   │   │   ├── figura2.png
   │   │   ├── figura3.png
   │   │   ├── figura4.png
   │   │   ├── figura5.png
   │   │   ├── figura6.png
   │   │   ├── figura7.png
   │   │   ├── figura8.png
   │   │   └── figura9.png
   │   └── relatorio_ep03_peressim.pdf
   └── requirements.txt

**Instruções**

* No diretório "data" devem residir o conjunto de dados para treino/validação/teste, além
disso também residem os csv's com as pontuações dos títulos da avaliação humana.


** Préprocessamento, Treinamento e testes dos modelos**

* Para realizar o preprocessamento e treinar os modelos, é necessário adicionar
o arquivo "B2W-Reviews01.csv" se desejar usar o dataset completo, ou, o arquivo
b2w-10k.csv caso queira apenas testar o dataset de 10k, no  diretório "data".

Esses arquivos não são necessários para se treinar os modelos, o código fornecido possui
tanto as opções de usar os csvs mencionados no diretório "data" quanto opções de baixar
os conjuntos de dados diretamente do github.

Dependendo do conjunto de dados escolhido para se usar e se será obtido direto do diretório
"data" ou não, é necessário descomentar a linha de código equivalente nos notebooks. Por exemplo,
na sétima célula dos notebooks você encontrara o seguinte trecho de código:

# path = '../data/b2w-10k.csv'
# sep = ','

# path = "https://raw.githubusercontent.com/feperessim/NLPortugues/master/Semana%2006/data/b2w-10k.csv"
# sep = ','

# path = '../data/B2W-Reviews01.csv
# sep = ';'

path = 'https://raw.githubusercontent.com/b2wdigital/b2w-reviews01/master/B2W-Reviews01.csv'
sep = ';'

Basta descomentar o código referente ao conjunto em que você desejar fazer os testes e comentar
aqueles que não serão utilizados. O Sep é um argumento que não pode ficar de fora, pois o conjunto
completo e o de 10k usam separadores diferentes.

Observação: Durante os experimentos finais o dataset completo foi utilizado.

* No diretório "nn" residem os arquivos "Bert.ipynb", "nn_bilstm.ipynb", "stats_human_score_bert.ipynb",
"stats_human_score_bilstm.ipynb". O arquivo "Bert.ipynb" possuí o código completo de preprocessamento, preparação,
treinamento, validação e testes do modelo BERT, além disso também possuí o histórico com os resultados dos experimentos
realizados no desenvolvimento deste exercício programa. De maneira semelhante o arquivo "nn_bilstm.ipynb" também possuí
o código completo de preprocessamento, preparação, treinamento, validação e testes do modelo Encoder-Decoder BiLSTM, além disso
também possuí o histórico com os resultados dos experimentos realizados no desenvolvimento deste exercício programa. Para realizar
os mesmos experimentos basta realizar o procedimento explicado anteriormente em relação aos dados necessários para tal, e então, executar
cada célula do notebook individualmente. Por último, recomenda-se ainda usar o Gooogle Colaboratory para realizar os experimentos, pois

Os arquivos  "stats_human_score_bert.ipynb", "stats_human_score_bilstm.ipynb" possuem o histórico da análise estatística e os gráficos
gerados com os dados das pontuações da avaliação humana realizada com os títulos gerados pelos BERT e Encoder-Decoder BiLSTM respectivamente.

**Relatório**

O relatório se encontra no diretório "rel". No diretório "figuras" se encontram as figuras utilizadas no relatório.

**Requirements**

O diretório src também inclui o arquivo "requirements.txt" que contém as bibliotecas necessárias
para executar os códigos.
