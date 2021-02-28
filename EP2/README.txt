O diretório src contém a seguinte estrutura:

 src
    ├── data
    ├── model_data
    ├── nn
    │   ├── nn.ipynb
    │   └── nn.py
    │   
    ├── pre
    │   ├── preprocessamento.ipynb
    │   └── preprocessamento.py
    │   
    ├── rel
    │   ├── figuras
    │   └── relatorio-ep2.pdf
    │   
    └── requirements.txt


**Instruções**

* No diretório "data" residem o conjunto de dados para treino/validação/teste e
o arquivo com os embeddings de 50 dimensões do NILC.


** Préprocessamento**

* Para realizar o preprocessamento e treinar os modelos, é necessário adicionar
os arquivos "B2W-Reviews01.csv" e "cbow_s50.txt" (embedding do NILC de 50 dimensões)
no diretório "data".

* No diretório "pre" residem os arquivos "preprocessamento.ipynb" e "preprocessamento.py".
Ambos contém o mesmo código que realiza a etapa de préprocessamento. No entanto,
no jupyter notebook contém o histórico do preprocessamento realizado durante o desenvolvimento do EP2.

Qualquer um destes arquivos pode ser utilizado para realizar o préprocessamento. Normalmente o script
em python pode ser mais conveniente, para se utilizar, basta acessar o diretório com o script pelo terminal
e executar o seguinte comando 'python3 preprocessamento.py'. Quando o preprocessamento é finalizado
algumas mensagens são apresentadas na tela e os conjuntos de dados de treinamento, validação e de testes
são salvos neste mesmo diretório.


**Treinamento e testes dos modelo**

* No diretório "nn" residem os arquivos "nn.ipynb" e "nn.py". De maneira análoga ambos possuem o mesmo código,
com a diferença que o jupyter notebook contém o histórico dos experimentos realizados no desenvolvimento do EP2.
Para treinar, validar e testar as redes neurais, basta executar o script no terminal, ou executar os códigos
das células no notebook, que os resultados são imprimidos na tela. Os modelos durante a época em que
apresentarem o melhor desempenho seram salvos no diretório "model_data". Além dos modelos o dicionário "history"
com as acurácias de treino de validação também são salvos neste mesmo diretório. As figuras com as curvas
de treino e validação são salvas no diretório "rel\figuras".

Nos códigos disponibilizados seguem funções que podem ser utilizadas para carregar os modelos e as acurácias. Os
resultados que você vai obter executar os scripts disponibilizados devem ser semelhantes aos que foram apresentados
no relatório visto que foi utilizado um seed fixo no preprocessamento.

Por último salienta-se ainda, que a etapa do preprocessamento de converter as palavras em índices é
realizada nos arquivos "nn", pois como existem diferentes métodos de vetorizar as sentenças, decidiu em salvar
os dados préprocessados em formato de texto.


**Relatório**

O relatório se encontra no diretório "rel".

**Requirements**

O diretório src também inclui o arquivo "requirements.txt" que contém as bibliotecas necessárias
para executar os códigos.
