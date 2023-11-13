from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
api = Api(app)

# Carregando os dados
data = pd.read_csv('RIOSP.csv', delimiter=';', encoding='latin-1')

# Substituindo vírgulas por pontos e convertendo para float
data['km'] = data['km'].str.replace(',', '.').astype(float)

# Selecionando as variáveis relevantes para o clustering
X = data[['km', 'automovel', 'bicicleta', 'caminhao', 'moto', 'onibus', 'outros', 'tracao_animal',
          'transporte_de_cargas_especiais', 'trator_maquinas', 'utilitarios', 'ilesos',
          'levemente_feridos', 'moderadamente_feridos', 'gravemente_feridos', 'mortos']]

# Tratando valores nulos
X.fillna(0, inplace=True)

# Normalizando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicando o K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data['cluster'] = kmeans.fit_predict(X_scaled)

class ClusterResource(Resource):
    def get(self):
        # Retorna os dados com os clusters atribuídos
        return jsonify(data.to_dict(orient='records'))

api.add_resource(ClusterResource, '/clusters')

if __name__ == '__main__':
    app.run(debug=True)

    