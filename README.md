# Despliegue de la API con Docker

## Desplegar (construir y levantar)
```
docker compose build --no-cache api
docker compose up -d
docker compose ps
```

## Verificar estado
```
curl http://localhost:8000/health
```

## Probar endpoints
### /predict
```
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" --data-binary "@api-test-data/predict.json"
```

### /retrain
```
curl -X POST "http://localhost:8000/retrain" -H "Content-Type: application/json" --data-binary "@api-test-data/retrain.json"
```

## Logs (opcional)
```
docker logs ods-api -f
```

## Detener
```
docker compose down
```
