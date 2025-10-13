# Despliegue de la App con Docker

## Desplegar (construir y levantar)
```
docker compose build --no-cache api
docker compose up -d
docker compose ps
```
Tras correr compose up, debería esperar alrededor de 1 minuto para que la API esté lista. Luego puede proceder a acceder a la interfaz o a probar los endpoints.

## Acceder a la interfaz
```
http://localhost:8000
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
