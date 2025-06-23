# Run Optimization

## Build the training image

```
docker build -t hi4lines-gpu:latest .
```

## Make a network

```
docker network create ml-net 2>/dev/null || true
```

## Set up database

```
docker run -d \
  --name optuna-postgres \
  --network ml-net \
  -e POSTGRES_USER=optuna_user \
  -e POSTGRES_PASSWORD=secretpass \
  -e POSTGRES_DB=optuna_db \
  -v pgdata:/var/lib/postgresql/data \
  postgres:14
```

## Make database

```
docker run -d   --name optuna-postgres   --network ml-net   -e POSTGRES_USER=optuna_user   -e POSTGRES_PASSWORD=secretpass   -e POSTGRES_DB=optuna_db   postgres:14
```

## Run training

```
docker run --network ml-net --rm -it -v "$(pwd)":/app --ipc=host -w /app hi4lines-gpu:latest bash
```

or

```
docker run --gpus all --network ml-net --rm -it -v "$(pwd)":/app -w /app --ipc=host hi4lines-gpu:latest bash
```

## View dashboard

```
docker run -it --rm --network ml-net -p 8080:8080 ghcr.io/optuna/optuna-dashboard postgresql+psycopg2://optuna_user:secretpass@optuna-postgres:5432/optuna_db  
```
