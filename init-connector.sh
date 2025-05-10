#!/bin/bash

# Ожидаем доступности Debezium Connect
until curl -f http://debezium-connect:8083/; do
  echo "Waiting for Debezium Connect to be available..."
  sleep 5
done

# Создаем коннектор
curl -X POST -H "Content-Type: application/json" http://debezium-connect:8083/connectors -d '{
  "name": "postgres-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "'"$DB_HOST"'",
    "database.port": "'"$DB_PORT"'",
    "database.user": "'"$DB_USER"'",
    "database.password": "'"$DB_PASSWORD"'",
    "database.dbname": "'"$DB_NAME"'",
    "topic.prefix": "postgres",
    "table.include.list": "public.*",
    "plugin.name": "pgoutput",
    "slot.name": "debezium",
    "publication.name": "dbz_publication",
    "snapshot.mode": "initial",
    "transforms": "unwrap",
    "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
    "transforms.unwrap.drop.tombstones": "false",
    "transforms.unwrap.delete.handling.mode": "rewrite"
  }
}'

echo "PostgreSQL connector created successfully"