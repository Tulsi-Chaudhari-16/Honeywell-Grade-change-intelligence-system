#!/bin/bash
# Wait for Kafka to be ready
echo "Waiting for Kafka to be ready..."
sleep 15

echo "Creating topics..."
kafka-topics.sh --create --if-not-exists --bootstrap-server kafka:29092 --topic gcis.tags.raw --partitions 3 --replication-factor 1
kafka-topics.sh --create --if-not-exists --bootstrap-server kafka:29092 --topic gcis.retrain_requests --partitions 1 --replication-factor 1

echo "Topics created successfully."
