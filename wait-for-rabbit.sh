#!/bin/sh
echo "⏳ Waiting for RabbitMQ at rabbitmq:5672..."

until nc -z rabbitmq 5672; do
  sleep 1
done

echo "✅ RabbitMQ is ready."
exec "$@"
