from kafka import KafkaProducer, KafkaConsumer

# Kafka Producer Example
def kafka_producer():
    producer = KafkaProducer(bootstrap_servers='localhost:9092')  # Replace with your Kafka broker address
    topic = 'example-topic'

    # Send a message to the topic
    producer.send(topic, b'Hello, Kafka!')
    producer.flush()
    producer.close()
    print(f"Message sent to topic: {topic}")

# Kafka Consumer Example
def kafka_consumer():
    consumer = KafkaConsumer(
        'example-topic',
        bootstrap_servers='localhost:9092',  # Replace with your Kafka broker address
        auto_offset_reset='earliest',
        group_id='example-group'
    )

    # Read messages from the topic
    for message in consumer:
        print(f"Received message: {message.value.decode('utf-8')}")

# Run the producer and consumer
if __name__ == "__main__":
    kafka_producer()
    kafka_consumer()