package com.example.demo;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.apache.kafka.common.protocol.types.Field;
import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.streams.KafkaStreams;
import org.apache.kafka.streams.StreamsBuilder;
import org.apache.kafka.streams.kstream.KStream;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.Properties;

import org.apache.kafka.streams.StreamsConfig;

@Service
public class KafkaStreamsService {

    private final WebClient webClient;

    public KafkaStreamsService() {
        this.webClient = WebClient.create("http://localhost:5000");
    }

    private void sendToPythonApi(String value) {
        webClient.post()
                .uri("/predict")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(value)
                .retrieve()
                .bodyToMono(String.class)
                .subscribe(response -> System.out.println("Prediction: " + response));
    }

    public void startStream() {
        Properties props = new Properties();
        props.put(StreamsConfig.APPLICATION_ID_CONFIG, "stream-app");
        props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost.:9092");
        props.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass());
        props.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass());

        StreamsBuilder builder = new StreamsBuilder();

        KStream<String, String> inputStream = builder.stream("postgres.public.users");

        inputStream.foreach((key, value) -> {
            try {
                String valueName = "salary";
                String variableName = "created_at";
                ObjectMapper objectMapper = new ObjectMapper();
                JsonNode jsonValue = objectMapper.readTree(value);
                ObjectNode payload = objectMapper.createObjectNode();
                payload.put(variableName, jsonValue.get("payload").get(variableName));
                payload.put(valueName, jsonValue.get("payload").get(valueName).asInt());
                sendToPythonApi(payload.toString());
            } catch (Exception e) {
                e.printStackTrace();
            }
        });

        KafkaStreams streams = new KafkaStreams(builder.build(), props);
        streams.start();
    }
}
