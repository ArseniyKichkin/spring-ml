package com.example.demo;


import org.apache.kafka.streams.KafkaStreams;
import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;

@RestController
@RequestMapping("/api/stream")
public class KafkaStreamsController {

    private final KafkaStreamsService kafkaStreamsService;
    private KafkaStreams streams;

    public KafkaStreamsController(KafkaStreamsService kafkaStreamsService) {
        this.kafkaStreamsService = kafkaStreamsService;
    }

    @PostMapping("/start")
    public ResponseEntity<String> startStream() {
        try {
            kafkaStreamsService.startStream();
            return ResponseEntity.ok("Kafka Streams processing started successfully");
        } catch (Exception e) {
            return ResponseEntity.internalServerError()
                    .body("Failed to start stream: " + e.getMessage());
        }
    }

    @PostMapping("/stop")
    public ResponseEntity<String> stopStream() {
        try {
            if (streams != null) {
                streams.close();
                return ResponseEntity.ok("Kafka Streams processing stopped successfully");
            }
            return ResponseEntity.ok("No active stream to stop");
        } catch (Exception e) {
            return ResponseEntity.internalServerError()
                    .body("Failed to stop stream: " + e.getMessage());
        }
    }

    @GetMapping("/status")
    public ResponseEntity<String> getStreamStatus() {
        if (streams != null && streams.state().isRunningOrRebalancing()) {
            return ResponseEntity.ok("Stream is RUNNING");
        }
        return ResponseEntity.ok("Stream is NOT RUNNING");
    }
}