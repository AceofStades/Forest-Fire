package com.forestfire.core.entity;

import jakarta.persistence.*;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.time.LocalDateTime;

@Entity
@Table(name = "simulation_scenarios")
@Data
@NoArgsConstructor
public class SimulationScenario {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    private String name;

    @Column(columnDefinition = "TEXT")
    private String configurationJson; // Store grid or params as JSON

    @Column(name = "created_at")
    private LocalDateTime createdAt = LocalDateTime.now();
}
