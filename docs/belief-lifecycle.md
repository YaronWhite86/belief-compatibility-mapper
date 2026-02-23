# Belief Lifecycle

## State Transitions

```mermaid
stateDiagram-v2
    [*] --> Created : add_belief()
    Created --> Embedded : generate_embeddings()
    Embedded --> Scored : analyze_pair() / set_score()
    Scored --> Visualized : export_heatmap() / export_network()

    Created --> Edited : edit_belief(text=...)
    Embedded --> Edited : edit_belief(text=...)
    Scored --> Edited : edit_belief(text=...)
    Edited --> Embedded : generate_embeddings()

    note right of Edited
        Changing text clears
        embedding + all scores
        for this belief
    end note

    Created --> [*] : remove_belief()
    Embedded --> [*] : remove_belief()
    Scored --> [*] : remove_belief()
    Visualized --> [*] : remove_belief()

    Scored --> Snapshotted : save_snapshot()
    Snapshotted --> Scored : load_snapshot()
```

## ID Lifecycle

```mermaid
flowchart LR
    subgraph "ID Pool (0-14)"
        IDS["IDs are numpy matrix indices\nMAX_BELIEFS = 15"]
    end

    ADD["add_belief()"] -->|"_next_id()\nlowest unused"| IDS
    IDS -->|"remove_belief()\nfrees the ID"| FREED["Freed ID\nre-enters pool"]
    FREED -->|"next add_belief()\nreuses freed ID"| IDS

    subgraph "Cache Independence"
        CACHE_KEY["cache.db keys:\nSHA-256 of belief text\n(not belief ID)"]
    end

    IDS -.->|"ID changes\ndon't invalidate cache"| CACHE_KEY
```

### ID reuse example

| Step | Action | IDs in use | Next available |
|---|---|---|---|
| 1 | `add_belief("A")` | {0} | 1 |
| 2 | `add_belief("B")` | {0, 1} | 2 |
| 3 | `add_belief("C")` | {0, 1, 2} | 3 |
| 4 | `remove_belief(1)` | {0, 2} | **1** |
| 5 | `add_belief("D")` | {0, **1**, 2} | 3 |

Belief "D" gets ID 1 (the lowest free slot). Cache entries for the old belief "B" remain in `cache.db` keyed by the SHA-256 hash of `"B"` -- they are harmless orphans and do not interfere.
