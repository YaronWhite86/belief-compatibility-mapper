# Data Models

## Class Diagram

```mermaid
classDiagram
    class Belief {
        +int id  [0-14]
        +str text
        +str expanded
        +list~float~ embedding
        +list~str~ tags
        +BeliefRole role
    }

    class BeliefRole {
        <<enum>>
        SELF_SCHEMA
        ASPIRATION
        SAFETY_STRATEGY
        LIMITING_BELIEF
        BRIDGE_BELIEF
        CORE_VALUE
        UNTAGGED
    }

    class TensionResult {
        +float score  [-1.0 to +1.0]
        +TensionCategory category
        +str justification
    }

    class TensionCategory {
        <<enum>>
        ENTAILED = "mutually_entailed"
        HARMONIOUS = "compatible_harmonious"
        NEUTRAL = "neutral"
        TENSIONED = "tensioned"
        CONTRADICTORY = "contradictory"
    }

    class BedrockPrinciple {
        +str principle
        +list~int~ belief_ids  [min 2]
        +float coherence  [0.0-1.0]
        +str explanation
    }

    class BeliefRecommendation {
        +str text
        +str justification
    }

    class DissonanceAlert {
        +int belief_id_a  [0-14]
        +int belief_id_b  [0-14]
        +float score  [-1.0 to +1.0]
        +list~int~ dependent_ids
        +float severity  [0.0-1.0]
    }

    class SimulationResult {
        +int removed_id
        +str removed_text
        +list~int~ stable_ids
        +list~int~ destabilized_ids
        +list~int~ orphaned_ids
    }

    Belief --> BeliefRole : has role
    TensionResult --> TensionCategory : categorized as
    BedrockPrinciple --> Belief : references belief_ids
    DissonanceAlert --> Belief : references belief_id_a, belief_id_b
    SimulationResult --> Belief : references removed_id
```

## Score Scale

| TensionCategory | Value | Score Range | Description |
|---|---|---|---|
| `mutually_entailed` | `"mutually_entailed"` | +0.8 to +1.0 | One belief necessitates the other |
| `compatible_harmonious` | `"compatible_harmonious"` | +0.1 to +1.0 | Support the same worldview |
| `neutral` | `"neutral"` | -0.3 to +0.3 | Unrelated beliefs |
| `tensioned` | `"tensioned"` | -1.0 to -0.1 | Difficult to hold both simultaneously |
| `contradictory` | `"contradictory"` | -1.0 to -0.4 | Logically impossible to hold both |

> **Note:** Ranges overlap intentionally. The LLM chooses both score and category; `engine.py` logs a warning if they are inconsistent but does not reject the result.
