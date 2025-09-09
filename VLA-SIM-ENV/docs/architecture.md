# Architecture

- **SceneLoader**: Parses YAML configs into normalized scene graphs and object specs.
- **ObjectManager**: Applies physical parameters (mass, friction, restitution) and materials.
- **InteractionEngine**: Canonical actions (pick, place, push, navigate). Bridges VLA intents to env ops.
- **OmniverseAdapter / GenesisAdapter**: Lazy imports, capability discovery, graceful fallback to **MockSim**.
- **Utils**: Logging, visualization, and basic metrics for sanity checking.
