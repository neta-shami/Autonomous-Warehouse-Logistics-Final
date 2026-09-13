"""Fixed-length scripted warehouse simulation entry point."""

from adapters.application_factory import build_application
from adapters.cli_parser import CLIParser


def main() -> None:
    print("Initializing Multi-Agent Warehouse Simulation...")
    runtime = build_application()
    commands = (
        "MOVE 2,5 0 8,5 0",
        "MOVE 4,4 1 4,2 1",
        "MOVE 6,6 0 2,6 0",
        "MOVE 8,3 0 8,7 0",
    )
    for task in CLIParser.parse_commands(list(commands), runtime.topology):
        runtime.reservations.process_new_task(
            task,
            runtime.environment.get_time(),
        )

    logic_ticks = 1_000
    print(f"Starting headless simulation for {logic_ticks} logic ticks...")
    for tick in range(logic_ticks):
        runtime.orchestrator.step()
        if tick % 100 == 0:
            print(f"Logic Tick {tick}/{logic_ticks}")

    print("\nSimulation Complete. Generating Report:")
    print(runtime.metrics.generate_report())


if __name__ == "__main__":
    main()
