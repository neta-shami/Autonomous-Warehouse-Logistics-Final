import queue
import threading
import time
from typing import Tuple

import mujoco
import mujoco.viewer

from adapters.application_factory import ApplicationRuntime, build_application
from adapters.cli_parser import CLIParser
from entities.enums import FSMStatus, TaskType
from entities.task import Task
from entities.warehouse_topology import WarehouseTopology


def build_contention_demo_tasks(
    topology: WarehouseTopology,
) -> Tuple[Task, ...]:
    """Create two MOVE jobs plus one deliberately conflicting request."""
    task_specs = (
        ("demo_p4_primary", "p4", (6, 6), 0, (2, 6), 0),
        ("demo_p4_conflict", "p4", (6, 6), 0, (4, 6), 0),
        ("demo_p3_vertical", "p3", (4, 4), 1, (4, 2), 1),
    )
    tasks = []
    for (
        task_id,
        payload_id,
        source,
        source_tier,
        target,
        target_tier,
    ) in task_specs:
        topology.validate_shelf_slot(
            source, source_tier, field_name=f"{task_id} source"
        )
        topology.validate_shelf_slot(
            target, target_tier, field_name=f"{task_id} target"
        )
        tasks.append(
            Task.transfer(
                task_id=task_id,
                task_type=TaskType.RELOCATE,
                source_coords=source,
                source_tier=source_tier,
                target_coords=target,
                target_tier=target_tier,
                payload_id=payload_id,
            )
        )
    return tuple(tasks)


class VisualizationDemoWorkload:
    """Advance a valid repeated MOVE/RETRIEVE/STORE demonstration."""

    def __init__(self, runtime: ApplicationRuntime) -> None:
        self.runtime = runtime
        self.stage = 0
        self.complete = False

    def start(self) -> None:
        """Start a four-robot stage containing every package operation."""
        print("Demo stage 1: adding p5, giving p2, and moving p3 and p4.")
        if not self._spawn_store("p5", (2, 2), 0):
            return
        self._submit_commands(("RETRIEVE p2 2,5 0",))
        for task in build_contention_demo_tasks(self.runtime.topology):
            self.runtime.reservations.process_new_task(
                task, self.runtime.environment.get_time()
            )
            state = (
                "ready"
                if task in self.runtime.reservations.ready_queue
                else "blocked by the existing package lease"
            )
            print(
                f"  {task.task_id}: {task.payload_id} "
                f"{task.source_coords}->{task.target_coords} [{state}]"
            )

    def advance_if_settled(self) -> None:
        """Submit the next dock stage only after physical state is stable."""
        if self.complete or not self._stage_is_settled():
            return
        if self.stage == 0:
            if not self._spawn_store("p7", (2, 5), 0):
                return
            self._submit_commands(
                (
                    "RETRIEVE p6 8,3 0",
                    "MOVE 4,2 1 8,7 1",
                    "MOVE 2,6 0 8,5 0",
                )
            )
            self.stage = 1
            print("Demo stage 2: adding p7, giving p6, and moving p3 and p4.")
            return
        self.complete = True
        print("Automated demo complete: 4 MOVE, 2 RETRIEVE, and 2 STORE tasks.")

    def _stage_is_settled(self) -> bool:
        expected_throughput = (4, 8)[self.stage]
        robots_idle = all(
            agent.fsm_status is FSMStatus.IDLE
            for agent in self.runtime.robots.values()
        )
        return (
            self.runtime.metrics.total_throughput >= expected_throughput
            and robots_idle
            and not self.runtime.reservations.ready_queue
            and not self.runtime.scenario.despawn_timers
        )

    def _submit_commands(self, commands: Tuple[str, ...]) -> None:
        tasks = CLIParser.parse_commands(list(commands), self.runtime.topology)
        if len(tasks) != len(commands):
            raise RuntimeError("Automated visualization stage is invalid")
        for task in tasks:
            self.runtime.reservations.process_new_task(
                task, self.runtime.environment.get_time()
            )

    def _spawn_store(
        self, package_id: str, target: Tuple[int, int], tier: int
    ) -> bool:
        if not self.runtime.orchestrator.spawn_package(package_id, target, tier):
            self.complete = True
            print(
                f"Automated demo stopped before {package_id}: its package or "
                "inbound dock was already used by an interactive command."
            )
            return False
        return True


def hide_rangefinder_rays(viewer) -> None:
    """Hide MuJoCo's yellow sensor-ray overlay without disabling sensors."""
    flag = int(mujoco.mjtVisFlag.mjVIS_RANGEFINDER)
    viewer.opt.flags[flag] = False


def configure_initial_view(viewer, topology: WarehouseTopology) -> None:
    """Frame the complete warehouse from an elevated diagonal free camera."""
    width = topology.max_x - topology.min_x
    height = topology.max_y - topology.min_y
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.fixedcamid = -1
    viewer.cam.trackbodyid = -1
    viewer.cam.lookat[:] = (
        (topology.min_x + topology.max_x) / 2.0,
        (topology.min_y + topology.max_y) / 2.0,
        0.5,
    )
    viewer.cam.distance = 1.6 * max(width, height)
    viewer.cam.azimuth = 90.0
    viewer.cam.elevation = -45.0


def format_robot_status(robot_id, agent, frame=None):
    """Format belief state without assuming localization has initialized."""
    estimate = agent.state.pose_estimate
    if estimate is None:
        location = "unknown"
    else:
        location = f"({estimate.pose.x:.1f},{estimate.pose.y:.1f})"

    payload = {
        True: "present",
        False: "empty",
        None: "unknown",
    }[agent.state.payload_present]
    prefix = f"[{frame}] " if frame is not None else ""
    return (
        f"  {prefix}{robot_id}: {agent.fsm_status.name} @ {location} "
        f"payload={payload}"
    )


def cli_input_thread(cmd_queue):
    print("\n==========================================")
    print("   INTERACTIVE WAREHOUSE SIMULATOR")
    print("==========================================")
    print("Commands:")
    print("  STORE pkg_id tgt_x,tgt_y tier")
    print("  RETRIEVE pkg_id src_x,src_y tier")
    print("  MOVE src_x,src_y src_tier tgt_x,tgt_y tgt_tier")
    print("  status | help")
    print("==========================================\n")
    while True:
        try:
            line = input("sim> ").strip()
            if not line:
                continue
            if line.lower() == 'help':
                print("  STORE pkg_id tgt_x,tgt_y tier")
                print("      -> Gets a new package at the Inbound Dock (9,1) and stores it at the target.")
                print("  RETRIEVE pkg_id src_x,src_y tier")
                print("      -> Retrieves a package from the shelf and ships it out via the Outbound Dock (9,8).")
                print("  MOVE src_x,src_y src_tier tgt_x,tgt_y tgt_tier")
                print("      -> Moves a package from the source shelf to the target shelf.")
                continue
            cmd_queue.put(line)
        except (KeyboardInterrupt, EOFError):
            break


def main():
    print("Initializing Multi-Agent Warehouse Simulation...")
    runtime = build_application()
    model, data, topology = runtime.model, runtime.data, runtime.topology
    orchestrator = runtime.orchestrator
    reservation_service = runtime.reservations
    environment = runtime.environment
    metrics = runtime.metrics

    cmd_queue = queue.Queue()
    input_thread = threading.Thread(target=cli_input_thread, args=(cmd_queue,), daemon=True)
    input_thread.start()

    demo_workload = VisualizationDemoWorkload(runtime)
    demo_workload.start()
    print(
        "Each stage has one STORE, one RETRIEVE, and two MOVE transfers. "
        "The duplicate p4 request must stay blocked by its package lease."
    )

    print(
        "Type commands like 'STORE p8 4,4 0' in the terminal "
        "to give more tasks!"
    )

    print("Launching MuJoCo Viewer. Close window or press ESC to exit.")

    # How much simulated time one orchestrator step covers, so playback runs at
    # roughly real speed instead of racing ahead.
    tick_duration = orchestrator.physics_steps_per_logic_tick * model.opt.timestep

    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_initial_view(viewer, topology)
        hide_rangefinder_rays(viewer)
        frame = 0
        while viewer.is_running():
            t0 = time.time()

            while not cmd_queue.empty():
                cmd_str = cmd_queue.get_nowait()
                if cmd_str.lower() == "status":
                    for rid, agent in orchestrator.agents.items():
                        print(format_robot_status(rid, agent))
                    print(f"  Throughput: {metrics.total_throughput}")
                    continue
                parsed = CLIParser.parse_commands([cmd_str], topology)
                for p in parsed:
                    try:
                        if p.task_type == TaskType.STORE:
                            pkg_id = (
                                p.task_id[3:]
                                if p.task_id.startswith("st_")
                                else p.task_id
                            )
                            accepted = orchestrator.spawn_package(
                                pkg_id, p.target_coords, p.target_tier
                            )
                            print(
                                f"  Store {'requested' if accepted else 'rejected'} "
                                f"for {pkg_id}"
                            )
                        else:
                            reservation_service.process_new_task(
                                p, environment.get_time()
                            )
                            print(
                                f"  Task queued: {p.task_id} ({p.task_type.name})"
                            )
                    except ValueError as exc:
                        print(f"  Command rejected: {exc}")

            orchestrator.step()
            demo_workload.advance_if_settled()
            viewer.sync()

            frame += 1
            if frame % 500 == 0:
                for rid, agent in orchestrator.agents.items():
                    print(format_robot_status(rid, agent, frame))

            elapsed = time.time() - t0
            sleep_time = tick_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("\nSimulation Closed. Final Metrics:")
    print(metrics.generate_report())


if __name__ == "__main__":
    main()
