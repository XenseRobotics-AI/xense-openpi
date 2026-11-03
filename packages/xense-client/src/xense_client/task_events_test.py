from types import SimpleNamespace

from xense_client.visualization import ObservedEnvironment
from xense_client.visualization import StateSample
from xense_client.visualization import VisualizationConfig
from xense_client.visualization import VisualizationSession


class Display:
    def __init__(self):
        self.tasks = []
        self.images = []

    def publish(self, frame, **kwargs):
        return True

    def publish_images(self, at, images, **kwargs):
        self.images.append((at, images, kwargs))
        return True

    def update_task(self, snapshot):
        self.tasks.append(snapshot)
        return True

    def close(self):
        return True


def test_runtime_events_do_not_treat_final_reset_as_a_new_episode():
    display = Display()
    adapter = SimpleNamespace(describe=object, read_state=lambda at: StateSample(None))
    session = VisualizationSession(
        VisualizationConfig(),
        lambda: adapter,
        display_factory=lambda _: display,
        instruction_metadata={"text": "Pick up the cup", "source": "policy", "applies_when": "request_has_no_prompt"},
        dry_run=True,
    )
    observed = {"head": object(), "left_wrist": object()}
    calls = []
    env = ObservedEnvironment(
        SimpleNamespace(
            reset=lambda: calls.append("reset"), get_observation=lambda: observed, apply_action=calls.append
        ),
        session.stream,
        events=session,
    )
    session.set_enabled(True)
    try:
        env.reset()
        session.on_episode_start()
        session.publish_observation(observed)
        assert env.get_observation() is observed
        env.apply_action({"actions": [1, 2]})
        session.on_step({}, {})
        session.on_step({}, {})
        session.on_episode_end()
        env.reset()
        session.finish("Run ended")
    finally:
        session.close()
    assert calls == ["reset", {"actions": [1, 2]}, "reset"]
    assert display.images[0][1] is observed
    last = display.tasks[-1]
    assert last.episode == 1
    assert last.instruction == "Pick up the cup"
    text = [event.text for event in last.events]
    assert text.count("Episode 1 started") == 1
    assert text.count("Episode 1 executing steps") == 1
    assert text.count("Reset completed") == 2
    assert any("no robot actions sent" in task.status for task in display.tasks)
    assert last.status == "Run ended"


def test_task_history_is_bounded_and_labels_lost_history():
    display = Display()
    session = VisualizationSession(
        VisualizationConfig(), lambda: SimpleNamespace(describe=object), display_factory=lambda _: display
    )
    session.prepare()
    for i in range(300):
        session.report_event(str(i), "Running")
    session.close()
    assert len(display.tasks[-1].events) == 256
    assert display.tasks[-1].dropped_events == 44
    assert display.tasks[-1].instruction is None
