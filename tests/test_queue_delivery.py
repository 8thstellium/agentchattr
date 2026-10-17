"""Queued wake prompts survive restarts, failures, and mixed destinations."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import wrapper


class _StopWatcher(BaseException):
    pass


class QueueDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.queue = Path(self.tmp.name) / "claude_queue.jsonl"

    def append(self, *, channel="general", job_id=None, prompt=""):
        entry = {"channel": channel, "text": "mention"}
        if job_id is not None:
            entry["job_id"] = job_id
        if prompt:
            entry["prompt"] = prompt
        with self.queue.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")

    def run_polls(self, inject, polls=1):
        completed = 0

        def sleep(seconds):
            nonlocal completed
            if seconds == 1:
                completed += 1
                if completed >= polls:
                    raise _StopWatcher()

        with (patch.object(wrapper.time, "sleep", side_effect=sleep),
              patch.object(wrapper, "_fetch_role", return_value=""),
              patch.object(wrapper, "_fetch_active_rules", return_value=None)):
            with self.assertRaises(_StopWatcher):
                wrapper._queue_watcher(lambda: ("claude", self.queue), inject)

    def test_existing_queue_is_delivered_on_watcher_start(self):
        self.append()
        prompts = []
        self.run_polls(prompts.append)
        self.assertEqual(len(prompts), 1)
        self.assertIn("read #general", prompts[0])

    def test_mixed_destinations_are_injected_separately(self):
        self.append(channel="general")
        self.append(channel="work")
        self.append(channel="general", job_id=17)
        self.append(channel="general", prompt="custom instruction")
        prompts = []
        self.run_polls(prompts.append)
        self.assertEqual(len(prompts), 4)
        self.assertIn("read #general", prompts[0])
        self.assertIn("read #work", prompts[1])
        self.assertIn("read job_id=17", prompts[2])
        self.assertIn("custom instruction", prompts[3])

    def test_failed_injection_is_retried_after_restart(self):
        self.append(channel="work")

        def fail(_prompt):
            raise RuntimeError("CLI unavailable")

        self.run_polls(fail)
        prompts = []
        self.run_polls(prompts.append)
        self.assertEqual(len(prompts), 1)
        self.assertIn("read #work", prompts[0])

    def test_processing_file_is_recovered_before_new_queue(self):
        processing = self.queue.with_suffix(".processing")
        processing.write_text(json.dumps({"channel": "old"}) + "\n", encoding="utf-8")
        self.append(channel="new")
        prompts = []
        self.run_polls(prompts.append, polls=2)
        self.assertEqual(len(prompts), 2)
        self.assertIn("read #old", prompts[0])
        self.assertIn("read #new", prompts[1])

    def test_new_append_during_processing_survives(self):
        self.append(channel="old")
        prompts = []

        def inject(prompt):
            prompts.append(prompt)
            if len(prompts) == 1:
                self.append(channel="new")

        self.run_polls(inject, polls=2)
        self.assertEqual(len(prompts), 2)
        self.assertIn("read #old", prompts[0])
        self.assertIn("read #new", prompts[1])

    def test_failed_second_destination_retries_only_the_second(self):
        self.append(channel="first")
        self.append(channel="second")
        prompts = []

        def fail_second(prompt):
            prompts.append(prompt)
            if "read #second" in prompt:
                raise RuntimeError("CLI unavailable")

        self.run_polls(fail_second)
        retry = []
        self.run_polls(retry.append)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(len(retry), 1)
        self.assertIn("read #second", retry[0])

    def test_open_queue_on_windows_is_retried(self):
        self.append(channel="work")
        real_replace = wrapper.os.replace
        attempts = 0

        def busy_once(source, target):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise PermissionError("queue held open by writer")
            return real_replace(source, target)

        prompts = []
        with patch.object(wrapper.os, "replace", side_effect=busy_once):
            self.run_polls(prompts.append, polls=2)
        self.assertEqual(len(prompts), 1)
        self.assertIn("read #work", prompts[0])


if __name__ == "__main__":
    unittest.main()
