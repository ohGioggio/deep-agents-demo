# ui/gui (reserved)

A graphical front end goes here. It talks to the engine the same way the CLI does:

- open `startup.resources.Resources` once, then `startup.runtime.build_runtime(resources, user)`
  per user;
- implement the ports in `core/ports.py`: `EventSink` (show `models/events.py` events),
  `Interaction` (approvals, choices, questions) and call `Interruptible.request_pause()` /
  `stop()` on a running plan or room;
- run turns with `core.turns.run_turn(agent, prompt, config, sink, approvals.decide, runners)`
  on a worker thread (the ports are synchronous).

Nothing here yet; `ui/cli/` is the reference implementation.
