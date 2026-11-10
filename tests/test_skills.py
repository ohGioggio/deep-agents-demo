"""Agent-written skills: format, versioned store, planning, approval and enforcement."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from deep_agents_demo.agent import build_agent
from deep_agents_demo.approvals import NO_HUMAN, Approver
from deep_agents_demo.chat import stream_turn
from deep_agents_demo.skill_manager import (GATED_TOOLS, SkillManager, read_skill_folder,
                                            render_skill_md, validate)
from deep_agents_demo.skill_store import SkillStore

KEY = "sk-proj-abcDEF1234567890ghijKLMNOP"


class Workspace(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "skills").mkdir()
        self.store = SkillStore(self.root / "skills.sqlite")
        self.manager = SkillManager(self.store, "alice", self.root)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def create(self, name="greeting-style", **kwargs):
        change = self.manager.plan_create(name, kwargs.pop("description", "How to greet. Use at the start."),
                                          kwargs.pop("instructions", "1. Say hi."), **kwargs)
        return self.manager.apply(change)


class FormatTests(unittest.TestCase):
    def test_rendered_skill_is_valid_and_loadable(self):
        files = {"SKILL.md": render_skill_md("my-skill", "Does X: when Y.\nSecond line", "Steps")}
        self.assertEqual(validate("my-skill", files), [])
        self.assertIn("description: 'Does X: when Y. Second line'", files["SKILL.md"])

    def test_invalid_skills_are_explained(self):
        good = render_skill_md("my-skill", "Does X.", "Steps")
        cases = {
            "Bad_Name": ({"SKILL.md": render_skill_md("Bad_Name", "d", "s")}, "lowercase"),
            "my-skill": ({"notes.md": "x"}, "SKILL.md is missing"),
            "other-name": ({"SKILL.md": good}, "does not match"),
        }
        for name, (files, expected) in cases.items():
            with self.subTest(name=name):
                self.assertTrue(any(expected in e for e in validate(name, files)), validate(name, files))
        self.assertTrue(validate("my-skill", {"SKILL.md": "no frontmatter"}))
        for path in ("../escape.md", "/abs.md", "C:/win.md", "a\\b.md"):
            self.assertTrue(validate("my-skill", {"SKILL.md": good, path: "x"}), path)
        secret = validate("my-skill", {"SKILL.md": good, "config.md": f"key = {KEY}"})
        self.assertTrue(any("credential" in e for e in secret))
        self.assertNotIn(KEY, " ".join(secret))


class SecretRuleTests(Workspace):
    def test_prose_about_keys_is_a_warning_the_user_decides_on(self):
        change = self.manager.plan_create(
            "api-setup", "How to configure the API.",
            "The API key is read from the OPENAI_API_KEY environment variable.\n"
            "Make sure the token is rotated every 90 days.")
        self.assertEqual(change.errors, [])
        self.assertEqual(len(change.warnings), 1)
        self.assertIn("Warning: SKILL.md mentions a password, token or key", change.summary())
        approver = Approver(self.manager.plan, ask=lambda _: "a", say=lambda *_: None)
        request = {"action_requests": [{"name": "create_skill", "args": {
            "name": "api-setup", "description": "How to configure the API.",
            "instructions": "The token is rotated every 90 days."}}]}
        self.assertEqual(approver.decide(request)["decisions"], [{"type": "approve"}])

    def test_concrete_credentials_are_still_rejected(self):
        change = self.manager.plan_create("leaky", "Leaky.", f"Use {KEY} to call the API.")
        self.assertTrue(any("credential" in e for e in change.errors))
        self.assertNotIn(KEY, change.summary())


class StoreTests(Workspace):
    def test_versions_remove_and_restore(self):
        self.assertEqual(self.create()["version"], 1)
        self.manager.apply(self.manager.plan_update("greeting-style", instructions="1. Say hello."))
        self.manager.apply(self.manager.plan_remove("greeting-style"))
        self.assertEqual(self.store.list_current("user", "alice"), [])
        history = self.store.history("user", "alice", "greeting-style")
        self.assertEqual([v.action for v in history], ["create", "update", "remove"])
        restored = self.manager.apply(self.manager.plan_restore("greeting-style", 1))
        self.assertEqual(restored["version"], 4)
        self.assertIn("1. Say hi.", self.store.get("user", "alice", "greeting-style").files["SKILL.md"])

    def test_users_are_separate(self):
        self.create()
        bob = SkillManager(self.store, "bob", self.root)
        self.assertEqual(self.store.list_current("user", "bob"), [])
        self.assertTrue(bob.plan_update("greeting-style", instructions="x").errors)


class PlanningTests(Workspace):
    def test_update_changes_only_what_is_given(self):
        self.create(files={"template.md": "Hello {name}"})
        change = self.manager.plan_update("greeting-style", description="New description.")
        self.assertEqual(change.errors, [])
        self.assertIn("New description.", change.after["SKILL.md"])
        self.assertIn("1. Say hi.", change.after["SKILL.md"])
        self.assertEqual(change.after["template.md"], "Hello {name}")
        self.assertEqual((change.version_from, change.version_to), (1, 2))
        self.assertIn("+description: New description.", change.diff())

    def test_planning_errors(self):
        self.create()
        self.assertIn("already exists", self.manager.plan_create("greeting-style", "d", "s").errors[0])
        self.assertTrue(self.manager.plan_update("missing", instructions="x").errors)
        self.assertTrue(self.manager.plan_update("greeting-style", delete_files=["SKILL.md"]).errors)
        self.assertTrue(self.manager.plan_restore("greeting-style", 9).errors)
        self.assertTrue(self.manager.plan("create_skill", {"nam": "x"}).errors)
        self.assertIsNone(self.manager.plan("execute", {"command": "ls"}))

    def test_import_from_a_folder(self):
        folder = self.root / "incoming" / "street-works"
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(render_skill_md("street-works", "Worksite checklist.", "1. Check."))
        (folder / "template.md").write_text("Fields: code, street")
        (folder / ".hidden").write_text("ignored")
        change = self.manager.plan_import(str(folder))
        self.assertEqual((change.name, change.errors), ("street-works", []))
        self.assertEqual(sorted(change.after), ["SKILL.md", "template.md"])
        self.assertEqual(read_skill_folder(folder / "SKILL.md"), change.after)
        (folder / "logo.bin").write_bytes(b"\xff\xfe\x00\x81")
        self.assertIn("not a text file", self.manager.plan_import(str(folder)).errors[0])
        self.assertTrue(self.manager.plan_import(str(self.root / "nowhere")).errors)

    def test_summary_diff_and_full(self):
        change = self.manager.plan_create("greeting-style", "How to greet.", "1. Say hi.",
                                          {"template.md": "Hi"})
        summary = change.summary()
        self.assertIn("CREATE 'greeting-style' (user: alice), new, v1", summary)
        self.assertIn("files: SKILL.md, template.md", summary)
        self.assertIn("format OK · no credentials found", summary)
        self.assertIn("=== template.md ===", change.full())

    def test_folder_mirrors_the_database(self):
        self.create()
        skill_md = self.root / "user-skills" / "alice" / "greeting-style" / "SKILL.md"
        self.assertTrue(skill_md.is_file())
        self.manager.apply(self.manager.plan_remove("greeting-style"))
        self.assertFalse(skill_md.exists())


class ApproverTests(Workspace):
    def approver(self, *answers, interactive=True):
        replies, said = iter(answers), []
        return Approver(self.manager.plan, ask=lambda _: next(replies), say=said.append,
                        interactive=interactive), said

    def request(self, tool_name, **args):
        return {"action_requests": [{"name": tool_name, "args": args}]}

    def test_invalid_change_is_rejected_without_asking(self):
        approver, said = self.approver()
        [decision] = approver.decide(self.request("create_skill", name="Bad_Name",
                                                  description="d", instructions="s"))["decisions"]
        self.assertEqual(decision["type"], "reject")
        self.assertIn("invalid", decision["message"])
        self.assertEqual(said, [])

    def test_one_shot_runs_reject(self):
        approver, _ = self.approver(interactive=False)
        [decision] = approver.decide(self.request("create_skill", name="ok-skill",
                                                  description="d.", instructions="s"))["decisions"]
        self.assertEqual(decision, {"type": "reject", "message": NO_HUMAN})

    def test_diff_then_approve_and_reject_with_reason(self):
        approver, said = self.approver("d", "f", "a")
        request = self.request("create_skill", name="ok-skill", description="d.", instructions="s")
        self.assertEqual(approver.decide(request)["decisions"], [{"type": "approve"}])
        self.assertIn("CREATE 'ok-skill'", said[0])
        self.assertIn("+++ b/SKILL.md", said[1])
        self.assertIn("=== SKILL.md ===", said[2])
        approver, _ = self.approver("r", "too vague")
        [decision] = approver.decide(request)["decisions"]
        self.assertEqual(decision["type"], "reject")
        self.assertIn("too vague", decision["message"])


class MarketplaceTests(Workspace):
    def bob(self):
        return SkillManager(self.store, "bob", self.root)

    def publish(self, manager, name="meeting-notes", instructions="1. Summarise."):
        manager.apply(manager.plan_create(name, "Turns notes into a summary.", instructions))
        return manager.apply(manager.plan_publish(name))

    def test_publish_then_install_by_another_user(self):
        result = self.publish(self.manager)
        self.assertEqual((result["scope"], result["version"]), ("marketplace", 1))
        bob = self.bob()
        [listed] = bob.browse()
        self.assertEqual((listed["name"], listed["installed"]), ("meeting-notes", False))
        self.assertEqual(bob.browse("nothing-like-this"), [])
        change = bob.plan_install("meeting-notes")
        self.assertEqual(change.errors, [])
        self.assertIn("INSTALL 'meeting-notes' (marketplace skill, for user bob), marketplace v1",
                      change.summary())
        bob.apply(change)
        self.assertTrue(bob.browse()[0]["installed"])
        self.assertTrue((bob.installed_folder / "meeting-notes" / "SKILL.md").is_file())
        self.assertEqual(self.store.installed("alice"), [], "installs are per user")

    def test_installed_skills_follow_the_latest_version(self):
        self.publish(self.manager)
        bob = self.bob()
        bob.apply(bob.plan_install("meeting-notes"))
        self.manager.apply(self.manager.plan_update("meeting-notes", instructions="1. Shorter.",
                                                    scope="marketplace"))
        bob.sync_folder()
        self.assertIn("1. Shorter.", (bob.installed_folder / "meeting-notes" / "SKILL.md").read_text())

    def test_marketplace_changes_are_labelled_shared(self):
        change = self.manager.plan_create("shared-one", "Shared.", "Steps", scope="marketplace")
        self.assertIn("(marketplace, shared with everyone), new, v1", change.summary())
        self.manager.apply(change)
        self.assertEqual(self.store.list_current("user", "alice"), [])
        self.assertEqual([s.name for s in self.store.list_current("marketplace", "")], ["shared-one"])

    def test_same_name_in_both_scopes_and_overview(self):
        self.publish(self.manager)
        bob = self.bob()
        bob.apply(bob.plan_install("meeting-notes"))
        bob.apply(bob.plan_create("meeting-notes", "Bob's own.", "1. Bob's way."))
        rows = {(r["name"], r["source"]): r for r in bob.overview()}
        self.assertIn(("meeting-notes", "user"), rows)
        self.assertIn("overridden", rows[("meeting-notes", "marketplace (installed)")]["note"])
        sources = bob.feature().skill_dirs
        self.assertEqual(sources, (bob.installed_rel, bob.folder_rel), "user skills load last and win")

    def test_removed_marketplace_skill_disappears_for_installers(self):
        self.publish(self.manager)
        bob = self.bob()
        bob.apply(bob.plan_install("meeting-notes"))
        self.manager.apply(self.manager.plan_remove("meeting-notes", scope="marketplace"))
        bob.sync_folder()
        self.assertFalse((bob.installed_folder / "meeting-notes").exists())
        [row] = bob.overview()
        self.assertEqual(row["note"], "removed from the marketplace")
        self.assertTrue(bob.apply(bob.plan_uninstall("meeting-notes"))["saved"])

    def test_install_and_publish_errors(self):
        bob = self.bob()
        self.assertTrue(bob.plan_install("missing").errors)
        self.assertTrue(bob.plan_uninstall("missing").errors)
        self.assertTrue(bob.plan_publish("missing").errors)
        self.publish(self.manager)
        bob.apply(bob.plan_install("meeting-notes"))
        self.assertIn("already installed", bob.plan_install("meeting-notes").errors[0])
        self.assertTrue(self.manager.plan("create_skill", {"name": "x", "description": "d",
                                                           "instructions": "s", "scope": "global"}).errors)

    def test_secrets_cannot_be_published(self):
        self.manager.apply(self.manager.plan_create("ok-skill", "Fine.", "1. Fine."))
        self.store.save("user", "alice", "ok-skill",
                        {"SKILL.md": render_skill_md("ok-skill", "Fine.", f"Use {KEY}.")}, "update")
        self.assertTrue(any("credential" in e for e in self.manager.plan_publish("ok-skill").errors))


class Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args):
    return AIMessage(content="", tool_calls=[{"name": name, "id": f"call-{name}", "args": args}])


class EndToEndTests(Workspace):
    def run_turn(self, *responses, answers=()):
        model = Model(responses=[*responses, AIMessage(content="done")])
        agent = build_agent(model, InMemorySaver(),
                            FilesystemBackend(root_dir=str(self.root), virtual_mode=True),
                            [self.manager.feature()])
        replies = iter(answers)
        approver = Approver(self.manager.plan, ask=lambda _: next(replies), say=lambda *_: None)
        config = {"configurable": {"thread_id": "t"}}
        stream_turn(agent, "go", config, approver=approver)
        return agent.get_state(config).values["messages"]

    def test_nothing_is_saved_until_approved(self):
        create = call("create_skill", {"name": "greeting-style", "description": "How to greet.",
                                       "instructions": "1. Say hi."})
        with patch("builtins.print"):
            messages = self.run_turn(create, answers=["r", "not now"])
        self.assertIsNone(self.store.get("user", "alice", "greeting-style"))
        self.assertTrue(any("not now" in str(m.content) for m in messages if m.type == "tool"))
        with patch("builtins.print"):
            self.run_turn(create, answers=["a"])
        self.assertEqual(self.store.get("user", "alice", "greeting-style").version, 1)

    def test_file_tools_cannot_write_edit_or_delete_skill_folders(self):
        seeded = self.root / "skills" / "seeded"
        seeded.mkdir()
        (seeded / "SKILL.md").write_text(render_skill_md("seeded", "Seeded.", "Original"))
        self.create("mine", instructions="Original")
        mine = self.manager.folder / "mine" / "SKILL.md"
        calls = []
        for path in ("/skills/seeded/SKILL.md", f"/{self.manager.folder_rel}mine/SKILL.md"):
            calls += [call("edit_file", {"file_path": path, "old_string": "Original", "new_string": "X"}),
                      call("delete", {"file_path": path})]
        for target in ("/skills/sneaky/SKILL.md", f"/{self.manager.folder_rel}sneaky/SKILL.md"):
            calls.append(call("write_file", {"file_path": target, "content": "sneaky"}))
        with patch("builtins.print"):
            messages = self.run_turn(*calls)
        results = [m for m in messages if m.type == "tool"]
        self.assertEqual(len(results), 6)
        self.assertTrue(all("permission denied" in str(m.content).lower() for m in results),
                        [m.content for m in results])
        self.assertIn("Original", (seeded / "SKILL.md").read_text())
        self.assertIn("Original", mine.read_text())
        self.assertFalse((self.root / "skills" / "sneaky").exists())

    def test_approval_works_through_the_general_purpose_subagent(self):
        delegate = call("task", {"description": "create the skill", "subagent_type": "general-purpose"})
        create = call("create_skill", {"name": "via-sub", "description": "Made by a subagent.",
                                       "instructions": "1. Hi."})
        for answers, expected in ((["a"], [1]), (["r", ""], [])):
            with self.subTest(answers=answers), patch("builtins.print"):
                self.store.db.execute("DELETE FROM skill_versions")
                self.store.db.execute("DELETE FROM skills")
                self.run_turn(delegate, create, AIMessage(content="sub done"), answers=answers)
                versions = [v.version for v in self.store.history("user", "alice", "via-sub")]
                self.assertEqual(versions, expected)

    def test_skill_tools_need_approval_everywhere(self):
        import deep_agents_demo.agent as agent_module
        with patch.object(agent_module, "create_deep_agent") as create:
            agent_module.build_agent(None, None, object(), [self.manager.feature()])
        kwargs = create.call_args.kwargs
        self.assertEqual(sorted(kwargs["interrupt_on"]), sorted(GATED_TOOLS))
        self.assertNotIn("skill_history", kwargs["interrupt_on"])
        # Deep Agents passes interrupt_on to every subagent unless one overrides it.
        self.assertTrue(all("interrupt_on" not in s for s in kwargs["subagents"]))


if __name__ == "__main__":
    unittest.main()
