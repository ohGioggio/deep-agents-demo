"""The decorator registry for swappable providers, and the sandbox providers in it."""
import unittest

from deep_agents_demo.providers.registry import ProviderRegistry


class RegistryTests(unittest.TestCase):
    def test_registry(self):
        registry = ProviderRegistry("thing")

        @registry.register("first")
        def first():
            return 1

        @registry.register("second")
        def second():
            return 2

        self.assertIs(first, registry.get("first"))  # the decorator returns the function
        self.assertEqual(registry.names(), ("first", "second"))
        self.assertEqual(registry.get("second")(), 2)
        with self.assertRaises(KeyError) as caught:
            registry.get("nope")
        self.assertIn("first, second", str(caught.exception))
        with self.assertRaises(ValueError):
            registry.register("first")(second)

    def test_sandbox_names_match_config(self):
        from deep_agents_demo.config import SANDBOX_PROVIDERS
        from deep_agents_demo.features.sandbox.providers import SANDBOXES
        self.assertEqual(set(SANDBOXES.names()), set(SANDBOX_PROVIDERS) - {"none"})


if __name__ == "__main__":
    unittest.main()
