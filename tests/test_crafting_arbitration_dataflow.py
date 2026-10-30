import unittest
from src.scripting.node_environment import (
    AutomaticCraftingArbitrationNode, AIReasonerDataflowNode, OreDictionary
)


class TestCraftingArbitrationDataflow(unittest.TestCase):
    def test_01_crafting_node_evaluation_and_admissibility(self):
        node = AutomaticCraftingArbitrationNode()
        # Initial evaluation under healthy parameters
        res = node.evaluate()
        self.assertEqual(res["out_admissible"], 1)
        self.assertAlmostEqual(res["out_pas_phase"], 1.0)
        self.assertEqual(res["out_material_id"], 20)  # ingotIron maps to 20 in defaults
        self.assertEqual(res["out_craft_status"], "APPROVED")

    def test_02_crafting_node_thermal_runaway_veto(self):
        node = AutomaticCraftingArbitrationNode()
        node.inputs["in_waste_heat"].value = 950.0  # Above 800 thermal runaway limit
        res = node.evaluate()
        self.assertEqual(res["out_admissible"], 0)
        self.assertEqual(res["out_craft_status"], "THERMAL_VETO")

    def test_03_crafting_node_linked_to_ai_reasoner(self):
        craft_node = AutomaticCraftingArbitrationNode()
        ai_node = AIReasonerDataflowNode()

        # Connect craft output phase to AI reasoner energy input
        craft_node.outputs["out_pas_phase"].connect(ai_node.inputs["in_energy"])
        
        # Evaluate upstream craft node
        craft_res = craft_node.evaluate()
        self.assertEqual(craft_res["out_admissible"], 1)
        
        # Evaluate downstream AI reasoner node
        ai_res = ai_node.evaluate()
        self.assertIn("out_pas_h", ai_res)
        self.assertIn("out_regime_code", ai_res)
        self.assertGreater(ai_res["out_pas_h"], 0.2727)  # Verified >= 3/11 knot-theoretic threshold


if __name__ == "__main__":
    unittest.main()

