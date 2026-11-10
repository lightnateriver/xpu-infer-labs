"""No-model tests for pairing, error statistics, and strict regression checks."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path=Path(__file__).resolve().parents[1]/'scripts/compare_scores.py'
spec=importlib.util.spec_from_file_location('compare_scores',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def records(values):
    return {str(i):dict(clip_id=str(i),scores={k:v for k in module.SCORES},input_sha256=str(i)) for i,v in enumerate(values)}

class CompareTests(unittest.TestCase):
    def test_known_error(self):
        result=module.compare(records([0,1,2]),records([1,2,3]),True)
        self.assertEqual(result['global_metrics']['mse'],1)
        self.assertEqual(result['global_metrics']['mae'],1)
        self.assertEqual(result['global_metrics']['rmse'],1)
        self.assertEqual(result['global_metrics']['bias_test_minus_reference'],1)
    def test_exact(self):
        result=module.compare(records([0,1,2]),records([0,1,2]),True)
        self.assertEqual(result['global_metrics']['mse'],0)
        self.assertEqual(result['clips_all_scores_exact'],3)
    def test_order_independent(self):
        ref=records([0,1,2]);test=dict(reversed(list(ref.items())))
        self.assertEqual(module.compare(ref,test)['global_metrics']['mse'],0)
    def test_missing_id(self):
        with self.assertRaises(ValueError):module.compare(records([0,1]),records([0]))
    def test_pixel_change(self):
        ref=records([0]);test=records([0]);test['0']['input_sha256']='changed'
        with self.assertRaises(ValueError):module.compare(ref,test,True)
    def test_constant_statistics(self):
        result=module.statistics([0,0],[1,1])
        self.assertEqual(result['mse'],1)
        self.assertIsNone(result['pearson_r'])
        self.assertIsNone(result['relative_l2'])
    def test_p95(self):
        self.assertAlmostEqual(module.statistics([0,0],[0,2])['p95_absolute_error'],1.9)
    def test_invalid_records(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'scores.json'
            for rows in [[],[records([1])['0']]*2,[dict(clip_id='x',scores={k:float('nan') for k in module.SCORES})]]:
                path.write_text(json.dumps(rows))
                with self.assertRaises(ValueError):module.read_scores(path)

if __name__=='__main__':unittest.main()
