import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.download_abo_chairs import load_chairs, write_metadata


class ABOChairDownloadTests(unittest.TestCase):
    def test_loads_all_csv_rows_as_chairs_and_writes_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "chairs.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output, fieldnames=["3dmodel_id", "path", "faces"]
                )
                writer.writeheader()
                writer.writerow({
                    "3dmodel_id": "chair-a",
                    "path": "a/model.glb",
                    "faces": "200",
                })
                writer.writerow({
                    "3dmodel_id": "chair-b",
                    "path": "/b/model.glb",
                    "faces": "300",
                })

            products = load_chairs(
                csv_path,
                root / "models",
                "https://example.test",
                "3dmodels/original",
            )
            metadata_path = root / "chairs.json"
            write_metadata(metadata_path, csv_path, products)
            metadata = json.loads(metadata_path.read_text())

            self.assertEqual(set(products), {"chair-a", "chair-b"})
            self.assertEqual(
                products["chair-b"]["download_url"],
                "https://example.test/3dmodels/original/b/model.glb",
            )
            self.assertEqual(metadata["summary"]["num_products"], 2)
            self.assertEqual(
                metadata["summary"]["product_type_counts"], {"CHAIR": 2}
            )


if __name__ == "__main__":
    unittest.main()
