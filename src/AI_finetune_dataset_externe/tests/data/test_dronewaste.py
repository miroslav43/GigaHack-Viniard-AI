import pytest

from fte.data.dronewaste import category_role, scale_annotations, select_images


def _coco():
    images = [{"id": i, "file_name": f"site{1 if i < 5 else 6}_{i}.png", "site": "site1" if i < 5 else "site6",
               "width": 640, "height": 640} for i in range(1, 9)]
    anns = [
        {"image_id": 1, "category_id": 9, "bbox": [10, 10, 20, 20], "segmentation": [[10, 10, 30, 10, 30, 30]]},
        {"image_id": 2, "category_id": 1, "bbox": [0, 0, 100, 100], "segmentation": []},
        {"image_id": 3, "category_id": 11, "bbox": [0, 0, 50, 50], "segmentation": []},
        {"image_id": 4, "category_id": 12, "bbox": [0, 0, 50, 50], "segmentation": []},
        {"image_id": 6, "category_id": 15, "bbox": [100, 100, 40, 40], "segmentation": [[100, 100, 140, 140, 100, 140]]},
    ]
    return {"images": images, "annotations": anns, "categories": []}


def test_category_roles():
    assert category_role(9) == "waste" and category_role(11) == "waste"
    assert category_role(12) == "ignore" and category_role(1) == "background"
    with pytest.raises(ValueError):
        category_role(99)


def test_select_images_roles():
    sel = select_images(_coco(), n_empty=2, pallet_cap=1)
    assert sel.roles["site1_1.png"] == "pos"
    assert sel.roles["site1_2.png"] == "neg_pile"
    assert sel.roles["site1_3.png"] == "pos"  # pallet-only, within cap
    assert "site1_4.png" not in sel.roles  # scrap-only image is dropped
    assert sel.counts()["neg_empty"] == 2


def test_scale_annotations_uses_site_gsd():
    coco = _coco()
    sel = select_images(coco, n_empty=0)
    out = scale_annotations(coco, sel)
    by_id = {i["id"]: i for i in out["images"]}
    assert by_id[1]["width"] == 512  # 2 cm -> 2.5 cm
    assert by_id[6]["width"] == 717  # site6: 2.8 cm -> 2.5 cm
    ann6 = next(a for a in out["annotations"] if a["image_id"] == 6)
    assert ann6["bbox"] == [112.0, 112.0, 44.8, 44.8]
    assert ann6["role"] == "waste"
    assert all(i["file_name"].endswith(".jpg") for i in out["images"])
