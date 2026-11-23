from dentex_bench.augment.locator import LocatedTooth, AnnotationToothLocator, ToothLocator


def test_annotation_locator_returns_teeth():
    anns = [
        {
            "id": 1,
            "image_id": 100,
            "category_id_1": 1,
            "category_id_2": 6,  # FDI 16
            "category_id": 1,
            "bbox": [100.0, 200.0, 50.0, 70.0],
            "segmentation": [[100, 200, 150, 200, 150, 270, 100, 270]],
        }
    ]
    locator = AnnotationToothLocator(anns)
    teeth = locator.teeth(100)

    assert len(teeth) == 1
    t = teeth[0]
    assert t.fdi == 16
    assert t.bbox == [100.0, 200.0, 50.0, 70.0]
    assert t.source == "annotations"
    assert t.score == 1.0


def test_annotation_locator_direct_fdi():
    anns = [
        {
            "id": 2,
            "image_id": 101,
            "fdi": 25,
            "bbox": [50.0, 60.0, 30.0, 40.0],
        }
    ]
    locator = AnnotationToothLocator(anns)
    teeth = locator.teeth(101)
    assert len(teeth) == 1
    assert teeth[0].fdi == 25
    assert teeth[0].segmentation is None


def test_annotation_locator_zero_indexed():
    anns = [
        {
            "id": 10,
            "image_id": 102,
            "category_id_1": 0,  # 0-indexed -> Q1
            "category_id_2": 0,  # 0-indexed -> tooth 1
            "bbox": [10.0, 20.0, 30.0, 40.0],
        },
        {
            "id": 11,
            "image_id": 102,
            "category_id_1": 3,  # 0-indexed -> Q4
            "category_id_2": 7,  # 0-indexed -> tooth 8
            "bbox": [100.0, 120.0, 30.0, 40.0],
        },
    ]
    locator = AnnotationToothLocator(anns)
    teeth = locator.teeth(102)
    assert len(teeth) == 2
    assert teeth[0].fdi == 11
    assert teeth[1].fdi == 48


def test_annotation_locator_unknown_image():
    locator = AnnotationToothLocator([])
    assert locator.teeth(999) == []


def test_annotation_locator_satisfies_protocol():
    anns = [{"image_id": 1, "fdi": 11, "bbox": [1, 2, 3, 4]}]
    locator = AnnotationToothLocator(anns)
    assert isinstance(locator, ToothLocator)
