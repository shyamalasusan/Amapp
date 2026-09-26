# PART B — DIOR SETUP (classes, sensitivity table)
# =============================================================================

DIOR_CLASSES = [
    "airplane", "airport", "baseballfield", "basketballcourt", "bridge",
    "chimney", "dam", "expresswayservicearea", "expresswaytollstation",
    "golffield", "groundtrackfield", "harbor", "overpass", "ship",
    "stadium", "storagetank", "tenniscourt", "trainstation",
    "vehicle", "windmill"
]

SENSITIVITY_TABLE = {
    "airport": 1.0, "dam": 0.9, "harbor": 0.8, "trainstation": 0.7,
    "expresswaytollstation": 0.6, "storagetank": 0.6, "chimney": 0.5,
    "expresswayservicearea": 0.5, "overpass": 0.4, "bridge": 0.4,
    "stadium": 0.3, "groundtrackfield": 0.3, "baseballfield": 0.2,
    "basketballcourt": 0.2, "tenniscourt": 0.2, "golffield": 0.2,
    "windmill": 0.1, "ship": 0.3, "vehicle": 0.2, "airplane": 0.4,
    "plane": 0.4, "large-vehicle": 0.2, "small-vehicle": 0.2,
    "helicopter": 0.5, "roundabout": 0.2, "soccer-ball-field": 0.2,
    "swimming-pool": 0.1, "baseball-diamond": 0.2, "tennis-court": 0.2,
    "basketball-court": 0.2, "ground-track-field": 0.3,
}


# PART H — SUB-REGION SETUP (classes, per-part sensitivity)
# =============================================================================

SUBREGION_CLASSES = {
    "airport", "harbor", "dam", "trainstation",
    "storagetank", "expresswaytollstation",
}

SENSITIVITY_TABLE_PARTS = {
    "airport": {
        "runway": 1.0,
        "terminal": 0.8,
        "parking": 0.5,
        "vegetation": 0.1,
        "other": 0.3,
    },
    "harbor": {
        "docking_area": 0.8,
        "vessel_cluster": 0.6,
        "shoreline": 0.3,
        "water": 0.1,
        "other": 0.3,
    },
    "dam": {
        "spillway_structure": 0.9,
        "reservoir_edge": 0.4,
        "water": 0.1,
        "other": 0.3,
    },
    "trainstation": {
        "platform_structure": 0.7,
        "rail_yard": 0.6,
        "surrounding_lot": 0.3,
        "other": 0.3,
    },
    "storagetank": {
        "tank_structure": 0.6,
        "containment_area": 0.4,
        "other": 0.3,
    },
    "expresswaytollstation": {
        "toll_structure": 0.6,
        "roadway": 0.3,
        "other": 0.3,
    },
}

# DOTA v1.0 class names, in the standard order used by the YOLOv5/ultralytics
# DOTA release (matches the class-id order models trained on that dataset
# will output).
DOTA_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool",
]

