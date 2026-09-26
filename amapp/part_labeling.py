
import numpy as np

# PART J — PART LABELING: AIRPORT + GENERIC FALLBACK
# =============================================================================

def _mask_bbox_and_aspect(mask):
    # Return (width, height, aspect_ratio, fill_ratio) of a mask's
    # fill_ratio = mask area / bounding-box area (0-1) -- how much of its own
    # a runway) has HIGH fill_ratio (it densely fills a thin, long box). A
    # elongated airfield) has a low fill_ratio, since most of its own
    # FINDING 5 FIX: the original version of this function returned only
    # against real DIOR airport detections) to frequently mislabel the
    # the mask itself is actually a thin strip. Adding fill_ratio lets
    # fill_ratio) from "large blob that happens to sit in an elongated
    # outer box" (low fill_ratio) -- see label_parts_airport for how this is
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, 1.0, 0.0
    w = xs.max() - xs.min() + 1
    h = ys.max() - ys.min() + 1
    long_side, short_side = max(w, h), max(min(w, h), 1)
    aspect_ratio = long_side / short_side
    bbox_area = max(w * h, 1)
    fill_ratio = mask.sum() / bbox_area
    return w, h, aspect_ratio, fill_ratio

def label_parts_airport(masks, bbox):
    # mask (aspect ratio well above 1, AND high fill_ratio -- a thin strip
    # FINDING 5 FIX: previously used aspect_ratio alone, which was found (via
    # visual spot-check) to frequently mislabel the actual runway as
    # strip. Now requires BOTH high aspect ratio AND high fill_ratio for the
    labeled = []
    if not masks:
        return labeled

    total_box_area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1)
    remaining = list(masks)

    # Runway: elongated (aspect >= 2.5) AND densely-filled (fill_ratio >=
    # box will have a much lower fill_ratio) and reasonably large (>3% of
    runway_candidates = [
        m for m in remaining
        if _mask_bbox_and_aspect(m[0])[2] >= 2.5
        and _mask_bbox_and_aspect(m[0])[3] >= 0.35
        and m[1] / total_box_area > 0.03
    ]
    if runway_candidates:
        # fill_ratio (most strip-like), not just the largest by area --
        # area alone was part of what caused Finding 5's mislabeling, since
        runway_mask = max(runway_candidates, key=lambda t: _mask_bbox_and_aspect(t[0])[3])
        labeled.append(("runway", runway_mask[0]))
        remaining = [m for m in remaining if m is not runway_mask]

    terminal_candidates = [
        m for m in remaining
        if not (_mask_bbox_and_aspect(m[0])[2] >= 2.5 and _mask_bbox_and_aspect(m[0])[3] >= 0.35)
        and m[1] / total_box_area > 0.03
    ]
    if terminal_candidates:
        terminal_mask = max(terminal_candidates, key=lambda t: t[1])
        labeled.append(("terminal", terminal_mask[0]))
        remaining = [m for m in remaining if m is not terminal_mask]

    parking_candidates = [m for m in remaining if m[1] / total_box_area > 0.01]
    if parking_candidates:
        parking_mask = max(parking_candidates, key=lambda t: t[1])
        labeled.append(("parking", parking_mask[0]))
        remaining = [m for m in remaining if m is not parking_mask]

    for mask, area in remaining:
        labeled.append(("vegetation", mask))

    return labeled

def label_parts_generic(masks, bbox, class_name):
    if not masks:
        return []
    part_names = list(SENSITIVITY_TABLE_PARTS.get(class_name, {}).keys())
    primary_name = part_names[0] if part_names else "other"

    labeled = [(primary_name, masks[0][0])]
    for mask, area in masks[1:]:
        labeled.append(("other", mask))
    return labeled

def label_parts(masks, bbox, class_name):
    dispatch = {
        "airport": label_parts_airport,
        "harbor": label_parts_harbor,
        "dam": label_parts_dam,
        "trainstation": label_parts_trainstation,
        "storagetank": label_parts_storagetank,
        "expresswaytollstation": label_parts_expresswaytollstation,
    }
    fn = dispatch.get(class_name)
    if fn is None:
        return label_parts_generic(masks, bbox, class_name)
    return fn(masks, bbox)



# PART M — PART LABELING: HARBOR/DAM/TRAINSTATION/STORAGETANK/TOLLSTATION
# =============================================================================

def label_parts_harbor(masks, bbox):
    labeled = []
    if not masks:
        return labeled

    total_box_area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1)
    remaining = list(masks)

    docking_candidates = [
        m for m in remaining
        if 1.8 <= _mask_bbox_and_aspect(m[0])[2] <= 4.0 and m[1] / total_box_area > 0.04
    ]
    if docking_candidates:
        docking_mask = max(docking_candidates, key=lambda t: t[1])
        labeled.append(("docking_area", docking_mask[0]))
        remaining = [m for m in remaining if m is not docking_mask]

    vessel_candidates = [
        m for m in remaining
        if _mask_bbox_and_aspect(m[0])[2] < 2.5 and 0.005 < m[1] / total_box_area < 0.05
    ]
    if vessel_candidates:
        combined_mask = np.zeros_like(vessel_candidates[0][0])
        for m, _ in vessel_candidates:
            combined_mask |= m
        labeled.append(("vessel_cluster", combined_mask))
        vessel_candidate_ids = {id(m) for m, _ in vessel_candidates}
        remaining = [m for m in remaining if id(m[0]) not in vessel_candidate_ids]

    if remaining:
        water_mask = max(remaining, key=lambda t: t[1])
        labeled.append(("water", water_mask[0]))
        remaining = [m for m in remaining if m is not water_mask]

    for mask, area in remaining:
        labeled.append(("shoreline", mask))

    return labeled

def label_parts_dam(masks, bbox):
    labeled = []
    if not masks:
        return labeled

    total_box_area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1)
    remaining = list(masks)

    structure_candidates = [
        m for m in remaining
        if _mask_bbox_and_aspect(m[0])[2] < 2.0 and m[1] / total_box_area > 0.10
    ]
    if structure_candidates:
        structure_mask = max(structure_candidates, key=lambda t: t[1])
        labeled.append(("spillway_structure", structure_mask[0]))
        remaining = [m for m in remaining if m is not structure_mask]

    edge_candidates = [m for m in remaining if _mask_bbox_and_aspect(m[0])[2] >= 2.0]
    if edge_candidates:
        edge_mask = max(edge_candidates, key=lambda t: t[1])
        labeled.append(("reservoir_edge", edge_mask[0]))
        remaining = [m for m in remaining if m is not edge_mask]

    if remaining:
        water_mask = max(remaining, key=lambda t: t[1])
        labeled.append(("water", water_mask[0]))
        remaining = [m for m in remaining if m is not water_mask]

    for mask, area in remaining:
        labeled.append(("other", mask))

    return labeled

def label_parts_trainstation(masks, bbox):
    labeled = []
    if not masks:
        return labeled

    total_box_area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1)
    remaining = list(masks)

    platform_candidates = [
        m for m in remaining
        if _mask_bbox_and_aspect(m[0])[2] < 2.2 and m[1] / total_box_area > 0.05
    ]
    if platform_candidates:
        platform_mask = max(platform_candidates, key=lambda t: t[1])
        labeled.append(("platform_structure", platform_mask[0]))
        remaining = [m for m in remaining if m is not platform_mask]

    rail_candidates = [
        m for m in remaining
        if _mask_bbox_and_aspect(m[0])[2] >= 2.2 and m[1] / total_box_area > 0.03
    ]
    if rail_candidates:
        rail_mask = max(rail_candidates, key=lambda t: t[1])
        labeled.append(("rail_yard", rail_mask[0]))
        remaining = [m for m in remaining if m is not rail_mask]

    lot_candidates = [m for m in remaining if m[1] / total_box_area > 0.01]
    if lot_candidates:
        lot_mask = max(lot_candidates, key=lambda t: t[1])
        labeled.append(("surrounding_lot", lot_mask[0]))
        remaining = [m for m in remaining if m is not lot_mask]

    for mask, area in remaining:
        labeled.append(("other", mask))

    return labeled

def label_parts_storagetank(masks, bbox):
    labeled = []
    if not masks:
        return labeled

    total_box_area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1)
    remaining = list(masks)

    tank_candidates = [
        m for m in remaining
        if _mask_bbox_and_aspect(m[0])[2] < 1.3 and m[1] / total_box_area > 0.02
    ]
    if tank_candidates:
        combined_mask = np.zeros_like(tank_candidates[0][0])
        for m, _ in tank_candidates:
            combined_mask |= m
        labeled.append(("tank_structure", combined_mask))
        tank_candidate_ids = {id(m) for m, _ in tank_candidates}
        remaining = [m for m in remaining if id(m[0]) not in tank_candidate_ids]

    if remaining:
        containment_mask = max(remaining, key=lambda t: t[1])
        labeled.append(("containment_area", containment_mask[0]))
        remaining = [m for m in remaining if m is not containment_mask]

    for mask, area in remaining:
        labeled.append(("other", mask))

    return labeled

def label_parts_expresswaytollstation(masks, bbox):
    labeled = []
    if not masks:
        return labeled

    total_box_area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1)
    remaining = list(masks)

    structure_candidates = [
        m for m in remaining
        if _mask_bbox_and_aspect(m[0])[2] < 2.0 and m[1] / total_box_area > 0.03
    ]
    if structure_candidates:
        structure_mask = max(structure_candidates, key=lambda t: t[1])
        labeled.append(("toll_structure", structure_mask[0]))
        remaining = [m for m in remaining if m is not structure_mask]

    roadway_candidates = [m for m in remaining if _mask_bbox_and_aspect(m[0])[2] >= 2.0]
    if roadway_candidates:
        roadway_mask = max(roadway_candidates, key=lambda t: t[1])
        labeled.append(("roadway", roadway_mask[0]))
        remaining = [m for m in remaining if m is not roadway_mask]

    for mask, area in remaining:
        labeled.append(("other", mask))

    return labeled


# =============================================================================
# PART M2 — LABEL DISPATCHER (overrides Part J's dispatcher for all 6 classes)
# =============================================================================

def label_parts(masks, bbox, class_name):
    dispatch = {
        "airport": label_parts_airport,
        "harbor": label_parts_harbor,
        "dam": label_parts_dam,
        "trainstation": label_parts_trainstation,
        "storagetank": label_parts_storagetank,
        "expresswaytollstation": label_parts_expresswaytollstation,
    }
    fn = dispatch.get(class_name)
    if fn is None:
        return label_parts_generic(masks, bbox, class_name)
    return fn(masks, bbox)


