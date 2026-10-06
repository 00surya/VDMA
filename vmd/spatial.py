"""Compare tracked people within one inverse-depth frame; never metric distance."""
from itertools import combinations
from math import hypot

import numpy as np

from .heuristics import Rules, body_contact_regions, guard_contact


def box_overlap(first, second):
    """Fraction of the smaller box covered by the intersection."""
    areas = [(box[2]-box[0]) * (box[3]-box[1])
             if box[2] > box[0] and box[3] > box[1] else 0 for box in (first, second)]
    if not all(np.isfinite(box).all() for box in (first, second)) or min(areas) <= 0:
        return 0.0
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height / min(areas)


def fight_depth_evidence(depth, people):
    """Conservative same-frame depth support, not a physical contact measurement."""
    if (depth.ndim != 2 or not depth.size or not np.isfinite(depth).all()
            or float(np.ptp(depth)) <= 1e-6):
        return []
    readings = {person.track_id: torso_depth(depth, person) for person in people}
    observations = []
    for a, b in combinations(people, 2):
        if min(a.track_id, b.track_id) < 0 or a.track_id == b.track_id:
            continue
        first, second = readings[a.track_id], readings[b.track_id]
        status, gap = "uncertain", None
        # Relative Z can support an outstretched grab even when torso boxes do
        # not overlap. Event rules separately require their own contact geometry.
        if first is not None and second is not None:
            gap = abs(first[0] - second[0])
            # Interior MAD includes actual torso surface slope; adding both
            # spreads twice treated that shape as fourfold measurement error.
            # Retain a noise cap and a margin from the less stable torso.
            uncertainty = max(first[1], second[1])
            # Similar relative Z only supports an interaction hypothesis. A noisy
            # torso or a flat map must never be interpreted as matching depth.
            if uncertainty <= .04 and gap + uncertainty <= .12:
                status = "compatible"
            elif gap >= max(.18, 3 * (first[1] + second[1])):
                status = "separated"
        contacts = strike_depth_evidence(depth, a, b, readings)
        gross_separation = gap is not None and gap > .30
        if gross_separation:
            # A matching foreground pixel cannot bridge arbitrary torso depth.
            for contact in contacts:
                contact['status'] = 'separated'
        fight_status = ('separated' if gross_separation else 'compatible'
                        if status == 'compatible' or any(contact['geometry_match']
                            and contact['status'] == 'compatible' for contact in contacts)
                        else 'uncertain')
        observations.append({"tracks": list(sorted((a.track_id, b.track_id))),
                             "status": status, "relative_gap": gap,
                             "fight_status": fight_status, "contacts": contacts})
    return observations


def limb_depth(depth, person, wrist):
    """Sample a visible wrist and its own distal forearm in the original frame."""
    if person.track_id < 0 or not person.pose_reliable or wrist not in (9, 10):
        return None
    hand, elbow = (person.keypoints[index] for index in (wrist, wrist-2))
    if (min(hand[2], elbow[2]) < .55
            or not np.isfinite([*hand, *elbow]).all()):
        return None
    height, width = depth.shape
    if any(not (0 <= point[0] < width and 0 <= point[1] < height) for point in (hand, elbow)):
        return None
    values, patch_spreads = [], []
    for fraction in (0, .2, .35):
        x, y = (int(round(hand[axis]*(1-fraction)+elbow[axis]*fraction)) for axis in (0, 1))
        # Hands are narrow and the model map has been enlarged to frame size.
        # Larger torso-sized windows blend hand and background at a silhouette.
        patch = depth[max(0, y-1):min(height, y+2), max(0, x-1):min(width, x+2)]
        if not patch.size or not np.isfinite(patch).all():
            return None
        value = float(np.median(patch))
        values.append(value)
        patch_spreads.append(float(np.median(np.abs(patch-value))))
    # An arm reaching across Z can have a smooth real depth gradient. Its
    # longitudinal variation is not measurement noise at the contact surface.
    # An abrupt foreground/background jump, however, cannot establish that
    # the wrist pixel belongs to this arm.
    if any(abs(first-second) > .10 for first, second in zip(values, values[1:])):
        return None
    return values[0], patch_spreads[0]


def body_surface_depth(depth, person, contact_point, torso):
    """Use the reached head surface for an above-shoulder strike."""
    shoulders = [person.keypoints[index] for index in (5, 6)]
    if min(point[2] for point in shoulders) < .55 or contact_point[1] >= sum(point[1] for point in shoulders)/2:
        return torso
    height, width = depth.shape
    face = [person.keypoints[index] for index in range(5)
            if person.keypoints[index][2] >= .55
            and 0 <= person.keypoints[index][0] < width and 0 <= person.keypoints[index][1] < height]
    if len(face) < 3 or not person.pose_reliable:
        return None
    x, y = (int(round(sum(point[axis] for point in face)/len(face))) for axis in (0, 1))
    patch = depth[max(0, y-2):min(height, y+3), max(0, x-2):min(width, x+3)]
    if not patch.size or not np.isfinite(patch).all():
        return None
    middle = float(np.median(patch))
    return middle, float(np.median(np.abs(patch-middle)))


def strike_depth_evidence(depth, a, b, torso_readings):
    """Keep each attacking arm and reached body/guard surface identifiable.

    Comparisons exist independently of raw contact geometry because stabilized
    motion can land just outside a raw landmark's boundary. Only geometrically
    eligible comparisons can support the pair status; the temporal rule must
    still match an accepted strike's actor, wrist, kind and target exactly.
    """
    rules = Rules()
    limbs = {(person.track_id, wrist): limb_depth(depth, person, wrist)
             for person in (a, b) for wrist in (9, 10)}
    contacts = []
    for actor, target in ((a, b), (b, a)):
        shoulders = [target.keypoints[index] for index in (5, 6)]
        hips = [target.keypoints[index] for index in (11, 12) if target.keypoints[index][2] >= .55]
        shoulder_width = hypot(shoulders[0][0]-shoulders[1][0], shoulders[0][1]-shoulders[1][1])
        torso_length = hypot(*[sum(point[axis] for point in shoulders)/2
                               - sum(point[axis] for point in hips)/len(hips)
                               for axis in (0, 1)]) if hips else 0
        allowance = min(.08*min(actor.scale, target.scale),
                        max(.5*shoulder_width, .2*torso_length))
        regions = body_contact_regions(target, rules)
        for wrist in (9, 10):
            source = limbs[(actor.track_id, wrist)]
            point = actor.keypoints[wrist]
            body_geometry = bool(source and hips and point[1] <= max(hip[1] for hip in hips) and any(
                region[0]-allowance <= point[0] <= region[2]+allowance
                and region[1]-allowance <= point[1] <= region[3]+allowance for region in regions))
            guard = guard_contact(actor, target, wrist, rules) if source else None
            targets = [('body', None, body_surface_depth(depth, target, point,
                        torso_readings[target.track_id]), body_geometry)]
            targets.extend(('guard', joint, limbs[(target.track_id, joint)],
                            bool(guard and guard['target_joint'] == joint)) for joint in (9, 10))
            for kind, joint, destination, geometry in targets:
                status, gap = 'uncertain', None
                if source is not None and destination is not None:
                    gap = abs(source[0]-destination[0])
                    uncertainty = max(source[1], destination[1])
                    if uncertainty <= .035 and gap+uncertainty <= .05:
                        status = 'compatible'
                    elif gap >= .10:
                        status = 'separated'
                contacts.append({'actor': actor.track_id, 'wrist': wrist,
                    'target_track': target.track_id, 'contact_kind': kind,
                    'target_joint': joint, 'status': status, 'relative_gap': gap,
                    'geometry_match': geometry})
    return contacts


def torso_depth(depth, person):
    """Sample the torso interior, away from noisy clothing/background edges."""
    if person.track_id < 0 or not person.pose_reliable:
        return None
    height, width = depth.shape
    joints = [person.keypoints[index] for index in (5, 6, 11, 12)
              if person.keypoints[index][2] >= .55
              and 0 <= person.keypoints[index][0] < width and 0 <= person.keypoints[index][1] < height]
    if len(joints) < 3:
        return None
    centre = np.mean(np.asarray(joints)[:, :2], axis=0)
    radius = max(2, min(6, int(person.height * .015)))
    values = []
    for x, y, _ in joints:
        x, y = .45*np.asarray((x, y)) + .55*centre
        cx, cy = int(x), int(y)
        patch = depth[max(0, cy-radius):min(height, cy+radius+1),
                      max(0, cx-radius):min(width, cx+radius+1)]
        finite = patch[np.isfinite(patch)]
        if finite.size:
            values.append(float(np.median(finite)))
    if len(values) < 3:
        return None
    middle = float(np.median(values))
    spread = float(np.median(np.abs(np.asarray(values)-middle)))
    if spread > .12:
        return None
    return middle, spread
