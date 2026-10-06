"""Track-local pose filtering and image-supported limb velocity."""
from dataclasses import replace
from math import exp, hypot


class PoseStabilizer:
    def __init__(self, confidence=.55, deadband=.025, max_gap=.65):
        self.confidence, self.deadband, self.max_gap = confidence, deadband, max_gap
        self.tracks = {}

    def update(self, people, timestamp, evidence=None):
        output, active = [], set()
        for person in people:
            active.add(person.track_id)
            old = self.tracks.get(person.track_id)
            dt = timestamp-old['time'] if old else 0
            reset = not old or not 0 < dt <= self.max_gap
            if old and (hypot(person.center[0]-old['center'][0], person.center[1]-old['center'][1]) > person.height or not .5 < person.height/old['height'] < 2):
                reset = True
            torso = [person.keypoints[i] for i in (5,6,11,12)]
            # Briefly occluded hips are common during close interaction. Two shoulders
            # plus one visible hip can anchor a pose; a changed anchor resets velocity.
            visible = tuple(i for i,p in enumerate(torso) if p[2]>=self.confidence)
            reliable = 0 in visible and 1 in visible and len(visible)>=3
            shoulder = ((torso[0][0]+torso[1][0])/2,(torso[0][1]+torso[1][1])/2)
            hips = [torso[i] for i in (2,3) if i in visible]
            hip = tuple(sum(p[axis] for p in hips)/len(hips) for axis in (0,1)) if hips else person.center
            anchor = tuple((shoulder[i]+hip[i])/2 for i in (0,1)) if reliable else person.center
            if old and visible != old['visible']:
                reset = True
            scale = max(person.height*.5, hypot(shoulder[0]-hip[0],shoulder[1]-hip[1])*2.5)
            if not reset:
                scale = old['scale']*.8 + scale*.2
            draw_anchor = anchor if reset else tuple(old['anchor'][i]*.6+anchor[i]*.4 for i in (0,1))
            normalized, points, speeds, counts, limb_motion = [], [], {}, [], {}
            for index, (x,y,confidence) in enumerate(person.keypoints):
                raw = ((x-anchor[0])/scale, (y-anchor[1])/scale)
                prev = None if reset else old['points'][index]
                valid = reliable and confidence >= self.confidence and x > 0 and y > 0
                support = evidence(person,index) if evidence else 0.0
                count = (old['counts'][index]+1) if valid and not reset else int(valid)
                speed = 0.0
                value = raw
                if valid and prev is not None and old['counts'][index] > 0:
                    delta = hypot(raw[0]-prev[0],raw[1]-prev[1])
                    if delta > max(.45, 8*dt):
                        # A discontinuity is missing evidence, not a fast strike.
                        valid, count = False, 0
                        value = prev
                    elif delta <= self.deadband:
                        value = prev
                    elif support < .12:
                        # Let the displayed estimate settle, but generate zero velocity.
                        value = (prev[0]+.12*(raw[0]-prev[0]),prev[1]+.12*(raw[1]-prev[1]))
                    else:
                        alpha = 1-exp(-dt/.055)
                        value = (prev[0]+alpha*(raw[0]-prev[0]), prev[1]+alpha*(raw[1]-prev[1]))
                        if count >= 3:
                            speed = max(0, hypot(value[0]-prev[0],value[1]-prev[1])-self.deadband)/dt
                if not valid:
                    count = 0
                normalized.append(value)
                counts.append(count)
                # Hide unstable/missing limbs; never draw a fabricated extrapolation.
                points.append((draw_anchor[0]+value[0]*scale,draw_anchor[1]+value[1]*scale,confidence if valid else 0.0))
                if index in (9,10,15,16):
                    speeds[index] = speed if valid and support >= .12 else 0.0
                    limb_motion[index] = support if speeds[index] > 0 else 0.0
            self.tracks[person.track_id] = dict(time=timestamp,points=normalized,counts=counts,scale=scale,center=person.center,height=person.height,anchor=draw_anchor,visible=visible)
            output.append(replace(person,keypoints=points,limb_speeds=speeds,limb_motion=limb_motion,body_scale=scale,pose_reliable=reliable))
        self.tracks = {key:value for key,value in self.tracks.items() if key in active}
        return output
