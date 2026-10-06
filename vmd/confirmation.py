"""Time supported interactions, then confirm with fresh matching depth."""
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from math import isfinite
import uuid

from .spatial import box_overlap


@dataclass
class PendingInteraction:
    since: float
    last: float
    last_positive: float
    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    positive: bool = True
    standard_positive: bool = False
    supported_seconds: float = 0.0
    depth_since: float | None = None
    depth_last: float | None = None
    depth_samples: int = 0
    depth_status: str = 'waiting'
    reported: bool = False
    possible_reported: bool = False
    samples: deque = field(default_factory=deque)
    strike_bouts: deque = field(default_factory=deque)
    strike_since: float | None = None
    strike_last: float | None = None
    strike_motion_last: float | None = None
    fight_seconds: float = 0.0
    timer_confirmed: bool = False
    timer_confirmed_slow: bool = False
    fast_depth_after: float = float('-inf')
    contact_depth: deque = field(default_factory=deque)
    directed_depth: deque = field(default_factory=deque)
    credited_contacts: set = field(default_factory=set)


class FightConfirmation:
    minimum_overlap = .05
    depth_max_gap = 2.5
    interaction_grace = 1.5  # Retain the pair between distinct strikes, not across long inactivity.

    def __init__(self, rules):
        self.rules = rules
        self.required_seconds = rules.fight_confirmation_seconds
        self.pending = {}
        self.cooldowns = {}
        self.last_time = None
        self.last_depth_time = float('-inf')
        self.separated_pairs = {}
        self.torso_statuses = {}
        self.paused_episodes = {}

    @staticmethod
    def _clear_depth(state, status):
        state.depth_since = state.depth_last = None
        state.depth_samples = 0
        state.depth_status = status
        state.strike_since = state.strike_last = None
        state.strike_motion_last = None
        state.fight_seconds = 0.0
        state.timer_confirmed = False
        state.timer_confirmed_slow = False
        state.fast_depth_after = float('-inf')
        state.contact_depth.clear()
        state.directed_depth.clear()
        state.credited_contacts.clear()
        state.samples.clear()
        state.strike_bouts.clear()
        state.supported_seconds = 0.0

    def depth_unavailable(self):
        for state in self.pending.values():
            self._clear_depth(state, 'unavailable')

    def observe_depth(self, source_time, observations, current_time):
        if (not isfinite(source_time) or source_time <= self.last_depth_time
                or not 0 <= current_time-source_time <= self.depth_max_gap):
            return
        self.last_depth_time = source_time
        pairs = {tuple(sorted(item['tracks'])): item for item in observations}
        # A depth sample can arrive before fast motion creates a pending episode.
        # Retain negative evidence briefly; never credit pre-episode positive depth.
        for pair, item in pairs.items():
            self.torso_statuses[pair] = (source_time, item.get('torso_status', item['status']))
            if item.get('status') == 'separated':
                self.separated_pairs[pair] = source_time
            else:
                self.separated_pairs.pop(pair, None)
        for pair, state in self.pending.items():
            if source_time < state.since:
                continue
            status = pairs.get(pair, {}).get('status', 'uncertain')
            state.contact_depth.append((source_time, status))
            contacts = pairs.get(pair, {}).get('contacts')
            torso_status = pairs.get(pair, {}).get('torso_status', status)
            state.directed_depth.append((source_time, contacts, torso_status))
            if torso_status == 'separated':
                # Local arm depth cannot preserve fast evidence across a known
                # torso separation. Rebase any retained timer to accepted slow
                # contact, never to an earlier fast-motion origin.
                state.fast_depth_after = source_time
                state.samples.clear()
                state.strike_bouts.clear()
                state.supported_seconds = 0.0
                state.standard_positive = False
                slow_times = [stamp for _, _, stamp in state.credited_contacts
                              if state.strike_since is not None and state.strike_since <= stamp <= source_time]
                if slow_times:
                    state.strike_since = min(slow_times)
                    state.fight_seconds = max(0.0, (state.strike_last or state.strike_since)-state.strike_since)
                else:
                    state.strike_since = state.strike_last = state.strike_motion_last = None
                    state.fight_seconds = 0.0
                    state.timer_confirmed_slow = False
                state.timer_confirmed = (state.timer_confirmed_slow
                                         and state.fight_seconds+1e-9 >= self.required_seconds)
            while state.contact_depth and source_time-state.contact_depth[0][0] > 8:
                state.contact_depth.popleft()
            while state.directed_depth and source_time-state.directed_depth[0][0] > 8:
                state.directed_depth.popleft()
            if (status == 'uncertain' and state.depth_last is not None
                    and source_time-state.depth_last <= self.depth_max_gap):
                # Brief ambiguity supplies no support and cannot confirm now.
                # It need not erase earlier valid samples before their usual
                # freshness deadline. Known separation still clears everything.
                state.depth_status = status
                continue
            if status != 'compatible':
                self._clear_depth(state, status)
                state.contact_depth.append((source_time, status))
                state.directed_depth.append((source_time, contacts, torso_status))
                if status == 'separated':
                    state.samples.clear()
                    state.strike_bouts.clear()
                continue
            if state.depth_last is None or source_time-state.depth_last > self.depth_max_gap:
                state.depth_since = source_time
                state.depth_samples = 0
            state.depth_last = source_time
            state.depth_samples += 1
            state.depth_status = 'compatible'

    def update(self, assessment, people, timestamp, camera_motion):
        rules = self.rules
        dt = timestamp-self.last_time if self.last_time is not None else 0
        valid_time = 0 < dt <= rules.max_gap_seconds
        if not valid_time or camera_motion > rules.camera_speed:
            for pair, state in self.pending.items():
                if state.possible_reported:
                    self.cooldowns[pair] = timestamp
            self.pending.clear()
            self.paused_episodes.clear()
            if self.last_time is not None or camera_motion > rules.camera_speed:
                self.separated_pairs.clear()
                self.torso_statuses.clear()
        self.last_time = timestamp
        self.cooldowns = {pair: time for pair, time in self.cooldowns.items()
                          if timestamp-time < rules.cooldown_seconds}
        self.paused_episodes = {pair: saved for pair, saved in self.paused_episodes.items()
                               if timestamp < saved[0]}
        result = replace(assessment, state='observing', event_type='possible_fight', trigger=False,
                         events=[], candidates=[], signals={**assessment.signals, 'active_fight_pairs': [],
                         'confirmation_seconds': 0.0, 'required_seconds': self.required_seconds, 'pending_pairs': [],
                         'fight_timer_seconds': 0.0, 'fight_timer_required_seconds': self.required_seconds,
                         'fight_timer_started': False,
                         'fight_timer_state': 'idle', 'fight_timer_reason': 'Waiting for a supported strike'})
        if camera_motion > rules.camera_speed:
            result.state = 'camera_moving'
            return result
        counts = Counter(person.track_id for person in people)
        tracks = {p.track_id: p for p in people if p.track_id >= 0 and counts[p.track_id] == 1}
        self.separated_pairs = {pair: stamp for pair, stamp in self.separated_pairs.items()
                                if 0 <= timestamp-stamp <= self.depth_max_gap
                                and all(track in tracks for track in pair)}
        self.torso_statuses = {pair: sample for pair, sample in self.torso_statuses.items()
                               if 0 <= timestamp-sample[0] <= self.depth_max_gap
                               and all(track in tracks for track in pair)}
        visible, nearby, choices, progress, events, rejected_pairs = set(), set(), [], [], [], []
        for candidate in assessment.candidates:
            pair = candidate.pair
            if not pair or pair[0] == pair[1] or any(track not in tracks for track in pair):
                continue
            a, b = (tracks[track] for track in pair)
            overlap = box_overlap(a.box, b.box)
            poses_ok = a.pose_reliable and b.pose_reliable
            if poses_ok:
                nearby.add(pair)
            flow = min(a.local_flow or 0, b.local_flow or 0)
            raw_positive = (valid_time and poses_ok
                            and candidate.signals.get('candidate', False))
            state = self.pending.get(pair)
            separated = pair in self.separated_pairs
            # A hand/foot can reach a body without the two detection boxes
            # overlapping. The raw rule already checks anatomical contact.
            geometry_ok = not separated
            if not geometry_ok:
                rejected_pairs.append(pair)
            standard_positive = (raw_positive and candidate.signals.get('body_contact', overlap >= self.minimum_overlap)
                                 and geometry_ok and self.torso_statuses.get(pair, (None, None))[1] != 'separated')
            slow_score = candidate.signals.get('slow_punch_score', 0.0)
            slow_positive = (valid_time and poses_ok and geometry_ok
                             and candidate.signals.get('slow_punch_activity', False)
                             and slow_score >= rules.threshold)
            # An approach opens the depth window; it is not a review warning.
            # A completed slow punch can be on withdrawal, outside the body.
            positive = standard_positive or slow_positive
            if not valid_time or not poses_ok:
                if state and state.possible_reported:
                    self.cooldowns[pair] = timestamp
                self.pending.pop(pair, None)
                state = None
            elif state and timestamp-state.last_positive > self.interaction_grace:
                # A quiet pause resets evidence, not the ability to observe a
                # resumed fight. Reuse the review record while the same pair
                # stays visible, avoiding duplicate incidents/response calls.
                if state.possible_reported or state.reported:
                    self.paused_episodes[pair] = (state.last_positive+rules.cooldown_seconds, state)
                self.pending.pop(pair, None)
                state = None
            if positive and state is None and pair not in self.cooldowns:
                state = PendingInteraction(timestamp, timestamp, timestamp)
                resumed = self.paused_episodes.pop(pair, None)
                if resumed and timestamp < resumed[0]:
                    state.episode_id = resumed[1].episode_id
                    state.possible_reported = resumed[1].possible_reported
                    state.reported = resumed[1].reported
                self.pending[pair] = state
            supported, confirmed = 0.0, False
            bilateral_counts = {track: 0 for track in pair}
            depth_punch_counts = {track: 0 for track in pair}
            slow_reciprocal = slow_evidence = motion_ok = timer_active = awaiting_contact_depth = False
            new_strikes = []
            fresh_depth_samples = 0
            timer_state, timer_reason = 'idle', 'Waiting for a supported strike'
            if state:
                visible.add(pair)
                if separated and state.depth_status != 'separated':
                    self._clear_depth(state, 'separated')
                if not geometry_ok:
                    state.samples.clear()
                    state.strike_bouts.clear()
                if (state.depth_status == 'separated' and not separated):
                    self._clear_depth(state, 'stale')
                if standard_positive:
                    if not state.samples or (not state.standard_positive and timestamp-state.samples[-1][0] > rules.signal_grace_seconds):
                        state.strike_bouts.append(timestamp)
                    grappling_now = bool(candidate.signals.get('grappling'))
                    # Integrate a grapple only between two observed positive
                    # endpoints. The short-strike credit cap must not make
                    # the configured duration impossible at a valid 2 FPS rate.
                    grapple_interval = dt if (grappling_now and state.standard_positive and state.samples
                                               and state.samples[-1][0] == state.last
                                               and state.samples[-1][2]) else 0.0
                    state.samples.append((timestamp, min(dt, .25), grappling_now, grapple_interval))
                if positive:
                    state.last_positive = timestamp
                state.last, state.positive = timestamp, positive
                state.standard_positive = standard_positive
                while state.samples and timestamp-state.samples[0][0] > self.required_seconds+1:
                    state.samples.popleft()
                while state.strike_bouts and timestamp-state.strike_bouts[0] > self.required_seconds+1:
                    state.strike_bouts.popleft()
                state.supported_seconds = sum(sample[1] for sample in state.samples)
                if state.depth_last is not None and timestamp-state.depth_last > self.depth_max_gap:
                    self._clear_depth(state, 'stale')
                depth_seconds = state.depth_last-state.depth_since if state.depth_last is not None else 0.0
                fresh_depth_samples = sum(status == 'compatible' and 0 <= timestamp-stamp <= self.depth_max_gap
                                          for stamp, status in state.contact_depth)
                # The local arm-depth exception belongs only to the directed
                # slow strike that identifies that arm. A different fast wrist
                # or a grapple must keep its original torso-depth gate.
                fast_depth_ok = (bool(state.directed_depth) and state.directed_depth[-1][2] == 'compatible'
                    and sum(status == 'compatible' and stamp > state.fast_depth_after
                            and 0 <= timestamp-stamp <= self.depth_max_gap
                            for stamp, _, status in state.directed_depth) >= 2)
                supported = min(timestamp-state.since, depth_seconds)
                reach_depth_ok = overlap >= self.minimum_overlap or state.depth_status == 'compatible'
                slow_striking = (slow_positive and candidate.signals.get('slow_punch_striking', False)
                                 and candidate.signals.get('slow_punch_striking_score', slow_score) >= rules.threshold)
                strikes, unresolved_strikes = [], []
                for strike in candidate.signals.get('slow_punch_strikes', []):
                    if (strike['actor'] not in pair or strike.get('two_handed')
                            or strike.get('score', slow_score) < rules.threshold
                            or not state.since <= strike['contact'] <= timestamp):
                        continue
                    match = min(state.contact_depth, key=lambda sample: abs(sample[0]-strike['contact']), default=None)
                    if not match or abs(match[0]-strike['contact']) > 1e-6:
                        unresolved_strikes.append(strike)
                        continue
                    details = next((contacts for stamp, contacts, _ in state.directed_depth
                                    if abs(stamp-strike['contact']) <= 1e-6), None)
                    # Model observations carry directed surfaces: a different
                    # person's hand or an old torso match cannot validate this arm.
                    contact_status = match[1]
                    if details is not None:
                        matching = [item for item in details if item.get('actor') == strike['actor']
                                    and item.get('wrist') == strike['wrist']
                                    and item.get('target_track') == strike.get('target_track', next(track for track in pair if track != strike['actor']))
                                    and item.get('contact_kind', 'body') == strike.get('contact_kind', 'body')
                                    and item.get('target_joint') == strike.get('target_joint')]
                        contact_status = matching[0]['status'] if matching else 'uncertain'
                    if contact_status == 'compatible':
                        strikes.append(strike)
                    elif contact_status == 'uncertain' and (details is None or matching):
                        unresolved_strikes.append(strike)
                current_contacts = candidate.signals.get('slow_punch_striking_contacts', strikes)
                def current_contact(strike):
                    return (timestamp-strike['contact'] <= self.interaction_grace
                        and any(contact.get('score', slow_score) >= rules.threshold
                                and all(contact.get(key) == strike.get(key) for key in ('actor', 'wrist', 'contact'))
                                for contact in current_contacts))
                current_strikes = [strike for strike in strikes if current_contact(strike)]
                state.credited_contacts = {key for key in state.credited_contacts if timestamp-key[2] <= 8}
                new_strikes = [strike for strike in strikes
                               if timestamp-strike['contact'] <= self.interaction_grace
                               and (strike['actor'], strike['wrist'], strike['contact']) not in state.credited_contacts]
                state.credited_contacts.update((s['actor'], s['wrist'], s['contact']) for s in new_strikes)
                # Depth can arrive after a one-frame contact has withdrawn.
                # Credit that original observation once, never its arrival time.
                slow_evidence = bool(slow_striking and current_strikes or new_strikes)
                strike_now = standard_positive or slow_evidence
                observed_time = (timestamp if standard_positive or slow_striking and current_strikes
                                 else max((s['contact'] for s in new_strikes), default=timestamp))
                if state.strike_motion_last is not None and observed_time-state.strike_motion_last > self.interaction_grace:
                    state.strike_since = state.strike_last = None
                    state.strike_motion_last = None
                    state.fight_seconds = 0.0
                    state.timer_confirmed = False
                    state.timer_confirmed_slow = False
                if (strike_now and reach_depth_ok and state.strike_since is None
                        and state.depth_status not in {'stale', 'unavailable'}):
                    state.strike_since = timestamp if standard_positive else min(s['contact'] for s in current_strikes+new_strikes)
                # Count source time while this supported interaction continues.
                # Withdrawals freeze the displayed clock; only resumed evidence
                # within the bounded grace bridges that gap. Silence never confirms.
                timer_active = strike_now and reach_depth_ok and state.strike_since is not None
                if timer_active:
                    state.strike_last = max(state.strike_last or observed_time, observed_time)
                    state.strike_motion_last = max(state.strike_motion_last or observed_time, observed_time)
                    state.fight_seconds = max(state.fight_seconds, observed_time-state.strike_since)
                elif (state.strike_since is not None and slow_striking
                      and state.depth_last is not None and state.depth_status in {'compatible', 'uncertain'}
                      and any(current_contact(strike) for strike in unresolved_strikes)):
                    # Brief uncertain contact depth freezes the clock, while
                    # continuing supported punches retain motion continuity.
                    # It cannot start a timer or outlive depth freshness.
                    state.strike_motion_last = timestamp
                    awaiting_contact_depth = True
                for cycle in candidate.signals.get('bilateral_punch_cycles', []):
                    actor = cycle['actor']
                    # Evidence is tied to this pair and episode. After a depth
                    # break, old punches cannot satisfy the fresh depth window.
                    if (actor in bilateral_counts and state.since <= cycle['contact'] <= cycle['time'] <= timestamp
                            and timestamp-cycle['time'] <= 8.0):
                        bilateral_counts[actor] += 1
                        if (state.depth_since is not None
                                and state.depth_since <= cycle['contact'] <= state.depth_last):
                            depth_punch_counts[actor] += 1
                strikes = [strike for strike in strikes if state.strike_since is not None
                           and state.strike_since <= strike['contact']]
                # Reciprocal directed strikes can qualify before their full
                # withdrawals complete. A simultaneous two-hand embrace alone
                # is one action, not an ongoing exchange of punches.
                slow_reciprocal = (len({strike['actor'] for strike in strikes}) == 2
                                  and max(s['contact'] for s in strikes)-min(s['contact'] for s in strikes) >= .5)
                cycle_times = candidate.signals.get('strike_cycle_times', [])
                cycles = sum(stamp >= max(state.samples[0][0], state.strike_since or state.since)
                             for stamp in cycle_times) if state.samples else 0
                # Distinct strikes occupy only a few contact frames. Do not
                # require the continuous-contact duty used for grappling.
                repeated = (len(state.strike_bouts) >= 2 and state.supported_seconds >= .2
                            or cycles >= 1 and state.supported_seconds >= .25)
                grappling = sum(sample[3] for sample in state.samples) >= self.required_seconds-1e-6
                # Continuous fast limb estimates alone are not repeated strikes.
                # Repeated bursts may contain short pauses, but need substantial motion evidence.
                motion_ok = (standard_positive and (grappling or repeated) and fast_depth_ok
                             or slow_evidence and slow_reciprocal)
                confirmed = (timer_active and motion_ok and state.fight_seconds+1e-9 >= self.required_seconds
                             and fresh_depth_samples >= 2 and state.depth_status == 'compatible')
                state.timer_confirmed |= confirmed
                state.timer_confirmed_slow |= bool(confirmed and slow_evidence and slow_reciprocal)
                if state.strike_since is not None:
                    timer_state = 'counting' if timer_active else 'paused'
                    timer_reason = 'Supported striking continues' if timer_active else 'Waiting for supported movement to resume'
                    if state.timer_confirmed and state.depth_status == 'compatible':
                        timer_state, timer_reason = 'confirmed', 'Fight detected with matching depth'
                    elif awaiting_contact_depth:
                        timer_state, timer_reason = 'waiting_depth', 'Checking depth at the striking contact frame'
                    elif state.fight_seconds+1e-9 >= self.required_seconds and timer_active:
                        if (state.depth_status != 'compatible' or fresh_depth_samples < 2
                                or standard_positive and not (slow_evidence and slow_reciprocal) and not fast_depth_ok):
                            timer_state, timer_reason = 'waiting_depth', 'Waiting for two fresh matching depth samples'
                        elif not motion_ok:
                            timer_state, timer_reason = 'waiting_strikes', 'Waiting for a continuing exchange of supported strikes'
                elif (standard_positive or slow_striking) and state.depth_status in {'stale', 'unavailable'}:
                    timer_state, timer_reason = 'waiting_depth', 'Depth evidence expired; waiting for fresh matching depth'
                elif slow_striking and not strikes:
                    timer_state, timer_reason = 'waiting_depth', 'Checking depth at the striking contact frame'
                if timer_active and state.depth_status == 'uncertain' and not state.depth_samples:
                    timer_state, timer_reason = 'waiting_depth', 'Waiting for reliable matching depth'
            blockers = list(candidate.signals.get('blockers', []))
            if slow_positive:
                blockers = [item for item in blockers if not item.startswith((
                    'No sufficiently fast supported strike', 'Interaction score',
                    'Not enough image motion', 'No limb reaches', 'Collecting repeated strikes'))]
                if not slow_reciprocal:
                    blockers.append('Waiting for body-directed strikes from both people')
            reach_depth_ok = overlap >= self.minimum_overlap or bool(state and state.depth_status == 'compatible')
            if not reach_depth_ok:
                blockers.append('Waiting for matching depth to support a limb reach between separate person boxes')
            if separated:
                blockers.append('Matching-frame relative depth places these people apart')
            if state and state.depth_status != 'compatible':
                blockers.append(f'Waiting for reliable matching depth: {state.depth_status}')
            if state and not confirmed:
                blockers.append(f'Fight timer: {state.fight_seconds:.1f} / {self.required_seconds:g} seconds')
            signals = {**candidate.signals, 'box_overlap': round(overlap, 3), 'both_people_flow': round(flow, 3),
                       'required_seconds': self.required_seconds, 'confirmation_seconds': round(supported, 2),
                       'fight_timer_seconds': round(state.fight_seconds, 3) if state else 0.0,
                       'fight_timer_required_seconds': self.required_seconds,
                       'fight_timer_started': bool(state and state.strike_since is not None),
                       'fight_timer_state': timer_state, 'fight_timer_reason': timer_reason,
                       'motion_supported_seconds': round(state.supported_seconds, 2) if state else 0.0,
                       'strike_bouts': len(state.strike_bouts) if state else 0,
                       'bilateral_punch_counts': bilateral_counts,
                       'depth_punch_counts': depth_punch_counts,
                       'review_seconds': round(timestamp-state.since, 2) if state else 0.0,
                       'depth_status': 'separated' if separated else state.depth_status if state else 'waiting',
                       'depth_samples': state.depth_samples if state else 0,
                       'fresh_depth_samples': fresh_depth_samples,
                       'depth_source_time': state.depth_last if state else None,
                       'possible_required_seconds': rules.hold_seconds,
                       'blockers': [] if confirmed else blockers}
            if state:
                signals['episode_id'] = state.episode_id
            # A saved warning is not evidence that current movement is still a fight.
            # Persistence is a minimum observation span. Repeated supported
            # strikes need not remain in contact during their withdrawals.
            review_motion = bool(state and (state.supported_seconds+1e-9 >= rules.hold_seconds
                                 or len(state.strike_bouts) >= 2 and state.supported_seconds >= .2))
            possible = bool(state and reach_depth_ok and timer_active and (
                standard_positive and (not candidate.signals.get('grappling') or state.possible_reported
                    or timestamp-state.since+1e-9 >= rules.hold_seconds and review_motion)
                or slow_evidence))
            event_type = 'fight' if confirmed else 'possible_fight'
            reasons = list(candidate.reasons)
            if slow_positive:
                reasons += ['Image-supported approach, body contact and withdrawal are tracked separately for each person']
            if confirmed:
                reasons += ['Body-directed strikes from both people with matching relative depth' if slow_positive and slow_reciprocal
                            else 'Repeated strikes or sustained grappling with matching relative depth',
                            f'The same interaction spans at least {self.required_seconds:g} seconds; review required']
            elif possible:
                reasons += ['Brief supported contact motion; fight is not confirmed']
            score = max(candidate.score, slow_score) if slow_positive else candidate.score
            if new_strikes:
                score = max(score, *(strike.get('score', slow_score) for strike in new_strikes))
            choice = replace(candidate, state='fight_detected' if confirmed else 'possible_fight' if possible else 'checking_interaction' if state and geometry_ok else 'observing',
                             score=score,
                             event_type=event_type, reasons=reasons, signals=signals, trigger=False, events=[], candidates=[])
            if confirmed and not state.reported:
                choice.trigger = True
                state.reported = True
                events.append(replace(choice))
            elif possible and not state.possible_reported and not state.reported:
                choice.trigger = True
                state.possible_reported = True
                events.append(replace(choice))
            # A confirmed episode never emits a second provisional record or downgrades storage.
            if state and state.reported and not confirmed and geometry_ok:
                choice.state = 'checking_interaction'
            choices.append(choice)
            if state:
                progress.append({'tracks': list(pair), 'seconds': round(state.fight_seconds, 2),
                                 'required_seconds': self.required_seconds, 'depth_status': state.depth_status,
                                 'state': choice.state})
        for pair, old in self.pending.items():
            if pair not in visible:
                rejected_pairs.append(pair)
                if old.possible_reported:
                    self.cooldowns[pair] = timestamp
        self.pending = {pair: state for pair, state in self.pending.items() if pair in visible}
        # A depth veto invalidates evidence, not the durable incident identity.
        # Resumption copies only ID/report flags into an empty observation window.
        self.paused_episodes = {pair: saved for pair, saved in self.paused_episodes.items()
                               if pair in nearby}
        if choices:
            result = max(choices, key=lambda item: (item.state == 'fight_detected', item.state == 'possible_fight',
                         item.state == 'checking_interaction', item.signals['confirmation_seconds'], item.score))
        result.events = events
        result.signals = {**result.signals, 'pending_pairs': progress, 'required_seconds': self.required_seconds,
                          'rejected_fight_pairs': rejected_pairs,
                          'active_fight_pairs': [item.pair for item in choices if item.state == 'fight_detected']}
        return result
