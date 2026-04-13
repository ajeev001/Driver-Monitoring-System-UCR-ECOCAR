import rtmaps.core as rt
import rtmaps.types
from rtmaps.base_component import BaseComponent
from statemachine import StateMachine, State
from dataclasses import dataclass
import time

class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self)

    def Dynamic(self):
        # Inputs
        self.add_input("dms_gaze_status", rtmaps.types.INTEGER64)
        self.add_input("dms_confidence", rtmaps.types.INTEGER64)
        self.add_input("hands_off_wheel", rtmaps.types.INTEGER64)
        self.add_input("dms_requested", rtmaps.types.INTEGER32) 
        self.add_input("current_speed", rtmaps.types.ANY)    

        # Outputs
        self.add_output("state", rtmaps.types.INTEGER64)
        self.add_output("cav_lock", rtmaps.types.INTEGER64)

        # Properties...
        self.add_property("dms_confidence_threshold", 4.0, rtmaps.types.FLOAT64)
        self.add_property("speed_threshold", 5.0, rtmaps.types.FLOAT64)
        self.add_property("attention_loss_warning_time", 5.0, rtmaps.types.FLOAT64)
        self.add_property("attention_loss_critical_time", 15.0, rtmaps.types.FLOAT64)
        self.add_property("hands_off_warning_time", 5.0, rtmaps.types.FLOAT64)
        self.add_property("hands_off_critical_time", 15.0, rtmaps.types.FLOAT64)
        self.add_property("cav_lock_duration", 30.0, rtmaps.types.FLOAT64)

    def Birth(self):
        self.DMS_machine = DMSStateMachine(
            dms_confidence_threshold=float(self.properties["dms_confidence_threshold"].data),
            attention_loss_warning_time=float(self.properties["attention_loss_warning_time"].data),
            attention_loss_critical_time=float(self.properties["attention_loss_critical_time"].data),
            hands_off_warning_time=float(self.properties["hands_off_warning_time"].data),
            hands_off_critical_time=float(self.properties["hands_off_critical_time"].data),
            cav_lock_duration=float(self.properties["cav_lock_duration"].data),
            speed_threshold=float(self.properties["speed_threshold"].data)
        )
        print("DMS State Controller initialized")

    def Core(self):
        inputs = DMSInputs(
            dms_gaze_status=self.inputs["dms_gaze_status"].ioelt.data if self.inputs["dms_gaze_status"].ioelt is not None else None,
            dms_confidence=self.inputs["dms_confidence"].ioelt.data if self.inputs["dms_confidence"].ioelt is not None else None,
            hands_off_wheel=bool(self.inputs["hands_off_wheel"].ioelt.data) if self.inputs["hands_off_wheel"].ioelt is not None else None,
            dms_requested=bool(self.inputs["dms_requested"].ioelt.data) if self.inputs["dms_requested"].ioelt is not None else None,
            current_speed=self.inputs["current_speed"].ioelt.data if self.inputs["current_speed"].ioelt is not None else None,
        )
        self.DMS_machine.update(inputs)
        if self.DMS_machine.current_state.id is not None:
            self.outputs["state"].write(self.DMS_machine.current_state.value)
        self.outputs["cav_lock"].write(1 if self.DMS_machine.cav_lock else 0)

    def Death(self):
        print("DMS State Controller terminated")

@dataclass
class DMSInputs:
    dms_gaze_status: int
    dms_confidence: float
    hands_off_wheel: bool
    dms_requested: bool
    current_speed: float

class DMSStateMachine(StateMachine):
    """Driver Monitoring System state machine"""
    off = State(initial=True, value=0)  # Renamed from inactive
    active = State(value=1)  # Keeping original value (skipping 1 which was standby)
    warning1 = State(value=2)
    warning2 = State(value=3)

    transitions = (
        off.to(active, cond="is_dms_requested")
        | active.to(off, cond="not is_dms_requested")
        | active.to(warning1, cond="is_attention_lost_5s or is_hands_off_5s")
        | warning1.to(active, cond="is_driver_reengaged")
        | warning1.to(off, cond="not is_dms_requested")
        | warning1.to(warning2, cond="is_attention_lost_15s or is_hands_off_15s")
        | warning2.to(off, cond="is_cav_lock_complete or not is_dms_requested")
    )

    def __init__(self, dms_confidence_threshold: float, 
                 attention_loss_warning_time: float, attention_loss_critical_time: float,
                 hands_off_warning_time: float, hands_off_critical_time: float,
                 cav_lock_duration: float, speed_threshold: float, *args, **kwargs):
        self.inputs = None
        self.attention_loss_start = None
        self.hands_off_start = None
        self.cav_lock_start_time = None
        self.dms_confidence_threshold = dms_confidence_threshold
        self.attention_loss_warning_time = attention_loss_warning_time
        self.attention_loss_critical_time = attention_loss_critical_time
        self.hands_off_warning_time = hands_off_warning_time
        self.hands_off_critical_time = hands_off_critical_time
        self.cav_lock_duration = cav_lock_duration
        self.cav_lock = False
        self.speed_threshold = speed_threshold
        super().__init__(*args, **kwargs, allow_event_without_transition=True)

    def update(self, inputs: DMSInputs):
        self.inputs = inputs
        self.transitions()

    def is_dms_requested(self):
        if not self.inputs:
            return False
        if not self.inputs.dms_requested:
            self.attention_loss_start = None
            self.hands_off_start = None
        return self.inputs.dms_requested

    def is_attention_lost_5s(self):
        if not self.inputs:
            return False
        return self.get_attention_loss_time() >= self.attention_loss_warning_time

    def is_attention_lost_15s(self):
        if not self.inputs:
            return False
        return self.get_attention_loss_time() >= self.attention_loss_critical_time

    def is_hands_off_5s(self):
        if not self.inputs:
            return False
        return self.get_hands_off_time() >= self.hands_off_warning_time

    def is_hands_off_15s(self):
        if not self.inputs:
            return False
        return self.get_hands_off_time() >= self.hands_off_critical_time

    def is_driver_reengaged(self):
        if not self.inputs:
            return False
        return (self.inputs.dms_gaze_status == 1 and 
                self.inputs.dms_confidence >= self.dms_confidence_threshold and 
                (not self.inputs.hands_off_wheel or self.inputs.current_speed <= self.speed_threshold))

    def is_cav_lock_complete(self):
        if not self.inputs:
            return False
        if self.is_driver_reengaged():
            if self.cav_lock_start_time is None:
                self.cav_lock_start_time = time.time()
                self.cav_lock = True
        if self.cav_lock_start_time is not None:
            elapsed = time.time() - self.cav_lock_start_time
            if elapsed >= self.cav_lock_duration:
                self.cav_lock = False
                self.cav_lock_start_time = None
                self.attention_loss_start = None
                self.hands_off_start = None
                return True
            else:
                self.cav_lock = True
                return False
        self.cav_lock = False
        return False

    def get_attention_loss_time(self):
        if not self.inputs.dms_gaze_status:
            self.attention_loss_start = None
            return 0.0
        if self.inputs.dms_gaze_status != 1 and self.inputs.dms_confidence >= self.dms_confidence_threshold:
            if self.attention_loss_start is None:
                self.attention_loss_start = time.time()
            return time.time() - self.attention_loss_start
        else:
            self.attention_loss_start = None
            return 0.0

    def get_hands_off_time(self):
        if not self.inputs.hands_off_wheel or self.inputs.current_speed <= self.speed_threshold:
            self.hands_off_start = None
            return 0.0
        if self.inputs.hands_off_wheel:
            if self.hands_off_start is None:
                self.hands_off_start = time.time()
            return time.time() - self.hands_off_start
        else:
            self.hands_off_start = None
            return 0.0

